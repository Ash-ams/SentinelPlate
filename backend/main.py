from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile
import json
import shutil
import logging
from typing import Optional, Dict, Any, List

from backend.models.schemas import (
    PipelineConfig, PipelineResponse, DatasetListResponse,
    TollEntry, TollSimulationResponse, TollLogsResponse,
    VideoMotionResponse
)
try:
    from backend.services.ivp_pipeline import ALPRPipeline
except ImportError:
    try:
        from services.ivp_pipeline import ALPRPipeline
    except ImportError:
        class ALPRPipeline:  # type: ignore
            """Fallback ALPR pipeline if ivp_pipeline.py is not yet loaded."""
            def process(self, image_path: str, config: dict = None, annotation_path: str = None):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="ALPRPipeline service is not currently available in backend.services.ivp_pipeline."
                )

from fastapi.staticfiles import StaticFiles
from backend.services.toll_service import TollService
from backend.services.dataset_service import DatasetService
from backend.services.video_motion import VideoMotionEngine
from backend.services.benchmark_service import BenchmarkService

app = FastAPI(
    title="ALPR - Automatic License Plate Recognition",
    description="Production-grade ALPR system with 7-stage IVP pipeline",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
STATIC_IMAGES_DIR = os.path.join(DATASET_DIR, "images") if os.path.exists(os.path.join(DATASET_DIR, "images")) else os.path.abspath("dataset/images")

# Mount /dataset/images as /api/static-images
if os.path.exists(STATIC_IMAGES_DIR):
    app.mount("/api/static-images", StaticFiles(directory=STATIC_IMAGES_DIR), name="static_images")

@app.get("/api/dataset/image/{filename}")
async def get_dataset_image(filename: str):
    """Serve dataset images by filename."""
    from fastapi.responses import FileResponse
    path = os.path.join(STATIC_IMAGES_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail=f"Image {filename} not found.")

def ensure_data_uri(b64_str: Optional[str]) -> str:
    """Guarantee that base64 string is formatted as data:image/png;base64,... without double-prefix."""
    if not b64_str:
        return ""
    b64_clean = str(b64_str).strip()
    if b64_clean.startswith("data:image/"):
        return b64_clean
    return f"data:image/png;base64,{b64_clean}"

pipeline = ALPRPipeline()
toll_service = TollService(os.path.join(BASE_DIR, "toll_parking.db"))
dataset_service = DatasetService(DATASET_DIR)
video_motion_engine = VideoMotionEngine()
benchmark_service = BenchmarkService(DATASET_DIR)

# Image hash matching cache for uploaded images to auto-match XML ground truth
_DATASET_HASH_MAP: Dict[str, str] = {}


def _get_dataset_hash_map() -> Dict[str, str]:
    global _DATASET_HASH_MAP
    if not _DATASET_HASH_MAP and os.path.exists(dataset_service.images_dir):
        import hashlib
        for fname in os.listdir(dataset_service.images_dir):
            p = os.path.join(dataset_service.images_dir, fname)
            if os.path.isfile(p):
                try:
                    with open(p, "rb") as fp:
                        h = hashlib.md5(fp.read()).hexdigest()
                        _DATASET_HASH_MAP[h] = fname
                except Exception:
                    pass
    return _DATASET_HASH_MAP


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "ALPR API", "dataset_dir": DATASET_DIR}


@app.post("/api/process-image", response_model=PipelineResponse)
async def process_image(
    file: UploadFile = File(None),
    dataset_filename: Optional[str] = Form(None),
    config: str = Form("{}")
):
    """
    Process an image through the 7-stage ALPR pipeline.
    Either upload a file OR specify a dataset_filename.
    Config is a JSON string of PipelineConfig fields.
    """
    # 1. Parse config JSON into dict, validate with PipelineConfig
    try:
        config_data = json.loads(config) if config else {}
        if not isinstance(config_data, dict):
            raise ValueError("Config must be a JSON dictionary.")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid configuration JSON: {str(e)}"
        )

    try:
        pipeline_cfg = PipelineConfig(**config_data)
        config_dict = pipeline_cfg.model_dump()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid configuration parameters: {str(e)}"
        )

    # 2. Check input source
    if not file and not dataset_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'file' upload or 'dataset_filename' form parameter must be provided."
        )

    temp_file_to_clean: Optional[str] = None
    resolved_filename: str = ""
    annotation_path: Optional[str] = None
    image_path: str = ""

    try:
        if dataset_filename:
            image_path = dataset_service.get_image_path(dataset_filename)
            if not image_path or not os.path.exists(image_path):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Dataset image '{dataset_filename}' not found."
                )
            annotation_path = dataset_service.get_annotation_path(dataset_filename)
            resolved_filename = os.path.basename(image_path)
        else:
            resolved_filename = file.filename or "uploaded_image.png"
            suffix = os.path.splitext(resolved_filename)[1]
            if not suffix:
                suffix = ".png"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_file_to_clean = tmp.name
            image_path = temp_file_to_clean

            # RULE 3: Auto-match uploaded photo to dataset XML if pixels match
            annotation_path = None
            try:
                import hashlib
                with open(image_path, "rb") as fp:
                    upload_md5 = hashlib.md5(fp.read()).hexdigest()
                hash_map = _get_dataset_hash_map()
                matched_fname = hash_map.get(upload_md5)
                if matched_fname:
                    annotation_path = dataset_service.get_annotation_path(matched_fname)
            except Exception:
                annotation_path = None

        # 3. Dynamic lookup for real ALPRPipeline if initially stubbed
        active_pipeline = pipeline
        if active_pipeline is None or active_pipeline.__class__.__module__ != "backend.services.ivp_pipeline":
            try:
                from backend.services.ivp_pipeline import ALPRPipeline as RealALPRPipeline
                active_pipeline = RealALPRPipeline()
            except Exception:
                pass

        if active_pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ALPR pipeline service is not available."
            )

        # 4. Process image
        result = active_pipeline.process(
            image_path=image_path,
            config=config_dict,
            annotation_path=annotation_path
        )

        if isinstance(result, dict):
            if not result.get("filename"):
                result["filename"] = resolved_filename

            # Ensure image_width and image_height are present
            if (not result.get("image_width") or not result.get("image_height")) and image_path and os.path.exists(image_path):
                try:
                    import cv2
                    orig_mat = cv2.imread(image_path)
                    if orig_mat is not None:
                        result["image_width"] = int(orig_mat.shape[1])
                        result["image_height"] = int(orig_mat.shape[0])
                except Exception as ex:
                    logging.warning(f"Could not read image dimensions: {ex}")

            # Ensure original_image is present and valid data URI
            if not result.get("original_image") and image_path and os.path.exists(image_path):
                try:
                    import cv2, base64
                    orig_mat = cv2.imread(image_path)
                    if orig_mat is not None:
                        _, buf = cv2.imencode('.png', orig_mat)
                        result["original_image"] = f"data:image/png;base64,{base64.b64encode(buf).decode('utf-8')}"
                except Exception as ex:
                    logging.warning(f"Could not encode original_image fallback: {ex}")
            elif result.get("original_image"):
                result["original_image"] = ensure_data_uri(result["original_image"])

            # Ensure all stage images are data URIs
            if "stages" in result and isinstance(result["stages"], list):
                for stage in result["stages"]:
                    if isinstance(stage, dict) and "images" in stage and isinstance(stage["images"], dict):
                        for k, v in list(stage["images"].items()):
                            stage["images"][k] = ensure_data_uri(v)

            return PipelineResponse(**result)
        elif hasattr(result, "filename"):
            if not result.filename:
                result.filename = resolved_filename
            return result
        return result

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error during ALPR pipeline processing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ALPR pipeline processing failed: {str(e)}"
        )
    finally:
        if temp_file_to_clean and os.path.exists(temp_file_to_clean):
            try:
                os.remove(temp_file_to_clean)
            except Exception as cleanup_err:
                logging.warning(f"Failed to delete temporary file {temp_file_to_clean}: {cleanup_err}")


@app.get("/api/dataset/list", response_model=DatasetListResponse)
async def list_dataset():
    """List available dataset images with ground truth info."""
    try:
        images = dataset_service.list_images()
        return DatasetListResponse(images=images, total=len(images))
    except Exception as e:
        logging.exception("Error listing dataset images")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list dataset: {str(e)}"
        )


@app.get("/api/dataset/batch-evaluate")
async def batch_evaluate():
    """
    Evaluate entire dataset against Pascal VOC XML ground truth annotations.
    Computes aggregated performance metrics across the 4 syllabus evaluation tables:
      - Table I: Enhancement Benchmark (MSE, PSNR, Shannon Entropy)
      - Table II: Edge Detection & Localization Benchmark (IoU, Precision, Recall, Latency)
      - Table III: Binarization & Character Segmentation Benchmark (tau*, Seg Rate, OCR Accuracy)
      - Table IV: Motion Estimation Benchmark (Search points, Speedup, Prediction PSNR, Latency)
    """
    try:
        results = benchmark_service.run_full_dataset_evaluation(video_motion_engine)
        return results
    except Exception as e:
        logging.exception("Error during dataset batch evaluation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch evaluation failed: {str(e)}"
        )


@app.post("/api/toll/simulate-entry", response_model=TollSimulationResponse)
async def toll_simulate_entry(entry: TollEntry):
    """Simulate a toll/parking entry or exit event."""
    try:
        result = toll_service.simulate_entry(
            entry.plate_number, entry.plate_type, entry.event_type, entry.gate_id
        )
        return result
    except Exception as e:
        logging.exception("Error executing toll simulation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Toll simulation failed: {str(e)}"
        )


@app.get("/api/toll/logs", response_model=TollLogsResponse)
async def toll_logs():
    """Get toll/parking logs and analytics."""
    try:
        return toll_service.get_logs()
    except Exception as e:
        logging.exception("Error fetching toll logs")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch toll logs: {str(e)}"
        )


@app.post("/api/process-video-frames", response_model=VideoMotionResponse)
async def process_video_frames(
    file: UploadFile = File(None),
    dataset_filename: Optional[str] = Form(None),
    algorithm: str = Form("tss"),
    num_frames: int = Form(6)
):
    """
    Unit 8: Digital Video Processing & Motion Estimation Engine.
    Ingests video file OR synthesizes approaching vehicle multi-frame sequence from dataset image.
    Computes Temporal Correlation (MAD, MSE, Pearson rho), Frame Classification (I/P/B),
    Block-Matching Motion Estimation (ESBM vs TSS vs 2D-Log), and triggers ALPR on vehicle halt.
    """
    if not file and not dataset_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'file' upload or 'dataset_filename' parameter must be provided."
        )

    temp_file_to_clean: Optional[str] = None
    frames = []

    try:
        if dataset_filename:
            image_path = dataset_service.get_image_path(dataset_filename)
            if not image_path or not os.path.exists(image_path):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Dataset image '{dataset_filename}' not found."
                )
            frames = video_motion_engine.generate_approaching_sequence(
                image_path, num_frames=max(2, min(num_frames, 12))
            )
        else:
            filename = file.filename or "upload.png"
            suffix = os.path.splitext(filename)[1].lower()
            if not suffix:
                suffix = ".png"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_file_to_clean = tmp.name

            if suffix in [".mp4", ".avi", ".mov", ".mkv"]:
                frames = video_motion_engine.ingest_video_file(temp_file_to_clean, max_frames=num_frames)
            else:
                frames = video_motion_engine.generate_approaching_sequence(
                    temp_file_to_clean, num_frames=max(2, min(num_frames, 12))
                )

        if not frames:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Unable to extract or generate frames for video motion analysis."
            )

        # Run Video Motion Estimation & Smart Trigger
        active_pipeline = pipeline
        result = video_motion_engine.process_sequence(
            frames=frames,
            alpr_pipeline=active_pipeline,
            algorithm=algorithm.lower()
        )
        return VideoMotionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error processing video motion frames")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video motion processing failed: {str(e)}"
        )
    finally:
        if temp_file_to_clean and os.path.exists(temp_file_to_clean):
            try:
                os.remove(temp_file_to_clean)
            except Exception as cleanup_err:
                logging.warning(f"Failed to delete temp video file {temp_file_to_clean}: {cleanup_err}")

