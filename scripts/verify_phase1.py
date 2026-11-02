#!/usr/bin/env python3
"""Phase 1 Automated Verification and Execution Audit Script.

Comprehensive verification of all 7 ALPR IVP pipeline stages,
FastAPI endpoints, and SQLite database persistence.

Usage:
    python scripts/verify_phase1.py
"""

import sys
import os
import json
import sqlite3
import base64
import numpy as np
import cv2

# Ensure project root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from backend.main import app
from backend.services.ivp_pipeline import ALPRPipeline


def is_valid_base64_image(b64_str: str) -> bool:
    """Validate that string is non-empty, valid base64 decodable to an image."""
    if not b64_str or not isinstance(b64_str, str):
        return False
    try:
        raw_data = base64.b64decode(b64_str)
        if len(raw_data) < 16:
            return False
        nparr = np.frombuffer(raw_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        return img is not None and img.size > 0
    except Exception:
        return False


def run_verification():
    print("=" * 80)
    print("      AUTOMATIC LICENSE PLATE RECOGNITION (ALPR) - PHASE 1 AUDIT")
    print("=" * 80)

    results_summary = {}
    client = TestClient(app)
    pipeline = ALPRPipeline()

    cars0_path = os.path.join(BASE_DIR, "dataset", "images", "Cars0.png")
    cars0_xml = os.path.join(BASE_DIR, "dataset", "annotations", "Cars0.xml")
    assert os.path.exists(cars0_path), f"Cars0.png not found at {cars0_path}"
    assert os.path.exists(cars0_xml), f"Cars0.xml not found at {cars0_xml}"

    cars0_bgr = cv2.imread(cars0_path)
    assert cars0_bgr is not None, "Failed to load Cars0.png"
    cars0_gray = cv2.cvtColor(cars0_bgr, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------------------------------
    # REQUIREMENT 1: Stage 1 (Fundamentals & Color Slicing)
    # -------------------------------------------------------------------------
    print("\n[CHECK 1/8] Verifying Stage 1: Fundamentals & Color Slicing...")
    s1 = pipeline._stage1_image_fundamentals(cars0_bgr)
    
    # 1. 256-bin histogram
    rgb_gray_hist = s1["histograms"]["rgb_gray"]
    assert len(rgb_gray_hist["labels"]) == 256, "Histogram labels length != 256"
    assert len(rgb_gray_hist["datasets"]) >= 4, "Expected R, G, B, Gray histogram datasets"
    for ds in rgb_gray_hist["datasets"]:
        assert len(ds["data"]) == 256, f"Histogram dataset {ds['name']} length != 256"

    # 2. HSV/YUV and Distance transforms
    req_images_s1 = [
        "original", "grayscale", "hsv", "yuv",
        "color_slice_mask", "euclidean_distance", "cityblock_distance"
    ]
    for key in req_images_s1:
        assert key in s1["images"], f"Stage 1 missing image key: {key}"
        assert is_valid_base64_image(s1["images"][key]), f"Stage 1 invalid base64 image: {key}"

    # 3. HSV Color Slicing Plate Classification (White, Yellow, Green)
    # Test on full car images
    cars1_bgr = cv2.imread(os.path.join(BASE_DIR, "dataset", "images", "Cars1.png"))
    cars2_bgr = cv2.imread(os.path.join(BASE_DIR, "dataset", "images", "Cars2.png"))

    s1_c0 = pipeline._stage1_image_fundamentals(cars0_bgr)
    s1_c1 = pipeline._stage1_image_fundamentals(cars1_bgr)
    s1_c2 = pipeline._stage1_image_fundamentals(cars2_bgr)

    assert s1_c0["_plate_type"] == "White Private", f"Expected White Private for Cars0, got {s1_c0['_plate_type']}"
    assert s1_c1["_plate_type"] == "Yellow Commercial", f"Expected Yellow Commercial for Cars1, got {s1_c1['_plate_type']}"
    assert s1_c2["_plate_type"] == "Green EV", f"Expected Green EV for Cars2, got {s1_c2['_plate_type']}"

    # Also test isolated patches to guarantee pure HSV slicing accuracy
    white_patch = np.full((32, 140, 3), 255, dtype=np.uint8)
    yellow_patch = np.zeros((32, 140, 3), dtype=np.uint8)
    yellow_patch[:, :] = (0, 200, 255) # BGR Yellow
    green_patch = np.zeros((32, 140, 3), dtype=np.uint8)
    green_patch[:, :] = (0, 180, 0) # BGR Green

    assert pipeline._stage1_image_fundamentals(white_patch)["_plate_type"] == "White Private"
    assert pipeline._stage1_image_fundamentals(yellow_patch)["_plate_type"] == "Yellow Commercial"
    assert pipeline._stage1_image_fundamentals(green_patch)["_plate_type"] == "Green EV"

    results_summary["Stage 1"] = {
        "Status": "PASSED",
        "Histogram Bins": len(rgb_gray_hist["labels"]),
        "Datasets": [ds["name"] for ds in rgb_gray_hist["datasets"]],
        "Cars0 Plate Type": s1_c0["_plate_type"],
        "Cars1 Plate Type": s1_c1["_plate_type"],
        "Cars2 Plate Type": s1_c2["_plate_type"]
    }
    print("  [PASS] Stage 1: 256-bin histograms, color conversions, distance transforms, and HSV color slicing verified.")

    # -------------------------------------------------------------------------
    # REQUIREMENT 2: Stage 2 (Point & Spatial Enhancement)
    # -------------------------------------------------------------------------
    print("\n[CHECK 2/8] Verifying Stage 2: Point & Spatial Enhancement...")
    # Test gamma = 0.6
    s2_g06 = pipeline._stage2_enhancement(cars0_gray, {"gamma": 0.6, "point_processing": "gamma"})
    mse_06 = s2_g06["metrics"]["mse"]
    psnr_06 = s2_g06["metrics"]["psnr"]
    assert isinstance(mse_06, (int, float)) and mse_06 > 0, f"Invalid MSE for gamma=0.6: {mse_06}"
    assert isinstance(psnr_06, (int, float)) and psnr_06 > 0, f"Invalid PSNR for gamma=0.6: {psnr_06}"

    # Test gamma = 1.8
    s2_g18 = pipeline._stage2_enhancement(cars0_gray, {"gamma": 1.8, "point_processing": "gamma"})
    mse_18 = s2_g18["metrics"]["mse"]
    psnr_18 = s2_g18["metrics"]["psnr"]
    assert isinstance(mse_18, (int, float)) and mse_18 > 0, f"Invalid MSE for gamma=1.8: {mse_18}"
    assert isinstance(psnr_18, (int, float)) and psnr_18 > 0, f"Invalid PSNR for gamma=1.8: {psnr_18}"

    # Verify all 9 enhancement images are returned as non-empty valid base64
    s2_full = pipeline._stage2_enhancement(cars0_gray, {
        "point_processing": "gamma",
        "gamma": 1.2,
        "filter_type": "gaussian",
        "enhancement": "clahe"
    })
    req_images_s2 = [
        "digital_negative", "contrast_stretching", "log_transform",
        "power_law_gamma", "gaussian_smoothing", "median_filter",
        "laplacian_sharpening", "histogram_equalization", "clahe",
        "selected_enhanced"
    ]
    for key in req_images_s2:
        assert key in s2_full["images"], f"Stage 2 missing image: {key}"
        assert is_valid_base64_image(s2_full["images"][key]), f"Stage 2 invalid image: {key}"

    results_summary["Stage 2"] = {
        "Status": "PASSED",
        "Gamma 0.6 PSNR": psnr_06,
        "Gamma 0.6 MSE": mse_06,
        "Gamma 1.8 PSNR": psnr_18,
        "Gamma 1.8 MSE": mse_18,
        "Verified Transforms": req_images_s2
    }
    print(f"  [PASS] Stage 2: Transforms verified (Gamma 0.6: PSNR={psnr_06:.2f}dB, Gamma 1.8: PSNR={psnr_18:.2f}dB).")

    # -------------------------------------------------------------------------
    # REQUIREMENT 3: Stage 3 (Edge Detection Benchmark)
    # -------------------------------------------------------------------------
    print("\n[CHECK 3/8] Verifying Stage 3: Edge Detection Benchmark...")
    enhanced_img = s2_full["_enhanced_image"]
    operators = ["sobel", "prewitt", "log", "canny"]
    edge_results = {}

    for op in operators:
        s3 = pipeline._stage3_edge_detection(enhanced_img, {"edge_detector": op})
        assert "images" in s3 and op in s3["images"], f"Stage 3 missing operator image: {op}"
        assert is_valid_base64_image(s3["images"][op]), f"Stage 3 invalid operator base64: {op}"
        assert is_valid_base64_image(s3["images"]["selected_edge"]), f"Stage 3 invalid selected_edge for {op}"
        pixel_count = s3["metrics"].get(f"{op}_edge_pixels", 0)
        assert pixel_count > 0, f"Edge operator {op} returned zero edge pixels"
        edge_results[op] = pixel_count

    results_summary["Stage 3"] = {
        "Status": "PASSED",
        "Operators Verified": operators,
        "Edge Pixel Counts": edge_results
    }
    print(f"  [PASS] Stage 3: All 4 edge operators returned non-empty edge maps ({edge_results}).")

    # -------------------------------------------------------------------------
    # REQUIREMENT 4: Stage 4 (Morphological Processing)
    # -------------------------------------------------------------------------
    print("\n[CHECK 4/8] Verifying Stage 4: Morphological Processing...")
    canny_s3 = pipeline._stage3_edge_detection(enhanced_img, {"edge_detector": "canny"})
    edge_map = canny_s3["_edge_map"]
    s4 = pipeline._stage4_morphology(edge_map)

    req_images_s4 = [
        "morphological_closing", "morphological_opening", "top_hat",
        "black_hat", "boundary_extraction", "skeleton_thinning"
    ]
    for key in req_images_s4:
        assert key in s4["images"], f"Stage 4 missing image key: {key}"
        assert is_valid_base64_image(s4["images"][key]), f"Stage 4 invalid base64: {key}"

    assert s4["metrics"]["closed_pixel_count"] > 0, "Morphological closing returned 0 pixels"
    assert s4["metrics"]["opened_pixel_count"] >= 0, "Morphological opening missing"
    assert s4["metrics"]["boundary_pixel_count"] > 0, "Boundary extraction returned 0 pixels"
    assert s4["metrics"]["skeleton_pixel_count"] > 0, "Skeleton thinning returned 0 pixels"

    results_summary["Stage 4"] = {
        "Status": "PASSED",
        "Closed Pixels": s4["metrics"]["closed_pixel_count"],
        "Opened Pixels": s4["metrics"]["opened_pixel_count"],
        "Boundary Pixels": s4["metrics"]["boundary_pixel_count"],
        "Skeleton Pixels": s4["metrics"]["skeleton_pixel_count"]
    }
    print(f"  [PASS] Stage 4: Closing, Opening, Top-Hat, Black-Hat, Boundary, and Thinning validated.")

    # -------------------------------------------------------------------------
    # REQUIREMENT 5: Stage 5 (Segmentation & Otsu's Method)
    # -------------------------------------------------------------------------
    print("\n[CHECK 5/8] Verifying Stage 5: Segmentation & Otsu's Method...")
    s5 = pipeline._stage5_segmentation(s4["_morph_closed"], cars0_bgr, {})
    
    # Thresholding images
    for t_key in ["global_threshold", "adaptive_threshold", "otsu_threshold", "character_segmentation"]:
        assert t_key in s5["images"], f"Stage 5 missing image: {t_key}"
        assert is_valid_base64_image(s5["images"][t_key]), f"Stage 5 invalid base64 image: {t_key}"

    # Otsu metrics
    tau_star = s5["metrics"].get("otsu_threshold", s5["metrics"].get("tau_star_otsu"))
    sigma_b_sq = s5["metrics"].get("between_class_variance")
    num_chars = s5["metrics"].get("num_characters_segmented", 0)

    assert isinstance(tau_star, (int, float)) and 0 <= tau_star <= 255, f"Invalid Otsu tau*: {tau_star}"
    assert isinstance(sigma_b_sq, (int, float)) and sigma_b_sq > 0, f"Invalid between-class variance: {sigma_b_sq}"
    assert num_chars > 0, f"No characters segmented in Otsu plate binary (got {num_chars})"

    results_summary["Stage 5"] = {
        "Status": "PASSED",
        "Detected BBox": s5["metrics"]["detected_bbox"],
        "Otsu Tau*": tau_star,
        "Between-Class Variance": sigma_b_sq,
        "Characters Segmented": num_chars
    }
    print(f"  [PASS] Stage 5: Otsu tau*={tau_star:.1f}, sigma_B^2={sigma_b_sq:.4f}, Segmented chars={num_chars}.")

    # -------------------------------------------------------------------------
    # REQUIREMENT 6: Stage 6 (DFT & 8x8 DCT Compression)
    # -------------------------------------------------------------------------
    print("\n[CHECK 6/8] Verifying Stage 6: Frequency Domain & 8x8 DCT Compression...")
    plate_roi = s5["_plate_roi"]
    assert plate_roi is not None and plate_roi.size > 0, "Stage 5 failed to extract plate ROI"

    s6 = pipeline._stage6_frequency_domain(plate_roi)
    assert "dft_magnitude_spectrum" in s6["images"], "Stage 6 missing 2D DFT spectrum image"
    assert is_valid_base64_image(s6["images"]["dft_magnitude_spectrum"]), "Stage 6 invalid DFT base64"
    assert is_valid_base64_image(s6["images"]["dct_reconstructed"]), "Stage 6 invalid DCT reconstructed image"

    cr = s6["metrics"].get("compression_ratio")
    rd = s6["metrics"].get("relative_redundancy", s6["metrics"].get("relative_data_redundancy"))
    recon_psnr = s6["metrics"].get("reconstruction_psnr")

    assert isinstance(cr, (int, float)) and cr > 1.0, f"Invalid compression ratio: {cr}"
    assert isinstance(rd, (int, float)) and 0.0 < rd < 1.0, f"Invalid relative redundancy: {rd}"

    results_summary["Stage 6"] = {
        "Status": "PASSED",
        "Compression Ratio": cr,
        "Relative Redundancy": rd,
        "Reconstruction PSNR": recon_psnr
    }
    print(f"  [PASS] Stage 6: 2D DFT image generated; 8x8 DCT CR={cr:.2f}:1, Redundancy={rd * 100:.1f}%, PSNR={recon_psnr}dB.")

    # -------------------------------------------------------------------------
    # REQUIREMENT 7: Stage 7 (Hybrid OCR & Pascal VOC IoU)
    # -------------------------------------------------------------------------
    print("\n[CHECK 7/8] Verifying Stage 7: Hybrid OCR & Pascal VOC IoU...")
    full_pipeline_res = pipeline.process(cars0_path, {}, cars0_xml)

    rec_plate = full_pipeline_res.get("recognized_plate")
    conf = full_pipeline_res.get("confidence", 0.0)
    gt = full_pipeline_res.get("ground_truth", {})
    classical_iou = gt.get("iou_classical")
    hybrid_iou = gt.get("iou_hybrid")

    assert rec_plate is not None and len(rec_plate) >= 6, f"OCR returned invalid plate: {rec_plate}"
    assert conf > 0.0, f"OCR confidence is 0: {conf}"
    assert isinstance(classical_iou, float) and classical_iou > 0.0, f"Classical IoU must be float > 0: {classical_iou}"
    assert isinstance(hybrid_iou, float) and hybrid_iou > 0.0, f"Hybrid IoU must be float > 0: {hybrid_iou}"

    results_summary["Stage 7"] = {
        "Status": "PASSED",
        "Recognized Plate": rec_plate,
        "Confidence": conf,
        "Classical IoU": classical_iou,
        "Hybrid IoU": hybrid_iou,
        "Ground Truth Box": gt.get("bbox")
    }
    print(f"  [PASS] Stage 7: Plate recognized='{rec_plate}' (conf={conf:.2f}), Classical IoU={classical_iou:.4f}, Hybrid IoU={hybrid_iou:.4f}.")

    # -------------------------------------------------------------------------
    # REQUIREMENT 8: FastAPI Endpoints & SQLite Persistence
    # -------------------------------------------------------------------------
    print("\n[CHECK 8/8] Verifying FastAPI Endpoints & SQLite Persistence...")

    # 1. GET /api/dataset/list
    resp_list = client.get("/api/dataset/list")
    assert resp_list.status_code == 200, f"/api/dataset/list failed: {resp_list.status_code}"
    list_json = resp_list.json()
    assert list_json["total"] >= 5, f"Expected at least 5 dataset images, found {list_json['total']}"
    cars0_item = next((item for item in list_json["images"] if item["filename"] == "Cars0.png"), None)
    assert cars0_item is not None, "Cars0.png not found in dataset list"
    assert cars0_item["has_annotation"] is True, "Cars0.png missing annotation"
    assert cars0_item["bbox"] == [140, 364, 280, 396], f"Unexpected Cars0 bbox: {cars0_item['bbox']}"

    # 2. POST /api/process-image
    resp_proc = client.post(
        "/api/process-image",
        data={"dataset_filename": "Cars0.png", "config": json.dumps({"gamma": 1.0})}
    )
    assert resp_proc.status_code == 200, f"/api/process-image failed: {resp_proc.status_code} - {resp_proc.text}"
    proc_json = resp_proc.json()
    assert proc_json["recognized_plate"] == rec_plate, "Endpoint returned different plate than pipeline"
    assert len(proc_json["stages"]) == 7, f"Expected 7 stages from API, got {len(proc_json['stages'])}"

    # 3. POST /api/toll/simulate-entry (Entry Simulation)
    entry_payload = {
        "plate_number": rec_plate,
        "plate_type": "White Private",
        "event_type": "entry",
        "gate_id": "GATE-01"
    }
    resp_entry = client.post("/api/toll/simulate-entry", json=entry_payload)
    assert resp_entry.status_code == 200, f"simulate-entry failed: {resp_entry.status_code} - {resp_entry.text}"
    entry_json = resp_entry.json()
    assigned_bay = entry_json["assigned_bay"]
    assert assigned_bay is not None and len(assigned_bay) >= 3, f"Invalid assigned bay: {assigned_bay}"
    assert entry_json["barrier_status"] == "open", "Barrier should be open on valid entry"

    # 4. POST /api/toll/simulate-entry (Exit Simulation)
    exit_payload = {
        "plate_number": rec_plate,
        "plate_type": "White Private",
        "event_type": "exit",
        "gate_id": "GATE-02"
    }
    resp_exit = client.post("/api/toll/simulate-entry", json=exit_payload)
    assert resp_exit.status_code == 200, f"simulate-exit failed: {resp_exit.status_code} - {resp_exit.text}"
    exit_json = resp_exit.json()
    tariff = exit_json["record"]["tariff"]
    duration = exit_json["record"]["duration_minutes"]
    assert tariff is not None and tariff > 0.0, f"Invalid calculated tariff: {tariff}"
    assert duration is not None and duration > 0.0, f"Invalid duration: {duration}"

    # 5. GET /api/toll/logs
    resp_logs = client.get("/api/toll/logs")
    assert resp_logs.status_code == 200, f"/api/toll/logs failed: {resp_logs.status_code}"
    logs_json = resp_logs.json()
    assert logs_json["total_entries"] >= 1, "Logs should record at least 1 entry"
    assert logs_json["total_exits"] >= 1, "Logs should record at least 1 exit"
    assert logs_json["total_revenue"] > 0.0, "Total revenue should be positive"

    # 6. Verify SQLite Database Persistence
    db_file = os.path.join(BASE_DIR, "toll_parking.db")
    assert os.path.exists(db_file), f"Database file {db_file} does not exist"
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM toll_logs WHERE plate_number = ?", (rec_plate,))
    db_count = c.fetchone()[0]
    conn.close()
    assert db_count >= 2, f"Expected at least 2 persistent logs in DB for {rec_plate}, found {db_count}"

    results_summary["FastAPI & DB"] = {
        "Status": "PASSED",
        "Dataset Count": list_json["total"],
        "Pipeline Stages": len(proc_json["stages"]),
        "Assigned Bay": assigned_bay,
        "Tariff Paid": f"${tariff:.2f}",
        "DB Records for Plate": db_count
    }
    print(f"  [PASS] FastAPI & SQLite: Endpoints passed. Bay={assigned_bay}, Tariff=${tariff:.2f}, DB persisted {db_count} records.")

    # -------------------------------------------------------------------------
    # PRINT FINAL AUDIT REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                    PHASE 1 VERIFICATION AUDIT PASSED (100%)")
    print("=" * 80)
    print(json.dumps(results_summary, indent=2))
    print("=" * 80)
    return True


if __name__ == "__main__":
    try:
        success = run_verification()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n[AUDIT FAILED] Unhandled exception during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
