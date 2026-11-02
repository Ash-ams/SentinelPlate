from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class PipelineConfig(BaseModel):
    gamma: float = Field(default=1.0, ge=0.1, le=5.0)
    filter_type: str = Field(default="gaussian")  # gaussian | median | laplacian_sharpen
    edge_detector: str = Field(default="canny")    # sobel | prewitt | log | canny
    threshold_method: str = Field(default="otsu")  # global | adaptive | otsu
    enhancement: str = Field(default="clahe")      # histogram_eq | clahe | none
    point_processing: str = Field(default="gamma")  # negative | contrast_stretch | log_transform | gamma | none


class HistogramDataset(BaseModel):
    name: str
    data: List[float]
    color: str


class HistogramData(BaseModel):
    labels: List[int]
    datasets: List[HistogramDataset]


class StageResult(BaseModel):
    stage_name: str
    stage_number: int
    description: str
    execution_time_ms: float
    images: Dict[str, str] = Field(default_factory=dict)  # name -> base64
    histograms: Dict[str, Any] = Field(default_factory=dict)  # name -> histogram data
    metrics: Dict[str, Any] = Field(default_factory=dict)  # name -> value
    details: Dict[str, Any] = Field(default_factory=dict)  # extra info


class GroundTruth(BaseModel):
    bbox: Optional[List[int]] = None
    iou_classical: Optional[float] = None
    iou_hybrid: Optional[float] = None


class PipelineResponse(BaseModel):
    filename: str
    original_image: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    stages: List[StageResult]
    plate_bbox: Optional[List[int]] = None
    hybrid_bbox: Optional[List[int]] = None
    detected_bbox: Optional[List[int]] = None
    recognized_plate: Optional[str] = None
    confidence: float = 0.0
    plate_type: str = "Unknown"
    ground_truth: Optional[GroundTruth] = None
    total_time_ms: float = 0.0


class DatasetImage(BaseModel):
    filename: str
    has_annotation: bool
    bbox: Optional[List[int]] = None
    thumbnail: Optional[str] = None
    plate_text: Optional[str] = None
    plate_type: Optional[str] = None
    bndbox: Optional[Dict[str, int]] = None


class DatasetListResponse(BaseModel):
    images: List[DatasetImage]
    total: int


class TollEntry(BaseModel):
    plate_number: str
    plate_type: str = "Unknown"  # White Private, Yellow Commercial, Green EV
    event_type: str = "entry"  # entry | exit
    gate_id: str = "GATE-A1"


class TollRecord(BaseModel):
    id: int
    plate_number: str
    plate_type: str
    event_type: str
    gate_id: str
    timestamp: str
    parking_bay: Optional[str] = None
    duration_minutes: Optional[float] = None
    tariff: Optional[float] = None
    barrier_status: str = "open"


class TollSimulationResponse(BaseModel):
    record: TollRecord
    message: str
    barrier_status: str
    assigned_bay: Optional[str] = None


class TollLogsResponse(BaseModel):
    logs: List[TollRecord]
    total_entries: int
    total_exits: int
    total_revenue: float
    occupancy: int
    capacity: int
    occupancy_percent: float
    revenue_by_type: Dict[str, float]
    occupied_bays: Optional[List[str]] = None


class VideoFrameMeta(BaseModel):
    frame_index: int
    frame_type: str  # I-frame, P-frame, B-frame
    description: str
    mad_vs_prev: float
    mse_vs_prev: float
    pearson_rho: float
    thumbnail: Optional[str] = None
    diff_image: Optional[str] = None


class MotionEstimationMeta(BaseModel):
    algorithm: str
    execution_time_ms: float
    active_blocks: int
    avg_points_per_block: float
    total_eval_points: int
    motion_compensated_psnr: float
    motion_compensated_mse: float
    avg_roi_velocity: float
    overlay_image: Optional[str] = None
    mc_predicted_image: Optional[str] = None
    mc_residual_image: Optional[str] = None
    frame_index: int
    source_frame: int
    target_frame: int


class SmartTollTriggerResult(BaseModel):
    triggered: bool
    trigger_keyframe_index: Optional[int] = None
    trigger_condition: str
    velocity_profile: List[float] = []
    recognized_plate: Optional[str] = None
    confidence: float = 0.0
    plate_type: str = "Unknown"
    alpr_stages_count: int = 0


class VideoMotionResponse(BaseModel):
    num_frames: int
    total_execution_time_ms: float
    frames: List[VideoFrameMeta]
    motion_estimation: List[MotionEstimationMeta]
    benchmark: Dict[str, Any]
    smart_toll_trigger: SmartTollTriggerResult
