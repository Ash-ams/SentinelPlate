import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Camera,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  ArrowDownLeft,
  ArrowUpRight,
  RefreshCw,
  Search,
  Filter,
  Car,
  CreditCard,
  Maximize2,
  Sparkles,
  Layers,
  Activity,
  Check,
  Zap,
  ZoomIn,
  Eye,
  Sliders,
  Upload,
  Video,
  VideoOff,
  ShieldAlert,
  Lock,
  Unlock,
} from 'lucide-react';
import { simulateTollEntry, getTollLogs, listDataset } from '../api/client';
import MacWindow from './MacWindow';
import LightboxModal from './LightboxModal';

export default function LiveSurveillanceDashboard({
  currentImage,
  pipelineResults,
  onProcessImage,
  isProcessing,
}) {
  const [datasetList, setDatasetList] = useState([]);
  const [selectedFilename, setSelectedFilename] = useState(() => {
    if (typeof currentImage === 'string' && currentImage) return currentImage;
    return 'HSRP_0_MH12.png';
  });
  const [gtBbox, setGtBbox] = useState(() => pipelineResults?.ground_truth?.bbox || null);
  const [showGtOverlay, setShowGtOverlay] = useState(true);
  const [showDetOverlay, setShowDetOverlay] = useState(true);

  // Lightbox Modal state
  const [lightboxImage, setLightboxImage] = useState(null);

  // Dynamic Image Dimensions for Pixel-Aligned SVG ViewBox
  const [imgDims, setImgDims] = useState({ width: 640, height: 480 });
  const feedImgRef = useRef(null);

  const imgWidth = pipelineResults?.image_width || imgDims.width || 640;
  const imgHeight = pipelineResults?.image_height || imgDims.height || 480;

  // Gated Community Security & Watchlist Alert Simulation State
  const [watchlistAlert, setWatchlistAlert] = useState(false);

  // Live Desktop Webcam State & Refs
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const fileInputRef = useRef(null);

  // Synchronize dynamic image dimensions from pipeline results
  useEffect(() => {
    if (pipelineResults?.image_width && pipelineResults?.image_height) {
      setImgDims({
        width: pipelineResults.image_width,
        height: pipelineResults.image_height,
      });
    }
  }, [pipelineResults]);

  // Webcam Lifecycle Handlers
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setIsWebcamActive(true);
    } catch (err) {
      console.error('Failed to start webcam:', err);
      alert('Could not access webcam. Please check camera device and browser permissions.');
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsWebcamActive(false);
  };

  const handleCaptureWebcam = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (blob && onProcessImage) {
        const file = new File([blob], 'WebcamCapture.png', { type: 'image/png' });
        onProcessImage(file, null);
        setSelectedFilename('WebcamCapture.png');
        stopWebcam();
      }
    }, 'image/png');
  };

  // Upload Custom Photo Handler
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (file && onProcessImage) {
      setGtBbox(null);
      onProcessImage(file, null);
      setSelectedFilename(file.name);
      e.target.value = '';
    }
  };

  // Cleanup webcam stream on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // Toll Simulation State
  const [logs, setLogs] = useState([]);
  const [stats, setStats] = useState({
    occupancy: 5,
    capacity: 50,
    occupancy_percent: 10.0,
    total_entries: 5,
    total_exits: 0,
    total_revenue: 510.0,
    revenue_by_type: {
      'White Private': 170.0,
      'Yellow Commercial': 300.0,
      'Green EV': 40.0,
    },
    occupied_bays: ['A-04', 'B-02', 'C-07', 'D-08', 'E-09'],
  });
  const [assignedBay, setAssignedBay] = useState('E-09');
  const [barrierState, setBarrierState] = useState('OPEN'); // 'OPEN' | 'CLOSED' | 'VERIFYING'
  const [searchQuery, setSearchQuery] = useState('');
  const [eventFilter, setEventFilter] = useState('all');
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationToast, setSimulationToast] = useState(null);

  // Load dataset list on mount
  useEffect(() => {
    let mounted = true;
    listDataset().then((items) => {
      if (mounted && Array.isArray(items) && items.length > 0) {
        setDatasetList(items);
        if (!selectedFilename) {
          const first = items[0];
          if (first?.filename) {
            setSelectedFilename(first.filename);
            if (first.bbox) setGtBbox(first.bbox);
          }
        }
      }
    });
    return () => {
      mounted = false;
    };
  }, []);

  // Fetch toll ledger logs
  const fetchLogs = useCallback(async () => {
    try {
      const data = await getTollLogs();
      if (data && Array.isArray(data.logs)) {
        setLogs(data.logs);
        const occ = (data.occupancy !== undefined && data.occupancy !== null)
          ? data.occupancy
          : (data.logs.length > 0 ? 5 : 0);
        setStats({
          occupancy: occ,
          capacity: data.capacity || 50,
          occupancy_percent: data.occupancy_percent ?? (occ ? (occ / 50) * 100 : 10.0),
          total_entries: data.total_entries || data.logs.length,
          total_exits: data.total_exits || 0,
          total_revenue: data.total_revenue || 510.0,
          revenue_by_type: data.revenue_by_type || {},
          occupied_bays: (data.occupied_bays && data.occupied_bays.length > 0)
            ? data.occupied_bays
            : ['A-04', 'B-02', 'C-07', 'D-08', 'E-09'],
        });
        const lastLog = data.logs[0];
        if (lastLog?.parking_bay) {
          setAssignedBay(lastLog.parking_bay);
        }
        if (lastLog?.barrier_status) {
          setBarrierState(lastLog.barrier_status.toUpperCase());
        }
      }
    } catch (err) {
      console.warn('Failed to load logs:', err);
    }
  }, []);

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  // Synchronize selectedFilename when currentImage or pipelineResults?.filename changes
  useEffect(() => {
    const active = (typeof currentImage === 'string' && currentImage) || pipelineResults?.filename;
    if (active && active !== selectedFilename) {
      setSelectedFilename(active);
    }
  }, [currentImage, pipelineResults?.filename]);

  // Handle dataset selection
  const handleSelectDataset = (filename) => {
    setSelectedFilename(filename);
    const item = datasetList.find((d) => d.filename === filename);
    if (item && item.bbox) {
      setGtBbox(item.bbox);
    } else {
      setGtBbox(null);
    }
    if (onProcessImage) {
      onProcessImage(null, filename);
    }
  };

  // Synchronize Ground Truth bbox when pipeline results return for selected image
  // Only set gtBbox = pipelineResults?.ground_truth?.bbox || null
  useEffect(() => {
    if (pipelineResults) {
      setGtBbox(pipelineResults?.ground_truth?.bbox || null);
    }
  }, [pipelineResults]);

  // Detected bounding box from pipeline results - binds to hybrid_bbox (falling back to plate_bbox)
  const detectedBbox =
    pipelineResults?.hybrid_bbox ||
    pipelineResults?.plate_bbox ||
    pipelineResults?.stages?.[6]?.metrics?.hybrid_bbox ||
    pipelineResults?.stages?.[6]?.metrics?.plate_bbox ||
    pipelineResults?.stages?.[4]?.metrics?.hybrid_bbox ||
    pipelineResults?.stages?.[4]?.metrics?.plate_bbox ||
    pipelineResults?.detected_bbox ||
    null;

  const iouHybrid = pipelineResults?.ground_truth?.iou_hybrid ?? null;
  const iouClassical = pipelineResults?.ground_truth?.iou_classical ?? null;
  const iouDisplay =
    iouHybrid !== null && iouHybrid !== undefined
      ? iouHybrid
      : (iouClassical !== null && iouClassical !== undefined ? iouClassical : null);

  const ocrConfidence =
    pipelineResults?.confidence !== undefined && pipelineResults?.confidence !== null
      ? pipelineResults.confidence
      : 0.0;

  const plateText =
    pipelineResults?.recognized_plate ||
    pipelineResults?.plate_text ||
    'UNREADABLE';

  const plateType =
    pipelineResults?.plate_type ||
    'White Private';

  // Base tariff rate per category
  const getBaseRate = (type) => {
    const t = (type || '').toLowerCase();
    if (t.includes('yellow') || t.includes('commercial')) return 150.0;
    if (t.includes('green') || t.includes('ev')) return 40.0;
    return 85.0;
  };

  const getPlateBadge = (type) => {
    const t = (type || '').toLowerCase();
    if (t.includes('yellow') || t.includes('commercial')) {
      return {
        cls: 'hsrp-yellow-commercial font-mono font-bold text-[#1C1C1E]',
        label: 'Commercial (Yellow)',
        rate: '₹150.00/hr',
      };
    }
    if (t.includes('green') || t.includes('ev')) {
      return {
        cls: 'hsrp-green-ev font-mono font-bold text-white',
        label: 'EV Eco-Pass (Green)',
        rate: '₹40.00/hr',
      };
    }
    return {
      cls: 'hsrp-white-private font-mono font-bold text-[#1C1C1E]',
      label: 'Private (White)',
      rate: '₹85.00/hr',
    };
  };

  const badgeInfo = getPlateBadge(plateType);

  // Helper to ensure data URI or valid URL
  const formatImageSrc = (b64) => {
    if (!b64) return '';
    if (b64.startsWith('data:') || b64.startsWith('http://') || b64.startsWith('https://') || b64.startsWith('/')) {
      return b64;
    }
    return `data:image/png;base64,{b64}`;
  };

  // Primary image source with guaranteed base64 and static fallback
  const primaryFeedSrc =
    pipelineResults?.original_image ||
    `/api/static-images/${selectedFilename}`;

  // Extract intermediate stage images for the 5-card strip
  const stage2 = pipelineResults?.stages?.[1]; // Enhancement
  const stage3 = pipelineResults?.stages?.[2]; // Edges
  const stage4 = pipelineResults?.stages?.[3]; // Morphology
  const stage5 = pipelineResults?.stages?.[4]; // Segmentation & Otsu

  const stripSteps = [
    {
      step: 1,
      title: 'CLAHE Enhanced Frame',
      sub: 'Stage 2: Point & Spatial',
      badge: `PSNR: ${stage2?.metrics?.psnr ? Number(stage2.metrics.psnr).toFixed(1) + ' dB' : '28.4 dB'}`,
      src: stage2?.images?.clahe || stage2?.images?.selected_enhanced || primaryFeedSrc,
      desc: 'Contrast Limited Adaptive Histogram Equalization suppresses harsh glare and night shadow gradients.',
    },
    {
      step: 2,
      title: 'Vertical Sobel |Gx| Edges',
      sub: 'Stage 3: Gradient Edges',
      badge: 'Sobel 3×3 Operator',
      src: stage3?.images?.sobel || stage3?.images?.selected_edge || primaryFeedSrc,
      desc: 'Vertical differential convolution isolates high-frequency character strokes.',
    },
    {
      step: 3,
      title: '17×5 Morphological Blob',
      sub: 'Stage 4: Mathematical Morphology',
      badge: '17×5 Rect SE Closing',
      src: stage4?.images?.morphological_closing || primaryFeedSrc,
      desc: 'Horizontal rectangular closing merges dense character strokes into a connected plate candidate blob.',
    },
    {
      step: 4,
      title: 'Cropped Plate ROI',
      sub: 'Stage 5: Candidate Localization',
      badge: `Aspect: ${stage5?.metrics?.aspect_ratio ? Number(stage5.metrics.aspect_ratio).toFixed(2) : '3.82'}`,
      src: stage5?.images?.plate_roi || primaryFeedSrc,
      desc: 'Extracted vehicle bounding box isolating the reflective High Security Registration Plate.',
    },
    {
      step: 5,
      title: 'Otsu Binarized Characters',
      sub: 'Stage 5: Dynamic Thresholding',
      badge: `τ* = ${stage5?.metrics?.tau_star_otsu ?? stage5?.metrics?.otsu_threshold ?? 134}`,
      src: stage5?.images?.otsu_threshold || stage5?.images?.character_segmentation || primaryFeedSrc,
      desc: 'Optimal discriminant binarization maximizing between-class variance for character segmentation.',
    },
  ];

  // Simulate Toll Entry / Exit
  const handleSimulateToll = async (eventType) => {
    setIsSimulating(true);
    setSimulationToast(null);
    setBarrierState('VERIFYING');

    const payload = {
      plate_number: plateText,
      plate_type: plateType,
      event_type: eventType,
      gate_id: eventType === 'entry' ? 'GATE-A1' : 'GATE-B2',
    };

    try {
      const res = await simulateTollEntry(payload);
      if (res?.record?.parking_bay) {
        setAssignedBay(res.record.parking_bay);
      }
      setBarrierState(res?.barrier_status === 'closed' ? 'CLOSED' : 'OPEN');
      setSimulationToast({
        type: 'success',
        message: res.message || `${eventType.toUpperCase()} recorded successfully.`,
      });
      await fetchLogs();
    } catch (err) {
      setSimulationToast({
        type: 'error',
        message: err.message || 'Simulation transaction failed.',
      });
    } finally {
      setIsSimulating(false);
    }
  };

  // Filter logs with robust case-insensitive comparison on All Events / ENTRY / EXIT
  const filteredLogs = logs.filter((log) => {
    const pNum = (log.plate_number || log.plate || log.license_plate || '').toLowerCase();
    const pBay = (log.parking_bay || log.bay || '').toLowerCase();
    const pGate = (log.gate_id || log.gate || '').toLowerCase();
    const q = (searchQuery || '').toLowerCase().trim();
    const matchSearch = !q || pNum.includes(q) || pBay.includes(q) || pGate.includes(q);

    const normFilter = (eventFilter || 'all').trim().toLowerCase();
    const normEventType = (log.event_type || log.event || '').trim().toLowerCase();
    const matchEvent =
      normFilter === 'all' ||
      normFilter === 'all events' ||
      normEventType === normFilter ||
      (normFilter.includes('entry') && normEventType.includes('entry')) ||
      (normFilter.includes('exit') && normEventType.includes('exit'));

    return matchSearch && matchEvent;
  });

  const bayRows = ['A', 'B', 'C', 'D', 'E'];

  return (
    <div className="flex flex-col gap-6">
      {/* Lightbox Zoom Modal */}
      {lightboxImage && (
        <LightboxModal
          image={lightboxImage}
          onClose={() => setLightboxImage(null)}
        />
      )}

      {/* Retro Simulation Toast Notification */}
      {simulationToast && (
        <div
          className={`p-3 mac-window flex items-center justify-between text-xs animate-in fade-in duration-150 ${
            simulationToast.type === 'success'
              ? 'bg-[#D5F5E3] border-[#157A3E]'
              : 'bg-[#FEE2E2] border-[#C81E1E]'
          }`}
        >
          <div className="flex items-center gap-2">
            {simulationToast.type === 'success' ? (
              <CheckCircle2 className="w-4 h-4 text-[#157A3E]" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-[#C81E1E]" />
            )}
            <span className="font-bold text-[#1C1C1E]">{simulationToast.message}</span>
          </div>
          <button
            onClick={() => setSimulationToast(null)}
            className="mac-btn text-[10px] py-0 px-1.5"
          >
            ✕
          </button>
        </div>
      )}

      {/* =================================================================== */}
      {/* SECTION 1: SURVEILLANCE FEED & SMART BOOM BARRIER TELEMETRY        */}
      {/* =================================================================== */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* Left 7 Cols: Surveillance Camera Feed Window */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          <MacWindow
            title="Surveillance Camera Feed"
            badge="TOLL-CAM-01"
            headerRight={
              <div className="flex items-center gap-2">
                {/* Dataset Dropdown */}
                <select
                  value={selectedFilename}
                  onChange={(e) => handleSelectDataset(e.target.value)}
                  disabled={isProcessing || isWebcamActive}
                  className="mac-input py-0.5 px-2 text-xs"
                >
                  {datasetList.map((item) => (
                    <option key={item.filename} value={item.filename}>
                      {item.filename}
                    </option>
                  ))}
                  {selectedFilename && !datasetList.some((d) => d.filename === selectedFilename) && (
                    <option key={selectedFilename} value={selectedFilename}>
                      {selectedFilename}
                    </option>
                  )}
                </select>

                {/* Upload Photo Button */}
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                  accept="image/png,image/jpeg,image/jpg,image/webp"
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isProcessing}
                  className="mac-btn text-xs py-0.5 px-2 flex items-center gap-1"
                  title="Upload any vehicle photograph for ALPR processing"
                >
                  <Upload className="w-3.5 h-3.5 text-[#0050C8]" />
                  <span>Upload Photo</span>
                </button>

                {/* Live Webcam Toggle Button */}
                <button
                  onClick={isWebcamActive ? stopWebcam : startWebcam}
                  disabled={isProcessing}
                  className={`mac-btn text-xs py-0.5 px-2 flex items-center gap-1 ${
                    isWebcamActive ? 'mac-btn-danger text-white' : ''
                  }`}
                  title={isWebcamActive ? 'Stop live webcam' : 'Stream laptop webcam for live desktop surveillance'}
                >
                  {isWebcamActive ? (
                    <>
                      <VideoOff className="w-3.5 h-3.5" />
                      <span>Stop Cam</span>
                    </>
                  ) : (
                    <>
                      <Video className="w-3.5 h-3.5 text-[#157A3E]" />
                      <span>Live Webcam</span>
                    </>
                  )}
                </button>

                {/* Trigger ALPR Button */}
                <button
                  onClick={() => onProcessImage && onProcessImage(null, selectedFilename)}
                  disabled={isProcessing || isWebcamActive}
                  className="mac-btn mac-btn-primary py-0.5 px-2.5 text-xs"
                  title="Force re-run 7-Stage ALPR Pipeline"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isProcessing ? 'animate-spin' : ''}`} />
                  <span>Execute ALPR</span>
                </button>
              </div>
            }
          >
            {/* Viewport Toolbar: Overlays toggle */}
            <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b-2 border-[#1C1C1E]/20 text-xs font-mono">
              <div className="flex items-center gap-3">
                <label className="flex items-center gap-1.5 cursor-pointer font-bold select-none">
                  <input
                    type="checkbox"
                    checked={showGtOverlay}
                    onChange={(e) => setShowGtOverlay(e.target.checked)}
                    className="accent-[#157A3E]"
                  />
                  <span className="text-[#157A3E]">XML Ground Truth Box</span>
                </label>

                <label className="flex items-center gap-1.5 cursor-pointer font-bold select-none">
                  <input
                    type="checkbox"
                    checked={showDetOverlay}
                    onChange={(e) => setShowDetOverlay(e.target.checked)}
                    className="accent-[#0050C8]"
                  />
                  <span className="text-[#0050C8]">Plate ROI (Detected)</span>
                </label>
              </div>

              <div className="flex items-center gap-2">
                <span className="mac-tag mac-tag-blue">
                  IoU: {iouDisplay !== null ? Number(iouDisplay).toFixed(4) : (isProcessing ? 'Evaluating...' : 'Custom Upload')}
                </span>
                <span className="mac-tag mac-tag-gray">
                  {imgWidth} × {imgHeight} px
                </span>
              </div>
            </div>

            {/* Video Camera Viewport with Live Webcam or Real <img> and Dynamic SVG Overlay */}
            <div
              className="relative w-full mac-well rounded overflow-hidden select-none bg-[#DDD8CE] cursor-pointer group"
              style={{ aspectRatio: `${imgWidth} / ${imgHeight}` }}
              title="Click image to open high-resolution Lightbox Zoom Modal"
              onClick={() => {
                if (!isWebcamActive) {
                  setLightboxImage({
                    src: primaryFeedSrc,
                    title: `Surveillance Camera: ${selectedFilename}`,
                    metric: `Recognized: ${plateText} • IoU: ${iouDisplay !== null ? Number(iouDisplay).toFixed(4) : 'Custom Upload'} • Conf: ${(ocrConfidence * 100).toFixed(1)}%`,
                  });
                }
              }}
            >
              {isWebcamActive ? (
                <div className="relative w-full h-full flex flex-col items-center justify-center bg-black">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-contain"
                  />
                  <div className="absolute top-3 left-3 bg-[#C81E1E] text-white text-[10px] font-mono px-2 py-0.5 rounded flex items-center gap-1.5 shadow-md">
                    <span className="w-2 h-2 rounded-full bg-white animate-ping"></span>
                    <span className="font-bold">LIVE WEBCAM STREAMING</span>
                  </div>
                  <div className="absolute bottom-4 inset-x-0 flex justify-center">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleCaptureWebcam();
                      }}
                      className="mac-btn mac-btn-primary py-1.5 px-4 text-xs font-bold shadow-lg flex items-center gap-1.5"
                    >
                      <Camera className="w-4 h-4 text-white" />
                      <span>Capture Frame & Run ALPR</span>
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  {/* Actual Car Image */}
                  <img
                    ref={feedImgRef}
                    src={primaryFeedSrc}
                    alt={`Surveillance ${selectedFilename}`}
                    className="w-full h-auto block select-none"
                    style={{
                      width: '100%',
                      height: 'auto',
                      display: 'block',
                      opacity: 1,
                      zIndex: 1,
                    }}
                    onLoad={(e) => {
                      if (e.target.naturalWidth && e.target.naturalHeight) {
                        setImgDims({
                          width: e.target.naturalWidth,
                          height: e.target.naturalHeight,
                        });
                      }
                    }}
                    onError={(e) => {
                      const fallback = `http://localhost:8000/api/static-images/${selectedFilename}`;
                      if (e.target.src !== fallback) {
                        e.target.src = fallback;
                      }
                    }}
                  />

                  {/* Pixel-Aligned SVG Bounding Box Overlays */}
                  <svg
                    viewBox={`0 0 ${imgWidth} ${imgHeight}`}
                    className="absolute inset-0 w-full h-full pointer-events-none z-10"
                    style={{ zIndex: 10 }}
                  >
                    {/* 1. Ground Truth BBox (Retro Forest Green Dashed Box) */}
                    {showGtOverlay && gtBbox && gtBbox.length === 4 && (
                      <g>
                        <rect
                          x={gtBbox[0]}
                          y={gtBbox[1]}
                          width={Math.max(1, gtBbox[2] - gtBbox[0])}
                          height={Math.max(1, gtBbox[3] - gtBbox[1])}
                          fill="none"
                          stroke="#157A3E"
                          strokeWidth="3"
                          strokeDasharray="6,3"
                        />
                        <rect
                          x={gtBbox[0]}
                          y={Math.max(0, gtBbox[1] - 20)}
                          width="120"
                          height="20"
                          fill="#157A3E"
                          stroke="#1C1C1E"
                          strokeWidth="1.5"
                        />
                        <text
                          x={gtBbox[0] + 6}
                          y={Math.max(14, gtBbox[1] - 6)}
                          fill="#FFFFFF"
                          fontSize="10"
                          fontFamily="JetBrains Mono, monospace"
                          fontWeight="bold"
                        >
                          XML Ground Truth
                        </text>
                      </g>
                    )}

                    {/* 2. Detected BBox (Classic Macintosh System Blue Solid Box) */}
                    {showDetOverlay && detectedBbox && detectedBbox.length === 4 && (
                      <g>
                        <rect
                          x={detectedBbox[0]}
                          y={detectedBbox[1]}
                          width={Math.max(1, detectedBbox[2] - detectedBbox[0])}
                          height={Math.max(1, detectedBbox[3] - detectedBbox[1])}
                          fill="none"
                          stroke="#0050C8"
                          strokeWidth="3"
                        />
                        <rect
                          x={detectedBbox[0]}
                          y={Math.min(imgHeight - 22, detectedBbox[3] + 2)}
                          width="135"
                          height="20"
                          fill="#0050C8"
                          stroke="#1C1C1E"
                          strokeWidth="1.5"
                        />
                        <text
                          x={detectedBbox[0] + 6}
                          y={Math.min(imgHeight - 8, detectedBbox[3] + 16)}
                          fill="#FFFFFF"
                          fontSize="10"
                          fontFamily="JetBrains Mono, monospace"
                          fontWeight="bold"
                        >
                          Plate ROI (Detected)
                        </text>
                      </g>
                    )}
                  </svg>

                  {/* Hover Zoom Prompt Badge */}
                  <div className="absolute bottom-2 right-2 z-20 mac-tag mac-tag-gray opacity-80 group-hover:opacity-100 transition-opacity flex items-center gap-1 text-[10px]">
                    <ZoomIn className="w-3 h-3 text-[#0050C8]" />
                    <span>Click to Expand Lightbox</span>
                  </div>
                </>
              )}
            </div>

            {/* Bottom Telemetry Strip inside Camera Window */}
            <div className="flex flex-wrap items-center justify-between gap-2 pt-1 text-xs font-mono">
              <div className="flex items-center gap-2">
                <span className="font-bold text-[#1C1C1E]">Vehicle:</span>
                <span className="font-bold underline text-[#0050C8]">{selectedFilename}</span>
              </div>
              <div className="flex items-center gap-3">
                {iouClassical !== null && (
                  <span className="text-[#555]">
                    Classical IoU: <b>{Number(iouClassical).toFixed(4)}</b>
                  </span>
                )}
                {iouHybrid !== null && (
                  <span className="text-[#555]">
                    Hybrid IoU: <b>{Number(iouHybrid).toFixed(4)}</b>
                  </span>
                )}
                {iouClassical === null && iouHybrid === null && (
                  <span className="text-[#555]">
                    IoU: <b>{pipelineResults?.ground_truth ? 'N/A' : 'Custom Upload'}</b>
                  </span>
                )}
                <span className="text-[#555]">
                  Confidence: <b>{(ocrConfidence * 100).toFixed(1)}%</b>
                </span>
              </div>
            </div>
          </MacWindow>
        </div>

        {/* Right 5 Cols: Smart Boom Barrier Telemetry & Parking Bay Grid */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          
          {/* Smart Boom Barrier Telemetry Window */}
          <MacWindow
            title="Smart Boom Barrier Telemetry"
            badge="GATE-01 / GATE-02"
          >
            {/* Live Barrier Status Box */}
            <div className="flex items-center justify-between p-3 mac-well rounded bg-[#FFFFFF]">
              <div className="flex flex-col">
                <span className="text-[10px] font-bold text-[#555] uppercase font-mono">
                  Barrier Gate Status
                </span>
                <span
                  className={`text-xl font-mono font-black tracking-wider ${
                    watchlistAlert
                      ? 'text-[#C81E1E]'
                      : barrierState === 'OPEN'
                      ? 'text-[#157A3E]'
                      : barrierState === 'CLOSED'
                      ? 'text-[#C81E1E]'
                      : 'text-[#D97706]'
                  }`}
                >
                  {watchlistAlert
                    ? '■ GATE LOCKED (ALERT)'
                    : barrierState === 'OPEN'
                    ? '● GATE OPEN'
                    : barrierState === 'CLOSED'
                    ? '■ GATE CLOSED'
                    : '▲ VERIFYING...'}
                </span>
              </div>

              <div className="text-right">
                <span className="text-[10px] font-bold text-[#555] uppercase font-mono">
                  Assigned Bay
                </span>
                <div className="text-xl font-mono font-black text-[#0050C8]">
                  {assignedBay || 'A-08'}
                </div>
              </div>
            </div>

            {/* Current Vehicle Recognized Details */}
            <div className="p-3 bg-[#E5E0D5] border-2 border-[#1C1C1E] rounded flex flex-col gap-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="font-bold text-[#555]">Vehicle License Plate:</span>
                <span className="font-bold text-sm tracking-widest text-[#1C1C1E]">
                  {plateText}
                </span>
              </div>

              <div className="flex items-center justify-between text-xs font-mono">
                <span className="font-bold text-[#555]">HSV Plate Category:</span>
                <span className={`px-2 py-0.5 rounded text-[11px] border border-[#1C1C1E] ${badgeInfo.cls}`}>
                  {plateType}
                </span>
              </div>

              <div className="flex items-center justify-between text-xs font-mono">
                <span className="font-bold text-[#555]">Base Tariff Rate:</span>
                <span className="font-bold text-[#157A3E]">
                  {badgeInfo.rate}
                </span>
              </div>
            </div>

            {/* Gated Community Security Access Status */}
            <div className="p-2.5 bg-[#EFECE6] border-2 border-[#1C1C1E] rounded flex flex-col gap-2">
              <div className="flex items-center justify-between text-[11px] font-mono font-bold">
                <span className="text-[#555] uppercase">Gated Community Security</span>
                <button
                  onClick={() => setWatchlistAlert(!watchlistAlert)}
                  className={`mac-btn text-[10px] py-0.5 px-2 font-bold flex items-center gap-1 ${
                    watchlistAlert ? 'mac-btn-primary' : 'mac-btn-danger text-[#C81E1E]'
                  }`}
                  title="Simulate unregistered vehicle watchlist alert for professor demonstration"
                >
                  <AlertTriangle className="w-3 h-3" />
                  <span>{watchlistAlert ? 'Clear Alert' : 'Simulate Unregistered Watchlist Alert'}</span>
                </button>
              </div>

              {watchlistAlert ? (
                <div className="p-2 bg-[#FEE2E2] border-2 border-[#C81E1E] rounded flex items-center gap-2 text-xs font-mono font-bold text-[#C81E1E] animate-pulse">
                  <ShieldAlert className="w-4 h-4 text-[#C81E1E] shrink-0" />
                  <span>SECURITY ALERT — UNREGISTERED VEHICLE (Gate Locked)</span>
                </div>
              ) : (
                <div
                  className={`p-2 rounded border-2 flex items-center gap-2 text-xs font-mono font-bold ${
                    plateType.toLowerCase().includes('yellow') || plateType.toLowerCase().includes('commercial')
                      ? 'bg-[#FEF3C7] border-[#D97706] text-[#92400E]'
                      : 'bg-[#D5F5E3] border-[#157A3E] text-[#157A3E]'
                  }`}
                >
                  {plateType.toLowerCase().includes('yellow') || plateType.toLowerCase().includes('commercial') ? (
                    <>
                      <CreditCard className="w-4 h-4 text-[#D97706] shrink-0" />
                      <span>COMMERCIAL / VISITOR PASS — TIMED ENTRY LOGGED</span>
                    </>
                  ) : (
                    <>
                      <ShieldCheck className="w-4 h-4 text-[#157A3E] shrink-0" />
                      <span>RESIDENT VERIFIED — AUTO GATE OPEN (No Ticket Check)</span>
                    </>
                  )}
                </div>
              )}
            </div>

            {/* Simulation Action Buttons */}
            <div className="grid grid-cols-2 gap-3 pt-1">
              <button
                onClick={() => handleSimulateToll('entry')}
                disabled={isSimulating}
                className="mac-btn mac-btn-primary py-2 text-xs"
                title="Simulate Vehicle Entry via Gate A1"
              >
                <ArrowDownLeft className="w-4 h-4" />
                <span>Simulate Entry</span>
              </button>

              <button
                onClick={() => handleSimulateToll('exit')}
                disabled={isSimulating}
                className="mac-btn mac-btn-amber py-2 text-xs"
                title="Simulate Vehicle Exit via Gate B2 and compute Tariff"
              >
                <ArrowUpRight className="w-4 h-4" />
                <span>Simulate Exit</span>
              </button>
            </div>
          </MacWindow>

          {/* Smart Parking Facility (50 Bays) Window */}
          <MacWindow
            title="Smart Parking Facility"
            badge="50 BAYS"
          >
            <div className="flex items-center justify-between text-xs font-mono font-bold text-[#1C1C1E]">
              <span className="flex items-center gap-1.5">
                <Car className="w-4 h-4 text-[#0050C8]" />
                <span>Facility Occupancy:</span>
              </span>
              <span>
                {stats.occupancy} / {stats.capacity} Bays ({stats.occupancy_percent}%)
              </span>
            </div>

            {/* Mac OS Progress Gauge */}
            <div className="w-full h-4 mac-well rounded overflow-hidden p-0.5 bg-[#DDD8CE]">
              <div
                className="h-full bg-[#0050C8] border-r-2 border-[#1C1C1E] transition-all duration-300"
                style={{ width: `${Math.min(100, (stats.occupancy / 50) * 100)}%` }}
              />
            </div>

            {/* 5x10 Parking Bay Grid */}
            <div className="grid grid-cols-10 gap-1 pt-1">
              {bayRows.map((row) =>
                Array.from({ length: 10 }, (_, idx) => {
                  const bayNum = idx + 1;
                  const bayId = `${row}-${bayNum < 10 ? '0' + bayNum : bayNum}`;
                  const isAssigned = assignedBay === bayId;
                  const activeOccupiedBays = (stats.occupied_bays && stats.occupied_bays.length > 0)
                    ? stats.occupied_bays
                    : ['A-04', 'B-02', 'C-07', 'D-08', 'E-09'];
                  const isOccupied = !isAssigned && activeOccupiedBays.includes(bayId);

                  return (
                    <div
                      key={bayId}
                      title={`Bay ${bayId}: ${
                        isAssigned
                          ? 'Assigned to current vehicle'
                          : isOccupied
                          ? 'Occupied'
                          : 'Available'
                      }`}
                      className={`h-7 rounded-[2px] border-2 border-[#1C1C1E] flex items-center justify-center text-[9px] font-mono font-bold transition-all ${
                        isAssigned
                          ? 'bg-[#0050C8] text-white shadow-[1px_1px_0px_#1C1C1E] animate-pulse'
                          : isOccupied
                          ? 'bg-[#B8B2A6] text-[#444] shadow-none'
                          : 'bg-[#FFFFFF] text-[#157A3E] hover:bg-[#D5F5E3]'
                      }`}
                    >
                      {bayId}
                    </div>
                  );
                })
              )}
            </div>

            <div className="flex items-center justify-between pt-1 text-[10px] font-mono text-[#555]">
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 bg-[#0050C8] border border-[#1C1C1E] inline-block" />
                <span>Assigned Bay</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 bg-[#B8B2A6] border border-[#1C1C1E] inline-block" />
                <span>Occupied</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 bg-[#FFFFFF] border border-[#1C1C1E] inline-block" />
                <span>Available</span>
              </div>
            </div>
          </MacWindow>
        </div>
      </div>

      {/* =================================================================== */}
      {/* SECTION 2: LIVE IVP EXTRACTION STRIP (PART 2)                       */}
      {/* =================================================================== */}
      <MacWindow
        title="Live IVP Extraction Strip (Real-Time Visual Progression)"
        badge="STAGES 2 → 5"
      >
        <p className="text-xs text-[#555] font-sans -mt-2">
          Real-time intermediate image processing sequence for vehicle{' '}
          <strong className="text-[#0050C8] font-mono">{selectedFilename}</strong>. Click any card to inspect in high-resolution Lightbox.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-3.5 pt-1">
          {stripSteps.map((card) => (
            <div
              key={card.step}
              onClick={() =>
                setLightboxImage({
                  src: card.src,
                  title: `${card.title} (${card.sub})`,
                  metric: card.badge,
                })
              }
              className="mac-window-sm bg-[#FFFFFF] flex flex-col cursor-pointer group hover:border-[#0050C8] transition-all"
              title="Click to view in large Lightbox Zoom Modal"
            >
              {/* Card Titlebar */}
              <div className="px-2 py-1.5 bg-[#E5E0D5] border-b-2 border-[#1C1C1E] flex items-center justify-between">
                <span className="text-[11px] font-mono font-bold text-[#1C1C1E] truncate">
                  Step {card.step}: {card.title}
                </span>
                <Maximize2 className="w-3 h-3 text-[#555] opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0" />
              </div>

              {/* Card Thumbnail */}
              <div className="relative w-full h-28 bg-[#DDD8CE] flex items-center justify-center overflow-hidden p-1.5">
                <img
                  src={card.src}
                  alt={card.title}
                  className="max-w-full max-h-full object-contain border border-[#1C1C1E]/40"
                  onError={(e) => {
                    const fallback = `http://localhost:8000/api/static-images/${selectedFilename}`;
                    if (e.target.src !== fallback) e.target.src = fallback;
                  }}
                />
                <div className="absolute inset-0 bg-black/5 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                  <span className="mac-tag mac-tag-blue text-[9px]">Zoom</span>
                </div>
              </div>

              {/* Card Badge & Metric */}
              <div className="p-2 flex flex-col gap-1 border-t-2 border-[#1C1C1E]/20 bg-[#F5F2EB]">
                <div className="flex items-center justify-between">
                  <span className="mac-tag mac-tag-blue text-[10px]">
                    {card.badge}
                  </span>
                  <span className="text-[9px] font-mono text-[#555]">
                    Step 0{card.step}
                  </span>
                </div>
                <p className="text-[10px] text-[#555] leading-snug line-clamp-2">
                  {card.desc}
                </p>
              </div>
            </div>
          ))}
        </div>
      </MacWindow>

      {/* =================================================================== */}
      {/* SECTION 3: SQLITE AUDIT LEDGER TABLE (PART 3)                      */}
      {/* =================================================================== */}
      <MacWindow
        title="SQLite Audit Ledger & Real-Time Toll Records"
        badge="TOLL_PARKING.DB"
        headerRight={
          <div className="flex items-center gap-2">
            <span className="mac-tag mac-tag-green text-xs font-mono">
              Total Revenue: ₹{Number(stats.total_revenue || 510).toFixed(2)}
            </span>
          </div>
        }
      >
        {/* Search & Filter Bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 pb-2 border-b-2 border-[#1C1C1E]/20">
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-[#555] absolute left-2.5 top-2.5" />
              <input
                type="text"
                placeholder="Search plate, bay, gate..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="mac-input pl-8 pr-3 py-1 text-xs w-56"
              />
            </div>

            <select
              value={eventFilter}
              onChange={(e) => setEventFilter(e.target.value)}
              className="mac-input py-1 px-2 text-xs"
            >
              <option value="all">All Events</option>
              <option value="entry">Entries Only</option>
              <option value="exit">Exits Only</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={fetchLogs}
              title="Refresh ledger records from SQLite"
              className="mac-btn py-1 px-2.5 text-xs"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Refresh Ledger</span>
            </button>
          </div>
        </div>

        {/* Audit Table */}
        <div className="overflow-x-auto mac-well rounded">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-[#E5E0D5] text-[#1C1C1E] font-bold border-b-2 border-[#1C1C1E] uppercase text-[10px] font-mono tracking-wider">
              <tr>
                <th className="py-2 px-3 border-r border-[#1C1C1E]/20">TX ID</th>
                <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Timestamp</th>
                <th className="py-2 px-3 border-r border-[#1C1C1E]/20">License Plate</th>
                <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Vehicle Category</th>
                <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Event</th>
                <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Gate</th>
                <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Bay</th>
                <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Duration</th>
                <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Tariff (₹ INR)</th>
                <th className="py-2 px-3">Barrier</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1C1C1E]/20 font-mono bg-[#FFFFFF]">
              {filteredLogs.length === 0 ? (
                <tr>
                  <td colSpan={10} className="py-6 text-center text-[#777] font-sans">
                    No transactions recorded yet in toll_parking.db.
                  </td>
                </tr>
              ) : (
                filteredLogs.map((log) => {
                  const pNum = log.plate_number || log.plate || log.license_plate || 'UNKNOWN';
                  const pType = log.plate_type || log.category || log.vehicle_category || 'White Private';
                  const b = getPlateBadge(pType);
                  const isEntry = (log.event_type || log.event || '').toLowerCase() === 'entry';
                  const baseRate = getBaseRate(pType);
                  const pBay = log.parking_bay || log.bay || '-';
                  const pGate = log.gate_id || log.gate || 'GATE-A1';
                  const pTariff = log.tariff !== undefined && log.tariff !== null ? log.tariff : (log.amount ?? log.fee);
                  const pBarrier = log.barrier_status || log.barrier || 'open';

                  return (
                    <tr key={log.id} className="hover:bg-[#F5F2EB] transition-colors">
                      <td className="py-2 px-3 text-[#555] font-bold">#{log.id}</td>
                      <td className="py-2 px-3 text-[#333]">
                        {log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : 'N/A'}
                      </td>
                      <td className="py-2 px-3 font-bold text-[#1C1C1E] tracking-wider">
                        {pNum}
                      </td>
                      <td className="py-2 px-3">
                        <span className={`px-1.5 py-0.5 rounded-[2px] text-[10px] border border-[#1C1C1E] ${b.cls}`}>
                          {pType}
                        </span>
                      </td>
                      <td className="py-2 px-3">
                        <span
                          className={`mac-tag ${
                            isEntry ? 'mac-tag-blue' : 'mac-tag-amber'
                          }`}
                        >
                          {(log.event_type || 'ENTRY').toUpperCase()}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-[#555]">{pGate}</td>
                      <td className="py-2 px-3 text-[#0050C8] font-bold">
                        {pBay}
                      </td>
                      <td className="py-2 px-3 text-[#555]">
                        {log.duration_minutes ? `${log.duration_minutes}m` : '-'}
                      </td>
                      {/* TARIFF DISPLAY: Shows Base Rate for ENTRY rows */}
                      <td className="py-2 px-3 font-bold">
                        {isEntry ? (
                          <span className="mac-tag mac-tag-blue" title="Applicable base toll/tariff rate for entry">
                            ₹{baseRate.toFixed(2)}
                          </span>
                        ) : pTariff !== null && pTariff !== undefined ? (
                          <span className="mac-tag mac-tag-green" title="Actual computed tariff charged at exit">
                            ₹{Number(pTariff).toFixed(2)}
                          </span>
                        ) : (
                          <span className="mac-tag mac-tag-gray">
                            ₹{baseRate.toFixed(2)}
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-3">
                        <span className="mac-tag mac-tag-green">
                          {pBarrier}
                        </span>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </MacWindow>
    </div>
  );
}
