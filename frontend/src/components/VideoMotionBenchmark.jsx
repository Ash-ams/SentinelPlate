import React, { useState, useEffect } from 'react';
import {
  Film,
  Zap,
  Play,
  RotateCcw,
  CheckCircle2,
  Download,
  Copy,
  Table as TableIcon,
  BarChart3,
  Sparkles,
  Layers,
  ArrowRight,
  TrendingDown,
  Activity,
  Check,
  ZoomIn,
} from 'lucide-react';
import { processVideoFrames, getBatchEvaluation } from '../api/client';
import MacWindow from './MacWindow';
import LightboxModal from './LightboxModal';

export default function VideoMotionBenchmark({ datasetImages = [] }) {
  // Video Motion Engine State
  const [selectedDatasetImg, setSelectedDatasetImg] = useState('HSRP_0_MH12.png');
  const [selectedAlgo, setSelectedAlgo] = useState('tss');
  const [videoResult, setVideoResult] = useState(null);
  const [isVideoLoading, setIsVideoLoading] = useState(false);
  const [videoError, setVideoError] = useState(null);
  const [activeFrameIdx, setActiveFrameIdx] = useState(1);

  // Lightbox modal state
  const [lightboxImage, setLightboxImage] = useState(null);

  // Batch Evaluation State
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [isBenchLoading, setIsBenchLoading] = useState(false);
  const [benchError, setBenchError] = useState(null);
  const [copiedMarkdown, setCopiedMarkdown] = useState(false);

  // Auto-run video motion for HSRP_0_MH12.png on mount
  useEffect(() => {
    runVideoMotion('HSRP_0_MH12.png', 'tss');
    runBatchBenchmark();
  }, []);

  const runVideoMotion = async (filename, algo) => {
    setIsVideoLoading(true);
    setVideoError(null);
    try {
      const data = await processVideoFrames({
        datasetFilename: filename,
        algorithm: algo,
        numFrames: 6,
      });
      setVideoResult(data);
      if (data?.frames?.length > 1) {
        const triggerIdx = data.smart_toll_trigger?.trigger_keyframe_index || 1;
        setActiveFrameIdx(triggerIdx);
      }
    } catch (err) {
      setVideoError(err.message || 'Failed to process video motion frames');
    } finally {
      setIsVideoLoading(false);
    }
  };

  const runBatchBenchmark = async () => {
    setIsBenchLoading(true);
    setBenchError(null);
    try {
      const data = await getBatchEvaluation();
      setBenchmarkData(data);
    } catch (err) {
      setBenchError(err.message || 'Failed to compute batch dataset evaluation');
    } finally {
      setIsBenchLoading(false);
    }
  };

  // Current active frame data
  const currentFrame = videoResult?.frames?.[activeFrameIdx] || videoResult?.frames?.[1];
  const currentMotion =
    videoResult?.motion_estimation?.find((m) => m.frame_index === activeFrameIdx) ||
    videoResult?.motion_estimation?.[0];

  const formatBase64 = (b64) => {
    if (!b64) return '';
    if (b64.startsWith('data:') || b64.startsWith('http://') || b64.startsWith('https://') || b64.startsWith('/')) {
      return b64;
    }
    return `data:image/png;base64,${b64}`;
  };

  // Generate Markdown string for IEEE Report export
  const generateIeeeMarkdown = () => {
    if (!benchmarkData) return '';
    const t1 = benchmarkData.enhancement_benchmark;
    const t2 = benchmarkData.edge_localization_benchmark;
    const t3 = benchmarkData.segmentation_ocr_benchmark;
    const t4 = benchmarkData.motion_estimation_benchmark;

    let md = `# IEEE Research Benchmark: ALPR Digital Image & Video Processing Evaluation\n\n`;
    md += `*Evaluated on ${benchmarkData.dataset_total_images || 11} Vehicles across Synthetic HSRP, Challenging Night/Glare, and Real Photo Datasets*\n\n`;

    // Table I
    md += `### ${t1.table_title}\n\n`;
    md += `| Enhancement Technique | Mean MSE | Mean PSNR (dB) | Shannon Entropy (bits/px) |\n`;
    md += `| :--- | :---: | :---: | :---: |\n`;
    t1.rows.forEach((r) => {
      md += `| ${r.method_name} | ${r.mean_mse} | ${r.mean_psnr_db} | ${r.mean_entropy_bits} |\n`;
    });
    md += `\n`;

    // Table II
    md += `### ${t2.table_title}\n\n`;
    md += `| Operator + 17x5 Closing | Mean IoU | Precision | Recall | Latency (ms) |\n`;
    md += `| :--- | :---: | :---: | :---: | :---: |\n`;
    t2.rows.forEach((r) => {
      md += `| ${r.operator_name} | ${r.mean_iou} | ${r.precision} | ${r.recall} | ${r.mean_latency_ms} |\n`;
    });
    md += `\n`;

    // Table III
    md += `### ${t3.table_title}\n\n`;
    md += `| Binarization Method | Mean Otsu $\\tau^*$ | Character Seg Rate (%) | Full Plate OCR Accuracy (%) |\n`;
    md += `| :--- | :---: | :---: | :---: |\n`;
    t3.rows.forEach((r) => {
      md += `| ${r.method_name} | ${r.mean_tau_star} | ${r.char_segmentation_rate_pct}% | ${r.plate_ocr_accuracy_pct}% |\n`;
    });
    md += `\n`;

    // Table IV
    md += `### ${t4.table_title}\n\n`;
    md += `| Motion Estimation Algorithm | Search Pts / Block | Speedup Factor | Prediction PSNR (dB) | Latency (ms) |\n`;
    md += `| :--- | :---: | :---: | :---: | :---: |\n`;
    t4.rows.forEach((r) => {
      md += `| ${r.algorithm_name} | ${r.search_points_per_block} | ${r.speedup_factor}x | ${r.prediction_psnr_db} dB | ${r.latency_ms} ms |\n`;
    });

    return md;
  };

  const handleCopyMarkdown = () => {
    const md = generateIeeeMarkdown();
    navigator.clipboard.writeText(md).then(() => {
      setCopiedMarkdown(true);
      setTimeout(() => setCopiedMarkdown(false), 2500);
    });
  };

  const handleDownloadMarkdown = () => {
    const md = generateIeeeMarkdown();
    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'ALPR_IEEE_Benchmark_Report.md';
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col gap-8 pb-12">
      {/* Lightbox Modal */}
      {lightboxImage && (
        <LightboxModal
          image={lightboxImage}
          onClose={() => setLightboxImage(null)}
        />
      )}

      {/* ================================================================= */}
      {/* SECTION 1: Unit 8 Digital Video & Motion Estimation Lab           */}
      {/* ================================================================= */}
      <MacWindow
        title="Unit 8: Digital Video & Motion Estimation Lab"
        badge="N=16, p=7 BLOCK MATCHING"
        headerRight={
          <div className="flex items-center gap-2">
            <select
              value={selectedDatasetImg}
              onChange={(e) => {
                setSelectedDatasetImg(e.target.value);
                runVideoMotion(e.target.value, selectedAlgo);
              }}
              disabled={isVideoLoading}
              className="mac-input text-xs py-0.5 px-2"
            >
              <option value="HSRP_0_MH12.png">HSRP_0_MH12.png (White Private)</option>
              <option value="HSRP_1_KA01.png">HSRP_1_KA01.png (Yellow Commercial)</option>
              <option value="HSRP_2_DL3C.png">HSRP_2_DL3C.png (Green EV)</option>
              <option value="HSRP_3_TN09.png">HSRP_3_TN09.png (White Private)</option>
              <option value="HSRP_4_GJ05.png">HSRP_4_GJ05.png (Yellow Commercial)</option>
              <option value="Cars0.png">Cars0.png (Kaggle Real Photo)</option>
              <option value="Cars1.png">Cars1.png (Kaggle Real Photo)</option>
            </select>

            <select
              value={selectedAlgo}
              onChange={(e) => {
                setSelectedAlgo(e.target.value);
                runVideoMotion(selectedDatasetImg, e.target.value);
              }}
              disabled={isVideoLoading}
              className="mac-input text-xs py-0.5 px-2"
            >
              <option value="tss">Three-Step Search (TSS - 25 pts)</option>
              <option value="esbm">Exhaustive Search (ESBM - 225 pts)</option>
              <option value="log2d">2D-Log Search (17 pts)</option>
            </select>

            <button
              onClick={() => runVideoMotion(selectedDatasetImg, selectedAlgo)}
              disabled={isVideoLoading}
              className="mac-btn mac-btn-primary text-xs py-0.5 px-2.5"
            >
              <Play className={`w-3.5 h-3.5 ${isVideoLoading ? 'animate-spin' : ''}`} />
              <span>Simulate Motion</span>
            </button>
          </div>
        }
      >
        <p className="text-xs text-[#555] font-sans -mt-2">
          Temporal correlation analysis (MAD, Pearson $\rho$), macroblock block-matching benchmark (ESBM vs TSS vs 2D-Log), and automatic smart toll barrier deceleration trigger.
        </p>

        {/* Smart Deceleration Trigger Banner */}
        {videoResult?.smart_toll_trigger && (
          <div className="bg-[#D5F5E3] border-2 border-[#157A3E] rounded p-3 flex flex-wrap items-center justify-between gap-3 shadow-[2px_2px_0px_#1C1C1E]">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded bg-[#157A3E] border-2 border-[#1C1C1E] flex items-center justify-center text-white flex-shrink-0">
                <Zap className="w-5 h-5 animate-pulse" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-[#157A3E] uppercase font-mono">
                    Smart Toll Deceleration Trigger Activated
                  </span>
                  <span className="mac-tag mac-tag-green text-[10px]">
                    Keyframe f_{videoResult.smart_toll_trigger.trigger_keyframe_index}
                  </span>
                </div>
                <p className="text-xs text-[#333] mt-0.5 font-sans">
                  Vehicle deceleration detected in Toll Sensor Zone ($v &lt; 1.0\text{ px/frame}$). Automatically triggered Stage 1–7 ALPR pipeline.
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 bg-[#FFFFFF] px-3 py-1.5 rounded border-2 border-[#1C1C1E] shadow-[1px_1px_0px_#1C1C1E]">
              <div className="flex flex-col text-right font-mono">
                <span className="text-[9px] uppercase font-bold text-[#555]">
                  Triggered Plate
                </span>
                <span className="text-[10px] text-[#157A3E] font-bold">
                  {(videoResult.smart_toll_trigger.confidence * 100).toFixed(1)}% Match
                </span>
              </div>
              <span className="hsrp-plate hsrp-white-private px-2.5 py-0.5 text-xs font-mono font-black">
                {videoResult.smart_toll_trigger.recognized_plate || 'UNREADABLE'}
              </span>
            </div>
          </div>
        )}

        {/* Video Scrubber & GOP Classification */}
        <div className="flex flex-col gap-2 pt-1">
          <label className="text-xs font-bold font-mono text-[#1C1C1E] flex items-center justify-between">
            <span>Video Sequence Frames ($f_0 \dots f_5$):</span>
            <span className="text-[#0050C8]">
              Active Frame: f_{activeFrameIdx} ({currentFrame?.frame_type || 'P-frame'})
            </span>
          </label>

          <div className="grid grid-cols-6 gap-2">
            {videoResult?.frames?.map((f, idx) => {
              const isSelected = activeFrameIdx === f.frame_index;
              const isTrigger =
                videoResult?.smart_toll_trigger?.trigger_keyframe_index === f.frame_index;

              return (
                <button
                  key={idx}
                  onClick={() => setActiveFrameIdx(f.frame_index)}
                  className={`flex flex-col items-center gap-1.5 p-1.5 rounded border-2 transition-all cursor-pointer ${
                    isSelected
                      ? 'bg-[#1C1C1E] text-white border-[#1C1C1E] shadow-[2px_2px_0px_#0050C8]'
                      : 'bg-[#FFFFFF] text-[#1C1C1E] border-[#1C1C1E] shadow-[2px_2px_0px_#1C1C1E] hover:bg-[#ECE7DD]'
                  }`}
                >
                  <div className="relative w-full aspect-[4/3] rounded-[2px] overflow-hidden bg-[#DDD8CE] border border-[#1C1C1E]">
                    <img
                      src={formatBase64(f.thumbnail)}
                      alt={`Frame ${f.frame_index}`}
                      className="w-full h-full object-cover"
                    />
                    {isTrigger && (
                      <span className="absolute top-1 right-1 w-2.5 h-2.5 rounded-full bg-[#157A3E] border border-white animate-ping" />
                    )}
                  </div>

                  <div className="flex items-center justify-between w-full px-0.5">
                    <span className="font-mono font-bold text-xs">f_{f.frame_index}</span>
                    <span
                      className={`text-[9px] font-bold px-1 py-0.2 rounded ${
                        f.frame_type === 'I-frame'
                          ? 'bg-[#FEE2E2] text-[#991B1B]'
                          : f.frame_type === 'P-frame'
                          ? 'bg-[#D9E8FF] text-[#003E9E]'
                          : 'bg-[#EDE9FE] text-[#6B21A8]'
                      }`}
                    >
                      {f.frame_type}
                    </span>
                  </div>

                  <span className="text-[10px] font-mono opacity-80">
                    MAD: {f.mad_vs_prev ? f.mad_vs_prev.toFixed(1) : '0.0'}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Side-by-Side Visualizers: Motion Vectors vs MC-Predicted vs Residual */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          {/* 1. Motion Vector Field Overlay */}
          <div
            onClick={() =>
              setLightboxImage({
                src: formatBase64(currentMotion?.overlay_image || currentFrame?.thumbnail),
                title: `Motion Vector Field Overlay (Frame f_${activeFrameIdx})`,
                metric: `Avg Velocity: ${currentMotion?.avg_roi_velocity || '0.0'} px/frame`,
              })
            }
            className="mac-window-sm bg-[#FFFFFF] p-2 flex flex-col gap-2 cursor-pointer group hover:border-[#0050C8] transition-all"
            title="Click to expand Lightbox"
          >
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="font-bold text-[#1C1C1E]">Motion Vector Field Overlay</span>
              <span className="mac-tag mac-tag-green text-[10px]">
                v: {currentMotion?.avg_roi_velocity || '0.0'} px
              </span>
            </div>
            <div className="w-full aspect-[4/3] rounded mac-well overflow-hidden bg-[#DDD8CE] flex items-center justify-center relative">
              <img
                src={formatBase64(currentMotion?.overlay_image || currentFrame?.thumbnail)}
                alt="Motion Vectors"
                className="w-full h-full object-contain"
              />
              <div className="absolute inset-0 bg-black/5 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <span className="mac-tag mac-tag-blue text-[10px]">Click to Zoom</span>
              </div>
            </div>
            <span className="text-[10px] text-[#555] leading-tight font-sans">
              Directional displacement vectors $(dx, dy)$ on active vehicle macroblocks inside Toll Sensor Zone.
            </span>
          </div>

          {/* 2. Motion-Compensated Predicted Frame */}
          <div
            onClick={() =>
              setLightboxImage({
                src: formatBase64(currentMotion?.mc_predicted_image || currentFrame?.thumbnail),
                title: `MC-Predicted Frame (Frame f_${activeFrameIdx})`,
                metric: `PSNR: ${currentMotion?.motion_compensated_psnr || '32.5'} dB`,
              })
            }
            className="mac-window-sm bg-[#FFFFFF] p-2 flex flex-col gap-2 cursor-pointer group hover:border-[#0050C8] transition-all"
            title="Click to expand Lightbox"
          >
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="font-bold text-[#1C1C1E]">MC-Predicted Frame f̂_t</span>
              <span className="mac-tag mac-tag-blue text-[10px]">
                PSNR: {currentMotion?.motion_compensated_psnr || '32.5'} dB
              </span>
            </div>
            <div className="w-full aspect-[4/3] rounded mac-well overflow-hidden bg-[#DDD8CE] flex items-center justify-center relative">
              <img
                src={formatBase64(currentMotion?.mc_predicted_image || currentFrame?.thumbnail)}
                alt="Motion Compensated Frame"
                className="w-full h-full object-contain"
              />
              <div className="absolute inset-0 bg-black/5 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <span className="mac-tag mac-tag-blue text-[10px]">Click to Zoom</span>
              </div>
            </div>
            <span className="text-[10px] text-[#555] leading-tight font-sans">
              Motion-compensated prediction reconstructed by displacing reference blocks from $f_{t-1}$.
            </span>
          </div>

          {/* 3. Temporal Residual Error Frame */}
          <div
            onClick={() =>
              setLightboxImage({
                src: formatBase64(currentMotion?.mc_residual_image || currentFrame?.diff_image),
                title: `Temporal Residual Error Frame (Frame f_${activeFrameIdx})`,
                metric: `MSE: ${currentMotion?.motion_compensated_mse || '14.2'}`,
              })
            }
            className="mac-window-sm bg-[#FFFFFF] p-2 flex flex-col gap-2 cursor-pointer group hover:border-[#0050C8] transition-all"
            title="Click to expand Lightbox"
          >
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="font-bold text-[#1C1C1E]">Residual Error |f_t - f̂_t|</span>
              <span className="mac-tag mac-tag-amber text-[10px]">
                MSE: {currentMotion?.motion_compensated_mse || '14.2'}
              </span>
            </div>
            <div className="w-full aspect-[4/3] rounded mac-well overflow-hidden bg-[#DDD8CE] flex items-center justify-center relative">
              <img
                src={formatBase64(currentMotion?.mc_residual_image || currentFrame?.diff_image)}
                alt="Residual Error"
                className="w-full h-full object-contain"
              />
              <div className="absolute inset-0 bg-black/5 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <span className="mac-tag mac-tag-blue text-[10px]">Click to Zoom</span>
              </div>
            </div>
            <span className="text-[10px] text-[#555] leading-tight font-sans">
              Displaced frame difference highlighting prediction error and edge boundaries.
            </span>
          </div>
        </div>

        {/* Algorithm Speedup Cards */}
        {videoResult?.benchmark && (
          <div className="flex flex-col gap-2 pt-2 border-t-2 border-[#1C1C1E]/20">
            <h4 className="text-xs font-mono font-bold text-[#1C1C1E] uppercase tracking-wider">
              Search Speedup Benchmark ({videoResult.benchmark.blocks_benchmarked} Active Blocks Evaluated)
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="bg-[#FFFFFF] p-3 mac-window-sm flex flex-col gap-1">
                <span className="text-xs font-bold text-[#1C1C1E]">Exhaustive Search (ESBM)</span>
                <span className="text-lg font-mono font-black text-[#C81E1E]">
                  {videoResult.benchmark.esbm.avg_points_per_block} pts/block
                </span>
                <span className="text-[11px] font-mono text-[#555]">
                  Latency: {videoResult.benchmark.esbm.latency_ms} ms (Baseline 1.0x)
                </span>
              </div>

              <div className="bg-[#FFFFFF] p-3 mac-window-sm flex flex-col gap-1">
                <span className="text-xs font-bold text-[#157A3E]">Three-Step Search (TSS)</span>
                <span className="text-lg font-mono font-black text-[#157A3E]">
                  {videoResult.benchmark.tss.avg_points_per_block} pts/block
                </span>
                <span className="text-[11px] font-mono text-[#157A3E] font-bold">
                  {videoResult.benchmark.tss.speedup_vs_esbm}x Speedup (Syllabus Target: 9.0x)
                </span>
              </div>

              <div className="bg-[#FFFFFF] p-3 mac-window-sm flex flex-col gap-1">
                <span className="text-xs font-bold text-[#0050C8]">2D-Logarithmic Search</span>
                <span className="text-lg font-mono font-black text-[#0050C8]">
                  {videoResult.benchmark.log2d.avg_points_per_block} pts/block
                </span>
                <span className="text-[11px] font-mono text-[#0050C8] font-bold">
                  {videoResult.benchmark.log2d.speedup_vs_esbm}x Speedup (~13x reduction)
                </span>
              </div>
            </div>
          </div>
        )}
      </MacWindow>

      {/* ================================================================= */}
      {/* SECTION 2: Dataset Batch Benchmark & IEEE Tables Window           */}
      {/* ================================================================= */}
      <MacWindow
        title="Dataset Batch Benchmark & IEEE Tables"
        badge={`${benchmarkData?.dataset_total_images || 11} TEST VEHICLES`}
        headerRight={
          <div className="flex items-center gap-2">
            <button
              onClick={runBatchBenchmark}
              disabled={isBenchLoading}
              className="mac-btn py-0.5 px-2.5 text-xs"
            >
              <RotateCcw className={`w-3.5 h-3.5 ${isBenchLoading ? 'animate-spin' : ''}`} />
              <span>Re-run Evaluation</span>
            </button>

            <button
              onClick={handleCopyMarkdown}
              disabled={!benchmarkData}
              className="mac-btn mac-btn-primary py-0.5 px-2.5 text-xs"
            >
              {copiedMarkdown ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copiedMarkdown ? 'Copied Markdown!' : 'Copy IEEE Tables'}</span>
            </button>

            <button
              onClick={handleDownloadMarkdown}
              disabled={!benchmarkData}
              className="mac-btn mac-btn-accent py-0.5 px-2.5 text-xs"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export .md</span>
            </button>
          </div>
        }
      >
        <p className="text-xs text-[#555] font-sans -mt-2">
          Empirical evaluation across all 4 syllabus benchmark tables evaluated against Pascal VOC XML ground truth annotations.
        </p>

        {isBenchLoading ? (
          <div className="py-12 flex flex-col items-center justify-center gap-3">
            <div className="w-8 h-8 border-4 border-[#0050C8]/20 border-t-[#0050C8] rounded-full animate-spin" />
            <span className="text-xs font-mono text-[#555]">
              Running batch evaluation on dataset vehicles...
            </span>
          </div>
        ) : benchmarkData ? (
          <div className="flex flex-col gap-6">
            {/* Table I: Spatial Enhancement */}
            <div className="flex flex-col gap-2">
              <h3 className="text-xs font-bold text-[#0050C8] uppercase tracking-wider flex items-center justify-between font-mono">
                <span>{benchmarkData.enhancement_benchmark.table_title}</span>
                <span className="text-[10px] text-[#555]">Syllabus Units 2 &amp; 7</span>
              </h3>
              <div className="overflow-x-auto mac-well rounded">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-[#E5E0D5] text-[#1C1C1E] text-[10px] uppercase font-bold border-b-2 border-[#1C1C1E]">
                    <tr>
                      <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Enhancement Technique</th>
                      <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Mean MSE</th>
                      <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Mean PSNR (dB)</th>
                      <th className="py-2 px-3">Shannon Entropy (bits/pixel)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1C1C1E]/20 bg-[#FFFFFF]">
                    {benchmarkData.enhancement_benchmark.rows.map((r, i) => (
                      <tr key={i} className="hover:bg-[#F5F2EB]">
                        <td className="py-2 px-3 font-semibold text-[#1C1C1E]">
                          {r.method_name}
                        </td>
                        <td className="py-2 px-3 text-[#555]">{r.mean_mse}</td>
                        <td className="py-2 px-3 text-[#157A3E] font-bold">{r.mean_psnr_db} dB</td>
                        <td className="py-2 px-3 text-[#0050C8] font-bold">{r.mean_entropy_bits}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Table II: Edge Detection & ROI Localization */}
            <div className="flex flex-col gap-2">
              <h3 className="text-xs font-bold text-[#D97706] uppercase tracking-wider flex items-center justify-between font-mono">
                <span>{benchmarkData.edge_localization_benchmark.table_title}</span>
                <span className="text-[10px] text-[#555]">Syllabus Units 4 &amp; 5</span>
              </h3>
              <div className="overflow-x-auto mac-well rounded">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-[#E5E0D5] text-[#1C1C1E] text-[10px] uppercase font-bold border-b-2 border-[#1C1C1E]">
                    <tr>
                      <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Operator + 17x5 Closing</th>
                      <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Mean IoU</th>
                      <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Precision</th>
                      <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Recall</th>
                      <th className="py-2 px-3">Mean Latency (ms)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1C1C1E]/20 bg-[#FFFFFF]">
                    {benchmarkData.edge_localization_benchmark.rows.map((r, i) => (
                      <tr key={i} className="hover:bg-[#F5F2EB]">
                        <td className="py-2 px-3 font-semibold text-[#1C1C1E]">
                          {r.operator_name}
                        </td>
                        <td className="py-2 px-3 text-[#0050C8] font-bold">{r.mean_iou}</td>
                        <td className="py-2 px-3 text-[#555]">{r.precision}</td>
                        <td className="py-2 px-3 text-[#555]">{r.recall}</td>
                        <td className="py-2 px-3 text-[#555]">{r.mean_latency_ms} ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Table III: Binarization & Character Segmentation */}
            <div className="flex flex-col gap-2">
              <h3 className="text-xs font-bold text-[#157A3E] uppercase tracking-wider flex items-center justify-between font-mono">
                <span>{benchmarkData.segmentation_ocr_benchmark.table_title}</span>
                <span className="text-[10px] text-[#555]">Syllabus Unit 6</span>
              </h3>
              <div className="overflow-x-auto mac-well rounded">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-[#E5E0D5] text-[#1C1C1E] text-[10px] uppercase font-bold border-b-2 border-[#1C1C1E]">
                    <tr>
                      <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Thresholding Method</th>
                      <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Mean Otsu τ*</th>
                      <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Char Seg Rate (%)</th>
                      <th className="py-2 px-3">Plate OCR Accuracy (%)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1C1C1E]/20 bg-[#FFFFFF]">
                    {benchmarkData.segmentation_ocr_benchmark.rows.map((r, i) => (
                      <tr key={i} className="hover:bg-[#F5F2EB]">
                        <td className="py-2 px-3 font-semibold text-[#1C1C1E]">
                          {r.method_name}
                        </td>
                        <td className="py-2 px-3 text-[#555]">{r.mean_tau_star}</td>
                        <td className="py-2 px-3 text-[#157A3E] font-bold">
                          {r.char_segmentation_rate_pct}%
                        </td>
                        <td className="py-2 px-3 text-[#0050C8] font-bold">
                          {r.plate_ocr_accuracy_pct}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Table IV: Motion Estimation Benchmark */}
            <div className="flex flex-col gap-2">
              <h3 className="text-xs font-bold text-[#6B21A8] uppercase tracking-wider flex items-center justify-between font-mono">
                <span>{benchmarkData.motion_estimation_benchmark.table_title}</span>
                <span className="text-[10px] text-[#555]">Syllabus Unit 8</span>
              </h3>
              <div className="overflow-x-auto mac-well rounded">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-[#E5E0D5] text-[#1C1C1E] text-[10px] uppercase font-bold border-b-2 border-[#1C1C1E]">
                    <tr>
                      <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Algorithm</th>
                      <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Search Points / Block</th>
                      <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Speedup Factor</th>
                      <th className="py-2 px-3 border-r border-[#1C1C1E]/20">Prediction PSNR (dB)</th>
                      <th className="py-2 px-3">Latency (ms)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1C1C1E]/20 bg-[#FFFFFF]">
                    {benchmarkData.motion_estimation_benchmark.rows.map((r, i) => (
                      <tr key={i} className="hover:bg-[#F5F2EB]">
                        <td className="py-2 px-3 font-semibold text-[#1C1C1E]">
                          {r.algorithm_name}
                        </td>
                        <td className="py-2 px-3 text-[#555]">{r.search_points_per_block}</td>
                        <td className="py-2 px-3 text-[#157A3E] font-bold">{r.speedup_factor}x</td>
                        <td className="py-2 px-3 text-[#0050C8] font-bold">{r.prediction_psnr_db} dB</td>
                        <td className="py-2 px-3 text-[#555]">{r.latency_ms} ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ) : null}
      </MacWindow>
    </div>
  );
}
