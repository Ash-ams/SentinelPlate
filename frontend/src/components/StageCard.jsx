import React, { useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Clock,
  Maximize2,
  X,
  CheckCircle,
  Info,
  Code,
  ZoomIn,
} from 'lucide-react';
import HistogramChart from './HistogramChart';
import LightboxModal from './LightboxModal';

export default function StageCard({ stage, defaultExpanded = false }) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const [modalImage, setModalImage] = useState(null);

  const stageNum = stage.stage_number || stage.stage || 1;
  const timeMs = stage.execution_time_ms ?? stage.time_ms ?? stage.duration_ms;
  const stageTitle = stage.stage_name || stage.name || stage.title || `Stage ${stageNum}`;

  // Normalize images list
  const imagesList = React.useMemo(() => {
    if (!stage.images) return [];
    if (Array.isArray(stage.images)) {
      return stage.images.map((item, idx) => ({
        title: item.title || item.name || item.label || `Image ${idx + 1}`,
        src: item.image || item.src || item.base64 || item.url || '',
      }));
    }
    if (typeof stage.images === 'object') {
      return Object.entries(stage.images).map(([title, val]) => ({
        title: title.replace(/_/g, ' ').toUpperCase(),
        src: typeof val === 'string' ? val : val?.image || val?.src || '',
      }));
    }
    return [];
  }, [stage.images]);

  // Normalize histograms list
  const histogramsList = React.useMemo(() => {
    if (!stage.histograms && !stage.histogram) return [];
    const source = stage.histograms || stage.histogram;
    if (Array.isArray(source)) {
      return source.map((h, i) => ({
        title: h.title || `Histogram ${i + 1}`,
        data: h.data || h,
      }));
    }
    if (source.labels && source.datasets) {
      return [{ title: source.title || 'Intensity Histogram', data: source }];
    }
    return [];
  }, [stage.histograms, stage.histogram]);

  // Format image source URI
  const formatImageSrc = (src) => {
    if (!src) return '';
    if (src.startsWith('data:') || src.startsWith('http://') || src.startsWith('https://') || src.startsWith('/')) {
      return src;
    }
    return `data:image/png;base64,${src}`;
  };

  const formatMetricVal = (val) => {
    if (val === null || val === undefined) return 'N/A';
    if (typeof val === 'number') {
      return Number.isInteger(val) ? val.toString() : val.toFixed(4);
    }
    if (typeof val === 'boolean') return val ? 'True' : 'False';
    if (typeof val === 'object') return JSON.stringify(val);
    return String(val);
  };

  return (
    <>
      {modalImage && (
        <LightboxModal
          image={modalImage}
          onClose={() => setModalImage(null)}
        />
      )}

      {/* Macintosh System 7 Window Frame */}
      <div className="mac-window flex flex-col">
        {/* Macintosh Titlebar with Pinstripes & Close Box */}
        <div
          className="mac-titlebar cursor-pointer"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {/* Classic Close Box */}
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              setIsExpanded(!isExpanded);
            }}
            className="mac-close-box"
            title="Expand / Collapse Stage"
          >
            <span className="w-1.5 h-1.5 bg-[#1C1C1E]" />
          </button>

          {/* Left Pinstripes */}
          <div className="mac-pinstripes" />

          {/* Centered Bold Title */}
          <div className="mac-title-text flex items-center gap-2">
            <span className="mac-tag mac-tag-blue text-[10px] py-0 px-1 font-mono">
              0{stageNum}
            </span>
            <span>{stageTitle}</span>
          </div>

          {/* Right Pinstripes */}
          <div className="mac-pinstripes" />

          {/* Right Latency Badge & Toggle Icon */}
          <div className="ml-2 flex items-center gap-2 flex-shrink-0">
            {timeMs !== undefined && timeMs !== null && (
              <span className="mac-tag mac-tag-gray text-[10px]">
                <Clock className="w-2.5 h-2.5 text-[#555]" />
                <span>{typeof timeMs === 'number' ? timeMs.toFixed(1) : timeMs} ms</span>
              </span>
            )}
            <div className="text-[#1C1C1E] p-0.5">
              {isExpanded ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
            </div>
          </div>
        </div>

        {/* Collapsible Content */}
        {isExpanded && (
          <div className="p-4 flex flex-col gap-4 bg-[#F5F2EB] animate-in fade-in-50 duration-150">
            {/* Description */}
            {stage.description && (
              <p className="text-xs text-[#333] leading-relaxed bg-[#FFFFFF] mac-well rounded p-2.5 font-sans">
                {stage.description}
              </p>
            )}

            {/* Images Grid */}
            {imagesList.length > 0 && (
              <div className="flex flex-col gap-2">
                <span className="text-xs font-bold font-mono text-[#1C1C1E] uppercase tracking-wider">
                  Generated Visualizations ({imagesList.length})
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {imagesList.map((imgItem, idx) => {
                    const fullSrc = formatImageSrc(imgItem.src);
                    return (
                      <div
                        key={idx}
                        className="mac-window-sm bg-[#FFFFFF] flex flex-col group cursor-pointer hover:border-[#0050C8] transition-all"
                        onClick={() =>
                          setModalImage({
                            src: fullSrc,
                            title: `${stageTitle}: ${imgItem.title}`,
                          })
                        }
                      >
                        {/* Image Title Bar */}
                        <div className="px-2.5 py-1.5 bg-[#E5E0D5] border-b-2 border-[#1C1C1E] flex items-center justify-between">
                          <span className="text-[11px] font-mono font-bold text-[#1C1C1E] truncate">
                            {imgItem.title}
                          </span>
                          <Maximize2 className="w-3.5 h-3.5 text-[#555] opacity-0 group-hover:opacity-100 transition-opacity" />
                        </div>

                        {/* Image Viewport */}
                        <div className="relative h-40 w-full flex items-center justify-center p-2 bg-[#DDD8CE] overflow-hidden">
                          <img
                            src={fullSrc}
                            alt={imgItem.title}
                            className="max-h-full max-w-full object-contain border border-[#1C1C1E]/30"
                          />
                          <div className="absolute inset-0 bg-black/5 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                            <span className="mac-tag mac-tag-blue text-[10px]">Click to Zoom</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Histograms Section */}
            {histogramsList.length > 0 && (
              <div className="flex flex-col gap-2">
                <span className="text-xs font-bold font-mono text-[#1C1C1E] uppercase tracking-wider">
                  Spectral Distributions &amp; Intensity Histograms
                </span>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {histogramsList.map((hist, idx) => (
                    <HistogramChart
                      key={idx}
                      title={hist.title}
                      data={hist.data}
                      height={180}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Metrics Section */}
            {stage.metrics && Object.keys(stage.metrics).length > 0 && (
              <div className="flex flex-col gap-2">
                <span className="text-xs font-bold font-mono text-[#1C1C1E] uppercase tracking-wider">
                  Stage Telemetry &amp; Classical Evaluation Metrics
                </span>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                  {Object.entries(stage.metrics).map(([key, val]) => (
                    <div
                      key={key}
                      className="bg-[#FFFFFF] mac-well rounded p-2 flex flex-col gap-0.5 font-mono"
                    >
                      <span className="text-[10px] text-[#555] uppercase truncate">
                        {key.replace(/_/g, ' ')}
                      </span>
                      <span className="text-xs font-bold text-[#1C1C1E] truncate">
                        {formatMetricVal(val)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Details Section */}
            {stage.details && (
              <div className="flex flex-col gap-1.5 font-mono">
                <div className="flex items-center gap-1.5 text-xs font-bold text-[#1C1C1E] uppercase">
                  <Code className="w-3.5 h-3.5 text-[#0050C8]" />
                  <span>Algorithm Parameters &amp; Implementation Details</span>
                </div>
                <div className="bg-[#FFFFFF] mac-well rounded p-2.5 overflow-x-auto text-[11px] text-[#333]">
                  <pre className="leading-relaxed whitespace-pre-wrap">
                    {typeof stage.details === 'string'
                      ? stage.details
                      : JSON.stringify(stage.details, null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </>
  );
}
