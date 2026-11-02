import React from 'react';
import {
  SlidersHorizontal,
  Sun,
  Wand2,
  Filter,
  Activity,
  Layers,
  Binary,
  RotateCcw,
} from 'lucide-react';
import MacWindow from './MacWindow';

export const DEFAULT_PIPELINE_CONFIG = {
  gamma: 1.0,
  point_processing: 'none',
  filter_type: 'gaussian',
  edge_detector: 'sobel',
  enhancement: 'clahe',
  threshold_method: 'otsu',
};

export default function PipelineConfig({ config, onChange, disabled }) {
  const handleChange = (key, value) => {
    onChange({
      ...config,
      [key]: value,
    });
  };

  const handleReset = () => {
    onChange(DEFAULT_PIPELINE_CONFIG);
  };

  return (
    <MacWindow
      title="Interactive Pipeline Parameters"
      badge="CONFIG"
      headerRight={
        <button
          onClick={handleReset}
          disabled={disabled}
          title="Reset to default settings"
          className="mac-btn py-0 px-2 text-[11px]"
        >
          <RotateCcw className="w-3 h-3 text-[#0050C8]" />
          <span>Reset</span>
        </button>
      }
    >
      <div className="flex flex-col gap-3 font-mono text-xs">
        {/* 1. Gamma Correction Slider */}
        <div className="flex flex-col gap-1 p-2 bg-[#FFFFFF] mac-well rounded">
          <div className="flex items-center justify-between">
            <label className="font-bold text-[#1C1C1E] flex items-center gap-1.5">
              <Sun className="w-3.5 h-3.5 text-[#D97706]" />
              <span>Gamma Correction (γ):</span>
            </label>
            <span className="mac-tag mac-tag-amber text-[10px]">
              γ = {Number(config.gamma || 1.0).toFixed(1)}
            </span>
          </div>
          <div className="flex items-center gap-2 pt-1">
            <span className="text-[10px] text-[#555]">0.1</span>
            <input
              type="range"
              min="0.1"
              max="5.0"
              step="0.1"
              value={config.gamma || 1.0}
              onChange={(e) => handleChange('gamma', parseFloat(e.target.value))}
              disabled={disabled}
              className="w-full h-2 bg-[#DDD8CE] rounded border border-[#1C1C1E] appearance-none cursor-pointer accent-[#0050C8] disabled:opacity-50"
            />
            <span className="text-[10px] text-[#555]">5.0</span>
          </div>
        </div>

        {/* 2. Point Processing Dropdown */}
        <div className="flex flex-col gap-1">
          <label className="font-bold text-[#1C1C1E] flex items-center gap-1.5">
            <Wand2 className="w-3.5 h-3.5 text-[#0050C8]" />
            <span>Point Processing:</span>
          </label>
          <select
            value={config.point_processing || 'none'}
            onChange={(e) => handleChange('point_processing', e.target.value)}
            disabled={disabled}
            className="w-full mac-input py-1.5 text-xs"
          >
            <option value="none">None (Identity)</option>
            <option value="negative">Image Negative (Inversion)</option>
            <option value="contrast_stretch">Contrast Stretching</option>
            <option value="log_transform">Logarithmic Transform</option>
            <option value="gamma">Power-Law (Gamma)</option>
          </select>
        </div>

        {/* 3. Spatial Filter Dropdown */}
        <div className="flex flex-col gap-1">
          <label className="font-bold text-[#1C1C1E] flex items-center gap-1.5">
            <Filter className="w-3.5 h-3.5 text-[#0050C8]" />
            <span>Spatial Filter Type:</span>
          </label>
          <select
            value={config.filter_type || 'gaussian'}
            onChange={(e) => handleChange('filter_type', e.target.value)}
            disabled={disabled}
            className="w-full mac-input py-1.5 text-xs"
          >
            <option value="gaussian">Gaussian Filter (Smoothing)</option>
            <option value="median">Median Filter (Salt & Pepper Denoise)</option>
            <option value="laplacian_sharpen">Laplacian Sharpening</option>
          </select>
        </div>

        {/* 4. Edge Detector Dropdown */}
        <div className="flex flex-col gap-1">
          <label className="font-bold text-[#1C1C1E] flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-[#157A3E]" />
            <span>Edge Detector:</span>
          </label>
          <select
            value={config.edge_detector || 'sobel'}
            onChange={(e) => handleChange('edge_detector', e.target.value)}
            disabled={disabled}
            className="w-full mac-input py-1.5 text-xs"
          >
            <option value="sobel">Sobel Operator (G_x, G_y Magnitude)</option>
            <option value="prewitt">Prewitt Operator</option>
            <option value="log">Laplacian of Gaussian (LoG)</option>
            <option value="canny">Canny Edge Detector</option>
          </select>
        </div>

        {/* 5. Histogram Enhancement Dropdown */}
        <div className="flex flex-col gap-1">
          <label className="font-bold text-[#1C1C1E] flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-[#D97706]" />
            <span>Histogram Enhancement:</span>
          </label>
          <select
            value={config.enhancement || 'clahe'}
            onChange={(e) => handleChange('enhancement', e.target.value)}
            disabled={disabled}
            className="w-full mac-input py-1.5 text-xs"
          >
            <option value="clahe">CLAHE (Contrast Limited Adaptive)</option>
            <option value="histogram_eq">Global Histogram Equalization</option>
            <option value="none">None (Raw Grayscale)</option>
          </select>
        </div>

        {/* 6. Thresholding Method Dropdown */}
        <div className="flex flex-col gap-1">
          <label className="font-bold text-[#1C1C1E] flex items-center gap-1.5">
            <Binary className="w-3.5 h-3.5 text-[#0050C8]" />
            <span>Threshold Binarization:</span>
          </label>
          <select
            value={config.threshold_method || 'otsu'}
            onChange={(e) => handleChange('threshold_method', e.target.value)}
            disabled={disabled}
            className="w-full mac-input py-1.5 text-xs"
          >
            <option value="otsu">Otsu's Global Variance Thresholding</option>
            <option value="adaptive">Adaptive Gaussian Thresholding</option>
            <option value="global">Fixed Global Thresholding (T = 127)</option>
          </select>
        </div>
      </div>
    </MacWindow>
  );
}
