import React, { useState, useRef, useEffect } from 'react';
import { Upload, Image as ImageIcon, Sparkles, AlertCircle, CheckCircle2, ChevronDown, RefreshCw, X, FileText } from 'lucide-react';
import { listDataset } from '../api/client';
import MacWindow from './MacWindow';

export default function ImageUpload({
  selectedFile,
  selectedDatasetFilename,
  onFileSelect,
  onDatasetSelect,
  onProcess,
  isProcessing,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [datasetItems, setDatasetItems] = useState([]);
  const [loadingDataset, setLoadingDataset] = useState(false);
  const fileInputRef = useRef(null);

  // Load dataset items on mount
  useEffect(() => {
    let isMounted = true;
    const fetchDataset = async () => {
      setLoadingDataset(true);
      try {
        const items = await listDataset();
        if (isMounted) {
          setDatasetItems(Array.isArray(items) ? items : []);
        }
      } catch (err) {
        console.error('Error fetching dataset in ImageUpload:', err);
      } finally {
        if (isMounted) setLoadingDataset(false);
      }
    };
    fetchDataset();
    return () => {
      isMounted = false;
    };
  }, []);

  // Update preview when file or dataset selection changes
  useEffect(() => {
    if (selectedFile) {
      const url = URL.createObjectURL(selectedFile);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else if (selectedDatasetFilename) {
      setPreviewUrl(`/api/static-images/${selectedDatasetFilename}`);
    } else {
      setPreviewUrl(null);
    }
  }, [selectedFile, selectedDatasetFilename]);

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onDatasetSelect(null);
        onFileSelect(file);
      }
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      onDatasetSelect(null);
      onFileSelect(file);
    }
  };

  const handleDatasetChange = (e) => {
    const val = e.target.value;
    if (val) {
      onFileSelect(null);
      onDatasetSelect(val);
    } else {
      onDatasetSelect(null);
    }
  };

  const handleClear = () => {
    onFileSelect(null);
    onDatasetSelect(null);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const currentDatasetItem = datasetItems.find(
    (item) => item.filename === selectedDatasetFilename
  );

  const canProcess = Boolean(selectedFile || selectedDatasetFilename);

  return (
    <MacWindow
      title="Input Image & Dataset Selection"
      badge="VIEWPORT"
      headerRight={
        canProcess ? (
          <button
            onClick={handleClear}
            disabled={isProcessing}
            className="mac-btn py-0 px-2 text-[11px]"
            title="Clear current selection"
          >
            <X className="w-3 h-3 text-[#C81E1E]" />
            <span>Clear</span>
          </button>
        ) : null
      }
    >
      {/* Dataset Picker Dropdown */}
      <div className="flex flex-col gap-1">
        <label className="text-xs font-bold font-mono text-[#1C1C1E] flex items-center justify-between">
          <span>Choose Benchmark Car Sample:</span>
          {loadingDataset && (
            <span className="text-[10px] text-[#555] flex items-center gap-1 font-mono">
              <RefreshCw className="w-2.5 h-2.5 animate-spin" /> Loading
            </span>
          )}
        </label>
        <div className="relative">
          <select
            value={selectedDatasetFilename || ''}
            onChange={handleDatasetChange}
            disabled={isProcessing}
            className="w-full mac-input py-2 text-xs"
          >
            <option value="">-- Select a benchmark image --</option>
            {datasetItems.map((item) => (
              <option key={item.filename} value={item.filename}>
                {item.filename} {item.plate_text ? `(${item.plate_text})` : ''}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Drag & Drop Upload Container */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isProcessing && fileInputRef.current?.click()}
        className={`relative border-2 border-dashed border-[#1C1C1E] rounded p-4 text-center cursor-pointer transition-all flex flex-col items-center justify-center min-h-[170px] ${
          isDragging
            ? 'bg-[#D9E8FF]'
            : 'bg-[#FFFFFF] hover:bg-[#ECE7DD]'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".png,.jpg,.jpeg"
          onChange={handleFileInput}
          className="hidden"
          disabled={isProcessing}
        />

        {previewUrl ? (
          <div className="relative w-full flex flex-col items-center gap-2">
            <div className="relative max-h-48 w-full flex items-center justify-center bg-[#DDD8CE] mac-well rounded p-1">
              <img
                src={previewUrl}
                alt="Selected preview"
                className="max-h-44 w-auto object-contain"
                onError={(e) => {
                  const fb = `http://localhost:8000/api/static-images/${selectedDatasetFilename}`;
                  if (e.target.src !== fb) e.target.src = fb;
                }}
              />
            </div>
            <p className="text-xs font-mono font-bold text-[#1C1C1E] truncate max-w-xs">
              {selectedFile ? selectedFile.name : selectedDatasetFilename}
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 py-3">
            <div className="w-10 h-10 rounded bg-[#E5E0D5] border-2 border-[#1C1C1E] shadow-[1px_1px_0px_#1C1C1E] flex items-center justify-center text-[#1C1C1E]">
              <Upload className="w-5 h-5 text-[#0050C8]" />
            </div>
            <div>
              <p className="text-xs font-bold text-[#1C1C1E]">
                Click to browse or drop photograph here
              </p>
              <p className="text-[10px] text-[#555] font-mono mt-0.5">
                PNG, JPG or JPEG (Max 10MB)
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Ground Truth Bounding Box Info Card if Dataset item is chosen */}
      {currentDatasetItem && (
        <div className="bg-[#FFFFFF] mac-well rounded p-2.5 flex flex-col gap-1.5 text-xs text-[#1C1C1E] font-mono">
          <div className="flex items-center justify-between text-[#0050C8] font-bold">
            <span className="flex items-center gap-1.5">
              <FileText className="w-3.5 h-3.5" /> Pascal VOC XML Ground Truth
            </span>
            <span className="mac-tag mac-tag-blue text-[9px]">
              ANNOTATED
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2 pt-1 border-t border-[#1C1C1E]/20 text-[11px]">
            <div>
              <span className="text-[#555]">Target Plate: </span>
              <span className="font-bold text-[#1C1C1E]">
                {currentDatasetItem.plate_text || (currentDatasetItem.filename === 'HSRP_0_MH12.png' ? 'MH12DE1433' : 'UNSPECIFIED')}
              </span>
            </div>
            <div>
              <span className="text-[#555]">Category: </span>
              <span className="font-bold text-[#157A3E]">
                {currentDatasetItem.plate_type ? currentDatasetItem.plate_type.replace('_', ' ') : (currentDatasetItem.filename === 'Cars4.png' || currentDatasetItem.filename === 'Cars1.png' ? 'Yellow Commercial' : currentDatasetItem.filename === 'Cars2.png' ? 'Green EV' : 'White Private')}
              </span>
            </div>
            {currentDatasetItem.bndbox && (
              <div className="col-span-2 flex items-center justify-between text-[#555] font-mono text-[10px] bg-[#E5E0D5] px-2 py-0.5 rounded border border-[#1C1C1E]/30">
                <span>xmin: {currentDatasetItem.bndbox.xmin}</span>
                <span>ymin: {currentDatasetItem.bndbox.ymin}</span>
                <span>xmax: {currentDatasetItem.bndbox.xmax}</span>
                <span>ymax: {currentDatasetItem.bndbox.ymax}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Process Button */}
      <button
        onClick={onProcess}
        disabled={!canProcess || isProcessing}
        className="mac-btn mac-btn-primary w-full py-2.5 text-xs"
      >
        {isProcessing ? (
          <>
            <RefreshCw className="w-4 h-4 animate-spin" />
            <span>Executing 7-Stage Pipeline...</span>
          </>
        ) : (
          <>
            <Sparkles className="w-4 h-4" />
            <span>Execute 7-Stage Classical IVP Pipeline</span>
          </>
        )}
      </button>
    </MacWindow>
  );
}
