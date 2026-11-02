import React, { useState } from 'react';
import {
  CheckCircle2,
  Clock,
  Crosshair,
  Percent,
  Sparkles,
  Layers,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  Info,
} from 'lucide-react';
import StageCard from './StageCard';
import MacWindow from './MacWindow';

export default function PipelineResults({ results, isProcessing, error }) {
  if (error) {
    return (
      <MacWindow title="Pipeline Execution Error" badge="ALERT">
        <div className="p-6 flex flex-col items-center text-center gap-3 bg-[#FEE2E2] rounded border-2 border-[#C81E1E]">
          <div className="w-12 h-12 rounded-full bg-[#FFFFFF] border-2 border-[#C81E1E] flex items-center justify-center text-[#C81E1E]">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-[#C81E1E] font-mono">
              Classical Pipeline Execution Halted
            </h3>
            <p className="text-xs text-[#555] font-sans max-w-md mt-1 leading-relaxed">
              {error}
            </p>
          </div>
        </div>
      </MacWindow>
    );
  }

  if (isProcessing) {
    return (
      <MacWindow title="Processing Pipeline..." badge="EXECUTING">
        <div className="py-16 flex flex-col items-center justify-center text-center gap-4 bg-[#FFFFFF] mac-well rounded">
          <div className="w-12 h-12 border-4 border-[#0050C8]/20 border-t-[#0050C8] rounded-full animate-spin" />
          <div>
            <h3 className="text-sm font-bold text-[#1C1C1E] font-mono">
              Executing 7-Stage Digital Image Processing Pipeline
            </h3>
            <p className="text-xs text-[#555] font-sans max-w-md mt-1">
              Applying CLAHE enhancement, vertical Sobel filtering, mathematical morphology closing, Otsu segmentation, and OCR extraction...
            </p>
          </div>
        </div>
      </MacWindow>
    );
  }

  if (!results || !results.stages || results.stages.length === 0) {
    return (
      <MacWindow title="Pipeline Output & Inspection Workbench" badge="IDLE">
        <div className="py-16 flex flex-col items-center justify-center text-center gap-4 bg-[#FFFFFF] mac-well rounded">
          <div className="w-12 h-12 rounded bg-[#E5E0D5] border-2 border-[#1C1C1E] flex items-center justify-center text-[#0050C8] shadow-[2px_2px_0px_#1C1C1E]">
            <Layers className="w-6 h-6" />
          </div>
          <div className="max-w-md">
            <h3 className="text-sm font-bold text-[#1C1C1E] font-mono">
              No Pipeline Transformations Computed Yet
            </h3>
            <p className="text-xs text-[#555] font-sans mt-1 leading-relaxed">
              Select a benchmark car on the left or upload an image and click "Execute 7-Stage Classical IVP Pipeline" to inspect intermediate visual stages.
            </p>
          </div>
        </div>
      </MacWindow>
    );
  }

  // Extract Summary Metrics
  const plateText =
    results.plate_text || results.recognized_plate || results.plate || 'UNREADABLE';
  const confidence =
    results.confidence !== undefined
      ? Number(results.confidence > 1 ? results.confidence : results.confidence * 100).toFixed(1)
      : '0.0';
  const plateType = results.plate_type || 'White Private';
  const totalTime =
    results.total_time_ms ??
    results.execution_time_ms ??
    results.stages.reduce((sum, s) => sum + (s.execution_time_ms || s.time_ms || 0), 0);
  const iouScore =
    results.ground_truth?.iou_hybrid ??
    results.ground_truth?.iou_classical ??
    results.iou ??
    null;

  const getHsrpClass = (type) => {
    const t = (type || '').toLowerCase();
    if (t.includes('yellow') || t.includes('commercial')) return 'hsrp-yellow-commercial';
    if (t.includes('green') || t.includes('ev')) return 'hsrp-green-ev';
    return 'hsrp-white-private';
  };

  const formattedPlate = (plateText && plateText !== 'UNREADABLE')
    ? plateText
        .toUpperCase()
        .replace(/\s+/g, '')
        .replace(/^([A-Z]{2})(\d{2})([A-Z]{1,2})(\d{4})$/, '$1 $2 $3 $4')
    : 'UNREADABLE';

  return (
    <div className="flex flex-col gap-5">
      {/* Top Summary Banner Mac Window */}
      <MacWindow
        title="Pipeline Execution Summary"
        badge="RECOGNITION"
      >
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-2 border-b-2 border-[#1C1C1E]/20">
          {/* Recognized HSRP Plate */}
          <div className="flex flex-col gap-1.5">
            <span className="text-[10px] font-mono font-bold text-[#555] uppercase tracking-wider">
              OCR Recognition Output
            </span>
            <div className="flex items-center gap-3">
              <div className={`hsrp-plate h-11 ${getHsrpClass(plateType)}`}>
                <div className="hsrp-ind-strip">
                  <span className="text-[6px]">❂</span>
                  <span>IND</span>
                </div>
                <div className="px-3 font-mono font-black text-lg md:text-xl tracking-widest">
                  {formattedPlate}
                </div>
              </div>

              <div className="flex flex-col font-mono text-xs">
                <span className="font-bold text-[#1C1C1E]">{plateType}</span>
                <span className="text-[10px] text-[#555]">Standard Indian HSRP</span>
              </div>
            </div>
          </div>

          {/* Quick Metrics Cards */}
          <div className="grid grid-cols-3 gap-2 font-mono">
            {/* Confidence */}
            <div className="bg-[#FFFFFF] mac-well-muted p-2 rounded flex flex-col">
              <div className="flex items-center gap-1 text-[10px] text-[#555] font-bold">
                <Percent className="w-3 h-3 text-[#157A3E]" />
                <span>Confidence</span>
              </div>
              <span className="text-sm font-bold text-[#157A3E] mt-0.5">
                {confidence ? `${confidence}%` : 'N/A'}
              </span>
            </div>

            {/* Total Execution Time */}
            <div className="bg-[#FFFFFF] mac-well-muted p-2 rounded flex flex-col">
              <div className="flex items-center gap-1 text-[10px] text-[#555] font-bold">
                <Clock className="w-3 h-3 text-[#D97706]" />
                <span>Latency</span>
              </div>
              <span className="text-sm font-bold text-[#D97706] mt-0.5">
                {typeof totalTime === 'number' ? totalTime.toFixed(1) : totalTime} ms
              </span>
            </div>

            {/* IoU Score */}
            <div className="bg-[#FFFFFF] mac-well-muted p-2 rounded flex flex-col">
              <div className="flex items-center gap-1 text-[10px] text-[#555] font-bold">
                <Crosshair className="w-3 h-3 text-[#0050C8]" />
                <span>Mean IoU</span>
              </div>
              <span className="text-sm font-bold text-[#0050C8] mt-0.5">
                {iouScore !== null && iouScore !== undefined ? Number(iouScore).toFixed(4) : 'N/A'}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between text-xs font-mono font-bold text-[#1C1C1E]">
          <div className="flex items-center gap-1.5">
            <Layers className="w-4 h-4 text-[#0050C8]" />
            <span>Interactive Syllabus Units Breakdown ({results.stages.length} Stages):</span>
          </div>
        </div>
      </MacWindow>

      {/* 7 Pipeline Stages Cards */}
      <div className="flex flex-col gap-4">
        {results.stages.map((stage, idx) => (
          <StageCard
            key={stage.stage_number || idx}
            stage={stage}
            defaultExpanded={idx < 2}
          />
        ))}
      </div>
    </div>
  );
}
