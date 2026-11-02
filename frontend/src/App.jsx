import React, { useState, useCallback, useEffect } from 'react';
import Header from './components/Header';
import LiveSurveillanceDashboard from './components/LiveSurveillanceDashboard';
import ImageUpload from './components/ImageUpload';
import PipelineConfig, { DEFAULT_PIPELINE_CONFIG } from './components/PipelineConfig';
import PipelineResults from './components/PipelineResults';
import VideoMotionBenchmark from './components/VideoMotionBenchmark';
import { processImage } from './api/client';

export default function App() {
  const [activeTab, setActiveTab] = useState('surveillance'); // 'surveillance' | 'workbench' | 'motion_benchmark'
  const [pipelineConfig, setPipelineConfig] = useState(DEFAULT_PIPELINE_CONFIG);
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedDatasetFilename, setSelectedDatasetFilename] = useState('HSRP_0_MH12.png');
  const [pipelineResults, setPipelineResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingError, setProcessingError] = useState(null);

  // Execute ALPR Pipeline
  const runPipeline = useCallback(
    async (overrideFile = null, overrideDatasetFilename = null, overrideConfig = null) => {
      let fileToUse = overrideFile;
      let datasetToUse = overrideDatasetFilename;

      if (overrideFile === null && overrideDatasetFilename === null) {
        fileToUse = selectedFile;
        datasetToUse = selectedDatasetFilename;
      } else if (overrideDatasetFilename) {
        fileToUse = null;
        setSelectedDatasetFilename(overrideDatasetFilename);
        setSelectedFile(null);
      } else if (overrideFile) {
        datasetToUse = null;
        setSelectedFile(overrideFile);
        setSelectedDatasetFilename(null);
      }

      const configToUse = overrideConfig || pipelineConfig;

      if (!fileToUse && !datasetToUse) {
        setProcessingError('Please select a file or pick an image from the dataset.');
        return;
      }

      setIsProcessing(true);
      setProcessingError(null);

      try {
        const data = await processImage(fileToUse, datasetToUse, configToUse);
        setPipelineResults(data);
      } catch (err) {
        console.error('ALPR pipeline processing error:', err);
        setProcessingError(
          err.message || 'Image processing failed. Ensure backend server is running on port 8000.'
        );
      } finally {
        setIsProcessing(false);
      }
    },
    [selectedFile, selectedDatasetFilename, pipelineConfig]
  );

  // Auto-run pipeline on initial load for HSRP_0_MH12.png
  useEffect(() => {
    runPipeline(null, 'HSRP_0_MH12.png');
  }, []);

  // Header quick metrics
  const recognizedPlate =
    pipelineResults?.plate_text ||
    pipelineResults?.recognized_plate ||
    pipelineResults?.plate ||
    null;
  const plateType = pipelineResults?.plate_type || 'white_private';
  const confidence = pipelineResults?.confidence;
  const processingTime =
    pipelineResults?.total_time_ms ??
    pipelineResults?.execution_time_ms ??
    (pipelineResults?.stages
      ? pipelineResults.stages.reduce((acc, s) => acc + (s.execution_time_ms || s.time_ms || 0), 0)
      : null);

  return (
    <div className="min-h-screen vintage-mac-desktop text-[#1C1C1E] flex flex-col font-sans selection:bg-[#0050C8] selection:text-white">
      {/* Top Application Header with Authentic Six-Color Apple Ribbon */}
      <Header
        recognizedPlate={recognizedPlate}
        plateType={plateType}
        confidence={confidence}
        processingTime={processingTime}
        activeTab={activeTab}
        onSelectTab={setActiveTab}
      />

      {/* Main Tab Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* VIEW 1: Live Surveillance & Toll Dashboard */}
        {activeTab === 'surveillance' && (
          <LiveSurveillanceDashboard
            currentImage={selectedFile || selectedDatasetFilename}
            pipelineResults={pipelineResults}
            onProcessImage={runPipeline}
            isProcessing={isProcessing}
          />
        )}

        {/* VIEW 2: Interactive 8-Stage IVP Syllabus Workbench */}
        {activeTab === 'workbench' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left Column: Upload & Parameters (5 cols) */}
            <div className="lg:col-span-5 flex flex-col gap-6 lg:sticky lg:top-4 lg:self-start lg:max-h-[calc(100vh-2rem)] lg:overflow-y-auto pr-1">
              <ImageUpload
                selectedFile={selectedFile}
                selectedDatasetFilename={selectedDatasetFilename}
                onFileSelect={(file) => {
                  setSelectedFile(file);
                  if (file) setSelectedDatasetFilename(null);
                }}
                onDatasetSelect={(filename) => {
                  setSelectedDatasetFilename(filename);
                  if (filename) setSelectedFile(null);
                }}
                onProcess={() => runPipeline()}
                isProcessing={isProcessing}
              />

              <PipelineConfig
                config={pipelineConfig}
                onChange={setPipelineConfig}
                disabled={isProcessing}
              />
            </div>

            {/* Right Column: 8-Stage Pipeline Results (7 cols) */}
            <div className="lg:col-span-7">
              <PipelineResults
                results={pipelineResults}
                isProcessing={isProcessing}
                error={processingError}
              />
            </div>
          </div>
        )}

        {/* VIEW 3: Video Motion Lab & Empirical Batch Benchmark */}
        {activeTab === 'motion_benchmark' && (
          <VideoMotionBenchmark />
        )}
      </main>

      {/* Classic Macintosh Desktop Footer */}
      <footer className="mt-auto border-t-2 border-[#1C1C1E] bg-[#F5F2EB] py-3 text-center text-xs font-mono text-[#555] shadow-[0_-2px_0px_#1C1C1E]">
        <div className="max-w-7xl mx-auto px-4 flex flex-wrap items-center justify-between gap-2">
          <span>SentinelPlate ALPR &bull; Vintage Macintosh System 7 IVP Command Center</span>
          <span className="mac-tag mac-tag-blue text-[10px]">68040 / PowerPC Emulation Active</span>
          <span>Digital Image &amp; Video Processing Syllabus Units 1–8</span>
        </div>
      </footer>
    </div>
  );
}
