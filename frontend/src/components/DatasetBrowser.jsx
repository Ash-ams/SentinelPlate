import React, { useState, useEffect } from 'react';
import {
  Database,
  CheckCircle2,
  XCircle,
  Play,
  Search,
  RefreshCw,
  FileText,
  AlertTriangle,
  Tag,
} from 'lucide-react';
import { listDataset } from '../api/client';

export default function DatasetBrowser({ onSelectImage, activeFilename }) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  const fetchItems = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listDataset();
      setItems(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error('Failed to load dataset in DatasetBrowser:', err);
      setError('Failed to retrieve dataset items. Backend might still be initializing.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchItems();
  }, []);

  const filteredItems = items.filter((item) => {
    const q = searchQuery.toLowerCase();
    const nameMatch = item.filename?.toLowerCase().includes(q);
    const plateMatch = item.plate_text?.toLowerCase().includes(q);
    return nameMatch || plateMatch;
  });

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-xl flex flex-col gap-6">
      {/* Header and Search Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-gray-800">
        <div>
          <div className="flex items-center gap-2">
            <Database className="w-5 h-5 text-indigo-400" />
            <h2 className="text-base font-semibold text-white">
              Benchmark Dataset Repository
            </h2>
          </div>
          <p className="text-xs text-gray-400 mt-0.5">
            Synthetic & field-acquired vehicle images with Pascal VOC XML ground-truth annotations
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Search bar */}
          <div className="relative">
            <Search className="w-4 h-4 text-gray-500 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search filename or plate..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-gray-950 border border-gray-800 text-xs text-white rounded-xl pl-9 pr-3.5 py-2 w-56 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500"
            />
          </div>

          {/* Refresh button */}
          <button
            onClick={fetchItems}
            disabled={loading}
            className="p-2 rounded-xl bg-gray-950 border border-gray-800 text-gray-400 hover:text-white hover:border-gray-700 transition-colors disabled:opacity-50"
            title="Reload dataset"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="bg-amber-950/40 border border-amber-800 text-amber-200 text-xs px-4 py-3 rounded-xl flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            onClick={fetchItems}
            className="text-amber-400 underline font-semibold hover:text-white"
          >
            Retry
          </button>
        </div>
      )}

      {/* Loading Skeleton */}
      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((n) => (
            <div
              key={n}
              className="bg-gray-950 border border-gray-800/80 rounded-xl p-3 flex flex-col gap-3 animate-pulse"
            >
              <div className="h-40 bg-gray-900 rounded-lg" />
              <div className="h-4 bg-gray-900 rounded w-2/3" />
              <div className="h-3 bg-gray-900 rounded w-1/2" />
              <div className="h-8 bg-gray-900 rounded-lg mt-2" />
            </div>
          ))}
        </div>
      ) : filteredItems.length === 0 ? (
        <div className="py-12 flex flex-col items-center justify-center text-center gap-2 text-gray-500">
          <Database className="w-8 h-8 text-gray-600" />
          <p className="text-sm">No dataset items match your filter.</p>
        </div>
      ) : (
        /* Image Cards Grid (4 columns) */
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {filteredItems.map((item) => {
            const hasAnnotation = Boolean(item.has_annotation || item.bndbox);
            const isCurrentActive = activeFilename === item.filename;
            const imgSrc = `/api/dataset/image/${item.filename}`;

            return (
              <div
                key={item.filename}
                className={`bg-gray-950 rounded-xl border overflow-hidden flex flex-col transition-all duration-200 hover:border-indigo-500/50 hover:shadow-lg hover:shadow-indigo-500/10 ${
                  isCurrentActive
                    ? 'border-indigo-500 ring-2 ring-indigo-500/30'
                    : 'border-gray-800'
                }`}
              >
                {/* Thumbnail Container */}
                <div className="relative h-44 w-full bg-gray-925 flex items-center justify-center p-2 border-b border-gray-800/80 overflow-hidden group">
                  <img
                    src={imgSrc}
                    alt={item.filename}
                    className="max-h-full max-w-full object-contain rounded transition-transform duration-200 group-hover:scale-105"
                    onError={(e) => {
                      // Fallback placeholder with canvas look
                      e.target.onerror = null;
                      e.target.src =
                        'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="300" height="200" viewBox="0 0 300 200" fill="%230f172a"><rect width="300" height="200"/><circle cx="150" cy="90" r="30" fill="%231e293b"/><path d="M70 160 C 100 120, 200 120, 230 160 Z" fill="%231e293b"/><text x="50%" y="85%" fill="%2364748b" dominant-baseline="middle" text-anchor="middle" font-family="monospace" font-size="12">' +
                        item.filename +
                        '</text></svg>';
                    }}
                  />

                  {/* Top-right Status Pill */}
                  <div className="absolute top-2.5 right-2.5">
                    {hasAnnotation ? (
                      <span className="flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-950/80 text-emerald-400 border border-emerald-800/80 backdrop-blur-sm">
                        <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                        <span>Annotated</span>
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-gray-900/80 text-gray-400 border border-gray-700/80 backdrop-blur-sm">
                        <XCircle className="w-3 h-3 text-gray-500" />
                        <span>Raw</span>
                      </span>
                    )}
                  </div>
                </div>

                {/* Card Body */}
                <div className="p-3.5 flex flex-col gap-2.5 flex-1 justify-between">
                  <div>
                    {/* Filename & Dimensions */}
                    <div className="flex items-center justify-between">
                      <span className="font-mono font-semibold text-xs text-white truncate">
                        {item.filename}
                      </span>
                      <span className="text-[10px] text-gray-500 font-mono">
                        {item.width && item.height ? `${item.width}×${item.height}` : '640×480'}
                      </span>
                    </div>

                    {/* Ground truth plate badge */}
                    {item.plate_text && (
                      <div className="mt-2 flex items-center justify-between bg-gray-900 px-2.5 py-1.5 rounded-lg border border-gray-800">
                        <span className="text-[10px] text-gray-400 font-medium">
                          Plate Ground Truth
                        </span>
                        <span className="font-mono font-bold text-xs text-indigo-300 tracking-wider">
                          {item.plate_text}
                        </span>
                      </div>
                    )}

                    {/* Bounding Box Info */}
                    {item.bndbox && (
                      <div className="mt-2 text-[10px] font-mono text-gray-400 bg-gray-900/60 p-2 rounded-lg border border-gray-800/60 flex flex-col gap-0.5">
                        <span className="text-gray-500 font-sans font-medium text-[9px] uppercase tracking-wider">
                          BndBox [xmin, ymin, xmax, ymax]
                        </span>
                        <span className="text-gray-300">
                          [{item.bndbox.xmin}, {item.bndbox.ymin}, {item.bndbox.xmax}, {item.bndbox.ymax}]
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Process Action Button */}
                  <button
                    onClick={() => onSelectImage(item.filename, true)}
                    className="w-full py-2 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 bg-indigo-600/20 hover:bg-indigo-600 text-indigo-300 hover:text-white border border-indigo-500/40 hover:border-transparent transition-all duration-150 cursor-pointer shadow-sm active:scale-[0.98]"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>Process in Pipeline</span>
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
