import React, { useEffect } from 'react';
import { X, ZoomIn, Info } from 'lucide-react';

/**
 * Vintage Macintosh Lightbox Zoom Modal for inspecting pipeline images in high fidelity.
 */
export default function LightboxModal({ image, onClose }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!image) return null;

  return (
    <div
      className="fixed inset-0 z-[999] flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-in fade-in duration-150"
      onClick={onClose}
    >
      <div
        className="mac-window max-w-4xl w-full max-h-[90vh] flex flex-col shadow-[8px_8px_0px_#1C1C1E] animate-in zoom-in-95 duration-150"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Titlebar */}
        <div className="mac-titlebar">
          <button
            onClick={onClose}
            className="mac-close-box"
            title="Close Lightbox (ESC)"
          >
            <span className="w-1.5 h-1.5 bg-[#1C1C1E]" />
          </button>
          <div className="mac-pinstripes" />
          <div className="mac-title-text flex items-center gap-2">
            <ZoomIn className="w-3.5 h-3.5 text-[#0050C8]" />
            <span>Inspection Lightbox: {image.title || 'Image Viewer'}</span>
          </div>
          <div className="mac-pinstripes" />
          <button
            onClick={onClose}
            className="ml-2 mac-btn text-[11px] py-0.5 px-2 font-mono"
          >
            ESC
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-4 bg-[#F5F2EB] flex flex-col gap-3 overflow-y-auto">
          {/* Inset Image Display */}
          <div className="mac-well p-3 flex items-center justify-center bg-[#DDD8CE] min-h-[350px] max-h-[65vh] overflow-hidden">
            <img
              src={image.src}
              alt={image.title || 'Zoomed Inspection'}
              className="max-w-full max-h-[60vh] object-contain shadow-md border border-[#1C1C1E]"
            />
          </div>

          {/* Details / Footer Bar */}
          <div className="flex flex-wrap items-center justify-between gap-2 p-2.5 bg-[#E5E0D5] border-2 border-[#1C1C1E] rounded text-xs font-mono">
            <div className="flex items-center gap-2 text-[#1C1C1E]">
              <Info className="w-4 h-4 text-[#0050C8]" />
              <span className="font-bold">{image.title || 'Inspection Sample'}</span>
              {image.metric && (
                <span className="mac-tag mac-tag-blue">
                  {image.metric}
                </span>
              )}
            </div>

            <div className="flex items-center gap-2 text-[11px] text-[#555]">
              <span>Original Aspect: 640 × 480 px</span>
              <button
                onClick={onClose}
                className="mac-btn mac-btn-primary text-xs py-1 px-3"
              >
                Close (ESC)
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
