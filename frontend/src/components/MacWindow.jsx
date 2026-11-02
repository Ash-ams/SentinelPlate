import React from 'react';

/**
 * Authentic Vintage Apple Macintosh (System 7 / Platinum) Window Component.
 * Features classic pinstripe titlebar, left square close box, 2px ink-black border,
 * and hard offset retro drop shadow.
 */
export default function MacWindow({
  title,
  children,
  className = '',
  headerRight = null,
  onClose = null,
  badge = null,
  hasPinstripes = true,
}) {
  return (
    <div className={`mac-window flex flex-col ${className}`}>
      {/* Macintosh System 7 Titlebar */}
      <div className="mac-titlebar">
        {/* Square Close Box */}
        <button
          type="button"
          onClick={onClose}
          aria-label="Close window"
          className="mac-close-box"
          title="Classic Mac OS Close Box"
        >
          <span className="w-1.5 h-1.5 bg-[#1C1C1E] opacity-0 hover:opacity-100 transition-opacity" />
        </button>

        {/* Left Pinstripes */}
        {hasPinstripes && <div className="mac-pinstripes" />}

        {/* Centered Bold Window Title */}
        <div className="mac-title-text flex items-center gap-1.5">
          <span>{title}</span>
          {badge && (
            <span className="text-[10px] px-1.5 py-0.2 bg-[#1C1C1E] text-white rounded-[2px] font-mono">
              {badge}
            </span>
          )}
        </div>

        {/* Right Pinstripes */}
        {hasPinstripes && <div className="mac-pinstripes" />}

        {/* Optional Right Action Controls */}
        {headerRight && (
          <div className="ml-2 flex items-center gap-1.5 flex-shrink-0">
            {headerRight}
          </div>
        )}
      </div>

      {/* Window Body */}
      <div className="p-4 flex-1 flex flex-col gap-4 bg-[#F5F2EB]">
        {children}
      </div>
    </div>
  );
}
