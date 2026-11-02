import React from 'react';
import { Camera, ShieldCheck, Zap, Layers, Monitor, HardDrive, Cpu } from 'lucide-react';

export default function Header({
  recognizedPlate,
  plateType,
  confidence,
  processingTime,
  activeTab,
  onSelectTab,
}) {
  const tabs = [
    { id: 'surveillance', label: 'View 1: Live Surveillance & Toll', icon: ShieldCheck },
    { id: 'workbench', label: 'View 2: IVP Syllabus Workbench', icon: Layers },
    { id: 'motion_benchmark', label: 'View 3: Video Motion & Batch Benchmark', icon: Zap },
  ];

  // Helper for HSRP plate styling
  const getHsrpStyle = (type) => {
    const t = (type || '').toLowerCase();
    if (t.includes('yellow') || t.includes('commercial')) {
      return {
        plateClass: 'hsrp-yellow-commercial',
        typeLabel: 'Commercial Vehicle (Yellow)',
        badgeClass: 'mac-tag-amber',
      };
    }
    if (t.includes('green') || t.includes('ev')) {
      return {
        plateClass: 'hsrp-green-ev',
        typeLabel: 'Eco EV Pass (Green)',
        badgeClass: 'mac-tag-green',
      };
    }
    return {
      plateClass: 'hsrp-white-private',
      typeLabel: 'Private Vehicle (White)',
      badgeClass: 'mac-tag-gray',
    };
  };

  const hsrp = getHsrpStyle(plateType);
  const formattedPlate = recognizedPlate && recognizedPlate.trim()
    ? recognizedPlate
        .toUpperCase()
        .replace(/\s+/g, '')
        .replace(/^([A-Z]{2})(\d{2})([A-Z]{1,2})(\d{4})$/, '$1 $2 $3 $4')
    : 'UNREADABLE';

  return (
    <header className="sticky top-0 z-50 bg-[#F5F2EB] border-b-2 border-[#1C1C1E] shadow-[0_2px_0px_#1C1C1E]">
      {/* Classic 6-Color Apple Stripe Ribbon */}
      <div className="w-full h-1 apple-six-color-h" />

      {/* Main Bar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-wrap items-center justify-between min-h-[70px] py-2 gap-4">
          
          {/* Left: Classic Mac System Logo & Title */}
          <div className="flex items-center gap-3">
            {/* Retro Mac Finder / Hardware Icon */}
            <div className="relative flex items-center justify-center w-11 h-11 bg-[#E5E0D5] border-2 border-[#1C1C1E] rounded shadow-[2px_2px_0px_#1C1C1E] flex-shrink-0">
              <Monitor className="w-6 h-6 text-[#1C1C1E]" />
              <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-[#157A3E] border-2 border-[#1C1C1E] flex items-center justify-center">
                <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
              </div>
            </div>

            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold tracking-tight text-[#1C1C1E] flex items-center gap-2 font-mono uppercase">
                  SentinelPlate ALPR
                  <span className="text-[10px] font-bold px-1.5 py-0.5 bg-[#1C1C1E] text-white rounded shadow-sm">
                    Mac OS System 7.5
                  </span>
                </h1>
              </div>
              <p className="text-[11px] text-[#555] font-sans font-medium hidden sm:block">
                Digital Image & Video Processing (Units 1–8) • Automatic License Plate Recognition
              </p>
            </div>
          </div>

          {/* Center: 3-View Macintosh Tactile Tab Navigation */}
          <nav className="flex items-center gap-2 bg-[#E5E0D5] p-1.5 border-2 border-[#1C1C1E] rounded shadow-[2px_2px_0px_#1C1C1E]">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => onSelectTab(tab.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold transition-all rounded-[2px] ${
                    isActive
                      ? 'bg-[#1C1C1E] text-white border-2 border-[#1C1C1E] shadow-[1px_1px_0px_rgba(0,0,0,0.5)]'
                      : 'bg-[#F5F2EB] text-[#1C1C1E] border-2 border-[#1C1C1E] shadow-[1px_1px_0px_#1C1C1E] hover:bg-[#ECE7DD]'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Right: Embossed Indian High Security Registration Plate Display */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2.5 bg-[#E5E0D5] p-1.5 pl-2.5 rounded border-2 border-[#1C1C1E] shadow-[2px_2px_0px_#1C1C1E]">
              <div className="flex flex-col text-right">
                <span className="text-[9px] text-[#555] font-bold uppercase tracking-wider font-mono">
                  Recognized HSRP
                </span>
                {confidence !== undefined && (
                  <span className="text-[10px] text-[#157A3E] font-bold font-mono">
                    {Math.round(confidence > 1 ? confidence : confidence * 100)}% Match
                  </span>
                )}
              </div>

              {/* Physical Embossed HSRP Plate */}
              <div
                className={`hsrp-plate h-9 px-0 ${hsrp.plateClass}`}
                title={`Recognized Indian License Plate: ${formattedPlate} (${hsrp.typeLabel})`}
              >
                {/* Left Blue IND Strip with Chakra */}
                <div className="hsrp-ind-strip">
                  <span className="w-2 h-2 rounded-full border border-white flex items-center justify-center text-[5px] leading-none mb-0.5 font-sans">
                    ❂
                  </span>
                  <span>IND</span>
                </div>

                {/* Plate Numbers */}
                <div className="px-2.5 font-mono font-black text-xs sm:text-sm tracking-widest whitespace-nowrap">
                  {formattedPlate}
                </div>
              </div>
            </div>

            {/* Latency Pill */}
            {processingTime !== undefined && processingTime !== null && (
              <div
                className="hidden lg:flex items-center gap-1 px-2 py-1 bg-[#E5E0D5] border-2 border-[#1C1C1E] rounded shadow-[1px_1px_0px_#1C1C1E] text-xs font-mono text-[#1C1C1E]"
                title="Total pipeline processing latency"
              >
                <Cpu className="w-3 h-3 text-[#0050C8]" />
                <span className="font-bold">{processingTime}ms</span>
              </div>
            )}
          </div>

        </div>
      </div>
    </header>
  );
}
