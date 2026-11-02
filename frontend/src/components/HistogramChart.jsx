import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';

export default function HistogramChart({ data, title, height = 180 }) {
  // Transform { labels, datasets } into Recharts [{ intensity, [dataset.name]: value }]
  const chartData = useMemo(() => {
    if (!data || !data.labels || !Array.isArray(data.datasets) || data.datasets.length === 0) {
      return [];
    }

    const labels = data.labels;
    return labels.map((label, idx) => {
      const item = { intensity: label };
      data.datasets.forEach((ds) => {
        item[ds.name] = (ds.data && ds.data[idx] !== undefined) ? ds.data[idx] : 0;
      });
      return item;
    });
  }, [data]);

  if (!chartData || chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-28 mac-well rounded text-xs font-mono text-[#777]">
        No histogram data computed
      </div>
    );
  }

  const fallbackColors = ['#0050C8', '#157A3E', '#D97706', '#C81E1E', '#6B21A8'];

  return (
    <div className="bg-[#FFFFFF] mac-well rounded p-2.5 flex flex-col gap-1.5">
      {title && (
        <div className="flex items-center justify-between text-xs font-bold font-mono text-[#1C1C1E]">
          <span>{title}</span>
          <span className="text-[10px] text-[#555]">0 - 255 Bins</span>
        </div>
      )}

      <div style={{ width: '100%', height: height }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartData}
            margin={{ top: 6, right: 10, left: -20, bottom: 0 }}
          >
            <defs>
              {data.datasets.map((ds, i) => {
                const color = ds.color || fallbackColors[i % fallbackColors.length];
                const gradId = `histGrad-${ds.name.replace(/\s+/g, '')}-${i}`;
                return (
                  <linearGradient key={gradId} id={gradId} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={color} stopOpacity={0.4} />
                    <stop offset="95%" stopColor={color} stopOpacity={0.05} />
                  </linearGradient>
                );
              })}
            </defs>

            <CartesianGrid strokeDasharray="2 2" stroke="#E5E0D5" vertical={false} />

            <XAxis
              dataKey="intensity"
              stroke="#1C1C1E"
              tickLine={false}
              ticks={[0, 64, 128, 192, 255]}
              tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            />

            <YAxis
              stroke="#1C1C1E"
              tickLine={false}
              tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }}
              tickFormatter={(v) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v)}
            />

            <Tooltip
              contentStyle={{
                backgroundColor: '#F5F2EB',
                border: '2px solid #1C1C1E',
                borderRadius: '2px',
                fontSize: '11px',
                fontFamily: 'JetBrains Mono',
                color: '#1C1C1E',
                boxShadow: '3px 3px 0px #1C1C1E',
              }}
              labelFormatter={(label) => `Intensity Bin: ${label}`}
              formatter={(value, name) => [value.toLocaleString(), name]}
            />

            {data.datasets.length > 1 && (
              <Legend
                verticalAlign="top"
                align="right"
                iconType="circle"
                wrapperStyle={{
                  paddingBottom: '4px',
                  fontSize: '10px',
                  fontFamily: 'JetBrains Mono',
                  color: '#1C1C1E',
                }}
              />
            )}

            {data.datasets.map((ds, i) => {
              const color = ds.color || fallbackColors[i % fallbackColors.length];
              const gradId = `histGrad-${ds.name.replace(/\s+/g, '')}-${i}`;
              return (
                <Area
                  key={ds.name}
                  type="monotone"
                  dataKey={ds.name}
                  stroke={color}
                  strokeWidth={1.5}
                  fill={`url(#${gradId})`}
                  fillOpacity={1}
                  isAnimationActive={false}
                />
              );
            })}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
