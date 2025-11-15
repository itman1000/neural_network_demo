import React from "react";

export function LineChart({
  series,
  height = 120,
  maxPoints = 300,
  yLabel,
}: {
  series: { name: string; data: number[] }[];
  height?: number;
  maxPoints?: number;
  yLabel: string;
}) {
  const width = 360;
  const pad = 24;
  const innerW = width - pad * 2;
  const innerH = height - pad * 2;
  const maxLen = Math.max(1, ...series.map((s) => s.data.length));
  const start = Math.max(0, maxLen - maxPoints);
  const xs = (i: number) =>
    pad + (innerW * (i - start)) / Math.max(1, Math.min(maxPoints, maxLen - 1));
  const view = series.map((s) => s.data.slice(start));
  const flat = view.flat();
  const minY = Math.min(...flat, 0);
  const maxY = Math.max(...flat, 1e-6);
  const ys = (v: number) => pad + innerH - (innerH * (v - minY)) / (maxY - minY || 1);
  return (
    <svg width={width} height={height} className="w-full">
      <rect x={0} y={0} width={width} height={height} fill="#fff" rx={12} />
      <text x={8} y={16} fontSize={10} fill="#475569">
        {yLabel}
      </text>
      {series.map((s, idx) => {
        const path = s.data
          .slice(start)
          .map((v, i) => `${i === 0 ? "M" : "L"} ${xs(i + start)},${ys(v)}`)
          .join(" ");
        return (
          <path
            key={s.name + idx}
            d={path}
            fill="none"
            stroke={idx % 2 ? "#16a34a" : "#2563eb"}
            strokeWidth={2}
          />
        );
      })}
    </svg>
  );
}
