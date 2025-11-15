import React from "react";

export function ConfMatrix({ cm }: { cm: number[][] }) {
  const total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1];
  const cell = (v: number) => `${v} (${((v / Math.max(total, 1)) * 100).toFixed(1)}%)`;
  return (
    <div className="grid grid-cols-3 text-xs gap-1">
      <div></div>
      <div className="text-center font-semibold">Pred 0</div>
      <div className="text-center font-semibold">Pred 1</div>
      <div className="font-semibold">True 0</div>
      <div className="text-center bg-slate-100 p-1 rounded">{cell(cm[0][0])}</div>
      <div className="text-center bg-slate-100 p-1 rounded">{cell(cm[0][1])}</div>
      <div className="font-semibold">True 1</div>
      <div className="text-center bg-slate-100 p-1 rounded">{cell(cm[1][0])}</div>
      <div className="text-center bg-slate-100 p-1 rounded">{cell(cm[1][1])}</div>
    </div>
  );
}
