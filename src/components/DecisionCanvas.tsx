import React, { useEffect, useRef } from "react";
import { forward } from "../nn/network";

export function DecisionCanvas({
  params,
  act,
  X,
  y,
}: {
  params: any;
  act: any;
  X: number[][];
  y: number[];
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const width = 420,
    height = 420;

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(width, height);
    for (let py = 0; py < height; py++) {
      for (let px = 0; px < width; px++) {
        const x1 = (px / (width - 1)) * 2 - 1;
        const x2 = (1 - py / (height - 1)) * 2 - 1;
        const { Yhat } = forward([[x1, x2]], params, act, { p1: 0, p2: 0 }, false, () =>
          Math.random(),
        );
        const p = Yhat[0][0];
        const r = Math.floor(255 * p),
          g = Math.floor(255 * (0.95 - Math.abs(p - 0.5))),
          b = Math.floor(255 * (1 - p));
        const idx = (py * width + px) * 4;
        img.data[idx] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    // plot points
    ctx.save();
    for (let i = 0; i < X.length; i++) {
      const [x1, x2] = X[i];
      const cx = ((x1 + 1) / 2) * (width - 1);
      const cy = (1 - (x2 + 1) / 2) * (height - 1);
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fillStyle = y[i] === 1 ? "#d11" : "#11d";
      ctx.fill();
    }
    ctx.restore();
  }, [params, act, X, y]);

  return (
    <canvas
      ref={ref}
      width={width}
      height={height}
      className="w-full rounded-xl border-2 border-indigo-200"
    />
  );
}
