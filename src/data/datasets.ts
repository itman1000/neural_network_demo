import { mulberry32, randn } from "../utils/linalg";

export function makeDataset(name: string, seed: number) {
  const prng = mulberry32(seed);
  const X: number[][] = [];
  const y: number[] = [];
  if (name === "AND" || name === "OR" || name === "XOR") {
    const points = [
      [-1, -1],
      [-1, 1],
      [1, -1],
      [1, 1],
    ];
    const f = (x1: number, x2: number) =>
      name === "AND"
        ? x1 === 1 && x2 === 1
          ? 1
          : 0
        : name === "OR"
          ? x1 === 1 || x2 === 1
            ? 1
            : 0
          : x1 !== x2
            ? 1
            : 0;
    for (const [x1, x2] of points as any) {
      for (let i = 0; i < 24; i++) {
        const j1 = (prng() - 0.5) * 0.12;
        const j2 = (prng() - 0.5) * 0.12;
        X.push([x1 + j1, x2 + j2]);
        y.push(f(x1, x2));
      }
    }
    return { X, y };
  }
  if (name === "CIRCLES") {
    const n = 240;
    const noise = 0.06;
    const r1 = 0.45,
      r2 = 0.88;
    for (let i = 0; i < n; i++) {
      const t = 2 * Math.PI * prng();
      const r = r1 + noise * randn(prng);
      X.push([r * Math.cos(t), r * Math.sin(t)]);
      y.push(0);
    }
    for (let i = 0; i < n; i++) {
      const t = 2 * Math.PI * prng();
      const r = r2 + noise * randn(prng);
      X.push([r * Math.cos(t), r * Math.sin(t)]);
      y.push(1);
    }
    return { X, y };
  }
  if (name === "SPIRAL") {
    const n = 260;
    const turns = 2.2;
    const noise = 0.12;
    for (let i = 0; i < n; i++) {
      const r = i / n;
      const t = turns * 2 * Math.PI * r + noise * randn(prng);
      X.push([r * Math.cos(t), r * Math.sin(t)]);
      y.push(0);
    }
    for (let i = 0; i < n; i++) {
      const r = i / n;
      const t = turns * 2 * Math.PI * r + Math.PI + noise * randn(prng);
      X.push([r * Math.cos(t), r * Math.sin(t)]);
      y.push(1);
    }
    return { X, y };
  }
  return { X, y };
}
