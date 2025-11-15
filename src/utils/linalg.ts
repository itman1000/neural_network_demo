export function mul(a: number[][], b: number[][]): number[][] {
  const m = a.length,
    n = a[0].length,
    p = b[0].length;
  const out = Array.from({ length: m }, () => Array(p).fill(0));
  for (let i = 0; i < m; i++) {
    for (let k = 0; k < n; k++) {
      const aik = a[i][k];
      for (let j = 0; j < p; j++) out[i][j] += aik * b[k][j];
    }
  }
  return out;
}
export function addBias(x: number[][], b: number[]): number[][] {
  return x.map((row) => row.map((v, j) => v + b[j]));
}
export function T(x: number[][]): number[][] {
  const m = x.length,
    n = x[0].length;
  const out = Array.from({ length: n }, () => Array(m).fill(0));
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[j][i] = x[i][j];
  return out;
}

export function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5) >>> 0;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function randn(prng: () => number) {
  let u = 0,
    v = 0;
  while (u === 0) u = prng();
  while (v === 0) v = prng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
export function shuffled(n: number, rnd: () => number) {
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = (rnd() * (i + 1)) | 0;
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function projectVec(vec: number[], P: number[][]): number[] {
  const k = P.length,
    n = P[0].length;
  const out = Array(k).fill(0);
  for (let i = 0; i < k; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) {
      s += P[i][j] * vec[j];
    }
    out[i] = s;
  }
  return out;
}
export function backProjectVec(vecK: number[], P: number[][]): number[] {
  const k = P.length,
    n = P[0].length;
  const out = Array(n).fill(0);
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let i = 0; i < k; i++) {
      s += P[i][j] * vecK[i];
    }
    out[j] = s;
  }
  return out;
}
export function vecDot(a: number[], b: number[]) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
export function vecNorm(a: number[]) {
  return Math.sqrt(vecDot(a, a)) + 1e-12;
}
export function vecAdd(a: number[], b: number[], alpha = 1) {
  const out = a.slice();
  for (let i = 0; i < a.length; i++) out[i] += alpha * b[i];
  return out;
}
export function vecScale(a: number[], c: number) {
  return a.map((v) => v * c);
}

export function gramSchmidtOrthonormalize(cols: number[][]) {
  const Q: number[][] = [];
  const eps = 1e-8;
  for (const v of cols) {
    const u = v.slice();
    for (const q of Q) {
      const dot = vecDot(u, q);
      for (let i = 0; i < u.length; i++) u[i] -= dot * q[i];
    }
    const n = vecNorm(u);
    if (n > eps) {
      Q.push(u.map((x) => x / n));
    }
  }
  return Q;
}

export function eigenDecomposeSmall(C: number[][], top = 3, powerIter = 10) {
  const n = C.length;
  const eigVecs: number[][] = [];
  const eigVals: number[] = [];
  const M = C.map((r) => r.slice());
  for (let comp = 0; comp < Math.min(n, top); comp++) {
    let v = Array.from({ length: n }, () => Math.random() - 0.5);
    const vn = vecNorm(v);
    v = v.map((x) => x / (vn || 1));
    for (let it = 0; it < powerIter; it++) {
      const w = Array(n).fill(0);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += M[i][j] * v[j];
        w[i] = s;
      }
      const wn = vecNorm(w);
      v = w.map((x) => x / (wn || 1));
    }
    let lambda = 0;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += M[i][j] * v[j];
      lambda += v[i] * s;
    }
    eigVecs.push(v.slice());
    eigVals.push(Math.abs(lambda));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i][j] -= lambda * v[i] * v[j];
  }
  return { eigVecs, eigVals };
}
