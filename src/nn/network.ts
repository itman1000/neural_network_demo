import { activations, ActKey } from "../utils/activations";
import { mul, addBias, T } from "../utils/linalg";

export type Params = {
  W1: number[][];
  b1: number[];
  W2?: number[][];
  b2?: number[];
  Wout: number[][];
  bout: number[];
};

export function xavierStd(fanIn: number, fanOut: number) {
  return Math.sqrt(2 / (fanIn + fanOut));
}
export function heStd(fanIn: number) {
  return Math.sqrt(2 / Math.max(1, fanIn));
}

export function initParams(
  inputDim: number,
  h1: number,
  h2: number,
  seed: number,
  act: ActKey = "tanh",
): Params {
  function mulberry32(a: number) {
    return function () {
      let t = (a += 0x6d2b79f5) >>> 0;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randn(prng: () => number) {
    let u = 0,
      v = 0;
    while (u === 0) u = prng();
    while (v === 0) v = prng();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }
  const rnd = mulberry32(seed);
  const useHe = act === "relu" || act === "silu" || act === "gelu";
  const std1 = useHe ? heStd(inputDim) : xavierStd(inputDim, h1);
  const W1 = Array.from({ length: inputDim }, () =>
    Array.from({ length: h1 }, () => randn(rnd) * std1),
  );
  const b1 = Array.from({ length: h1 }, () => (act === "relu" ? 0.01 : 0));
  let W2: number[][] | undefined = undefined,
    b2: number[] | undefined = undefined,
    prev = h1;
  if (h2 > 0) {
    const std2 = useHe ? heStd(h1) : xavierStd(h1, h2);
    W2 = Array.from({ length: h1 }, () => Array.from({ length: h2 }, () => randn(rnd) * std2));
    b2 = Array.from({ length: h2 }, () => (act === "relu" ? 0.01 : 0));
    prev = h2;
  }
  const stdO = xavierStd(prev, 1);
  const Wout = Array.from({ length: prev }, () => [randn(rnd) * stdO]);
  const bout = [0];
  return { W1, b1, W2, b2, Wout, bout };
}

export function l2Loss(p: Params, lambda: number) {
  const sq = (a: number[][]) => a.reduce((s, r) => s + r.reduce((t, v) => t + v * v, 0), 0);
  return 0.5 * lambda * (sq(p.W1) + (p.W2 ? sq(p.W2) : 0) + sq(p.Wout));
}

export function applyDropout(A: number[][], p: number, prng: () => number, train: boolean) {
  if (!train || p <= 0) return { A, M: undefined as undefined | number[][] };
  const kp = 1 - p,
    M = A.map((r) => r.map(() => (prng() < kp ? 1 / kp : 0)));
  return { A: A.map((row, i) => row.map((v, j) => v * M[i][j])), M };
}

export function forward(
  X: number[][],
  params: Params,
  act: ActKey,
  dropout: { p1: number; p2: number },
  train: boolean,
  prng: () => number,
) {
  const { W1, b1, W2, b2, Wout, bout } = params;
  const f = activations[act].f;
  const Z1 = addBias(mul(X, W1), b1);
  let A1 = Z1.map((r) => r.map(f));
  const d1 = applyDropout(A1, dropout.p1, prng, train);
  A1 = d1.A;
  let Z2: number[][] | undefined;
  const d2 =
    W2 && b2
      ? (() => {
          const Z = addBias(mul(A1, W2), b2);
          const A = Z.map((r) => r.map(f));
          Z2 = Z;
          return applyDropout(A, dropout.p2, prng, train);
        })()
      : undefined;
  const H = d2?.A ?? A1;
  const Zo = addBias(mul(H, Wout), bout);
  const Yhat = Zo.map((row) => [1 / (1 + Math.exp(-row[0]))]);
  return { Z1, A1, Z2, A2: d2?.A, Zo, Yhat, M1: d1.M, M2: d2?.M };
}

export function accuracy(y: number[], yhat: number[][]) {
  return y.reduce((c, yi, i) => c + ((yhat[i][0] >= 0.5 ? 1 : 0) === yi ? 1 : 0), 0) / y.length;
}

export function backprop(
  X: number[][],
  y: number[],
  params: Params,
  act: ActKey,
  dropout: { p1: number; p2: number },
  lambda: number,
  prng: () => number,
  train = true,
) {
  const { W1, b1, W2, b2, Wout } = params;
  const n = X.length;
  const df = activations[act].df;
  const { Z1, A1, Z2, A2, Yhat, M1, M2 } = forward(X, params, act, dropout, train, prng);
  const dZo = Yhat.map((row, i) => [row[0] - y[i]]);
  const H = A2 ?? A1;
  const dWo = mul(T(H), dZo).map((r) => r.map((v) => v / n));
  const dbo = [dZo.reduce((s, r) => s + r[0], 0) / n];
  let dH = mul(dZo, T(Wout));
  let dW2: number[][] | undefined, db2: number[] | undefined, dZ2: number[][] | undefined;
  if (W2 && b2 && Z2 && A2) {
    dZ2 = Z2.map((row, i) => row.map((z, j) => dH[i][j] * df(z) * (M2 ? M2[i][j] : 1)));
    dW2 = mul(T(A1), dZ2).map((r) => r.map((v) => v / n));
    db2 = Array.from({ length: b2.length }, (_, j) => dZ2!.reduce((s, r) => s + r[j], 0) / n);
    dH = mul(dZ2, T(W2));
  }
  const dZ1 = Z1.map((row, i) => row.map((z, j) => dH[i][j] * df(z) * (M1 ? M1[i][j] : 1)));
  const dW1 = mul(T(X), dZ1).map((r) => r.map((v) => v / n));
  const db1 = Array.from({ length: b1.length }, (_, j) => dZ1.reduce((s, r) => s + r[j], 0) / n);
  for (let i = 0; i < params.W1.length; i++)
    for (let j = 0; j < params.W1[0].length; j++) dW1[i][j] += lambda * params.W1[i][j];
  if (params.W2 && dW2)
    for (let i = 0; i < params.W2.length; i++)
      for (let j = 0; j < params.W2[0].length; j++) dW2[i][j] += lambda * params.W2[i][j];
  for (let i = 0; i < params.Wout.length; i++) dWo[i][0] += lambda * params.Wout[i][0];
  const baseLoss = (() => {
    let s = 0;
    for (let i = 0; i < n; i++) {
      const t = Math.min(Math.max(Yhat[i][0], 1e-6), 1 - 1e-6);
      s += -(y[i] * Math.log(t) + (1 - y[i]) * Math.log(1 - t));
    }
    return s / n;
  })();
  const loss = baseLoss + l2Loss(params, lambda);
  const acc = accuracy(y, Yhat);
  return { dW1, db1, dW2, db2, dWo, dbo, loss, acc };
}

// flatten/rebuild helpers
export function flattenGrads(grads: any) {
  const vec: number[] = [];
  const push = (M: any) => {
    if (!M) return;
    for (let i = 0; i < M.length; i++) {
      if (Array.isArray(M[i])) {
        for (let j = 0; j < M[i].length; j++) vec.push(M[i][j]);
      } else {
        vec.push(M[i]);
      }
    }
  };
  push(grads.dW1);
  push(grads.db1);
  push(grads.dW2);
  push(grads.db2);
  push(grads.dWo);
  push(grads.dbo);
  return vec;
}
export function flattenParamsForShape(params: Params) {
  const vec: number[] = [];
  const push = (M: any) => {
    if (!M) return;
    for (let i = 0; i < M.length; i++) {
      if (Array.isArray(M[i])) {
        for (let j = 0; j < M[i].length; j++) vec.push(M[i][j]);
      } else {
        vec.push(M[i]);
      }
    }
  };
  push(params.W1);
  push(params.b1);
  push(params.W2);
  push(params.b2);
  push(params.Wout);
  push(params.bout);
  return vec;
}
export function rebuildVectorAsGradShapes(params: Params, fullVec: number[]) {
  let idx = 0;
  const take2d = (shape: number[][]) => shape.map((row) => row.map(() => fullVec[idx++] || 0));
  const take1d = (shape: number[]) => shape.map(() => fullVec[idx++] || 0);
  return {
    dW1: take2d(params.W1),
    db1: take1d(params.b1),
    dW2: params.W2 ? take2d(params.W2) : undefined,
    db2: params.b2 ? take1d(params.b2) : undefined,
    dWo: take2d(params.Wout),
    dbo: take1d(params.bout),
  };
}
