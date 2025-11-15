import {
  Params,
  flattenGrads,
  rebuildVectorAsGradShapes,
  flattenParamsForShape,
} from "../nn/network";
import { projectVec, backProjectVec, eigenDecomposeSmall, vecNorm } from "../utils/linalg";

// ApoLLO-like: directional confidence scaled Adam-ish update
export function stepApoLLO(
  params: Params,
  grads: any,
  lr: number,
  state: any,
  cfg: { beta2?: number; eps?: number; beta3?: number; gamma?: number },
) {
  const beta2 = cfg.beta2 ?? 0.999,
    eps = cfg.eps ?? 1e-8,
    beta1 = 0.9,
    beta3 = cfg.beta3 ?? 0.9,
    gamma = cfg.gamma ?? 1.0;
  if (!state.vW1) {
    state.vW1 = params.W1.map((r) => r.map(() => 0));
    state.vb1 = Array(params.b1.length).fill(0);
    state.vWo = params.Wout.map((r) => r.map(() => 0));
    state.vbo = Array(params.bout.length).fill(0);
  }
  if (!state.mW1) {
    state.mW1 = params.W1.map((r) => r.map(() => 0));
    state.mb1 = Array(params.b1.length).fill(0);
    state.mWo = params.Wout.map((r) => r.map(() => 0));
    state.mbo = Array(params.bout.length).fill(0);
    if (params.W2) {
      state.mW2 = params.W2.map((r) => r.map(() => 0));
      state.mb2 = Array(params.b2!.length).fill(0);
      state.vW2 = params.W2.map((r) => r.map(() => 0));
      state.vb2 = Array(params.b2!.length).fill(0);
    }
    state.t = 0;
  }
  if (!state.dW1) {
    state.dW1 = params.W1.map((r) => r.map(() => 0));
    state.db1 = Array(params.b1.length).fill(0);
    state.dWo = params.Wout.map((r) => r.map(() => 0));
    state.dbo = Array(params.bout.length).fill(0);
    if (params.W2) {
      state.dW2 = params.W2.map((r) => r.map(() => 0));
      state.db2 = Array(params.b2!.length).fill(0);
    }
  }
  state.t = (state.t ?? 0) + 1;
  const t = state.t;

  const g = flattenGrads(grads);
  const dn = Math.sqrt(g.reduce((s, x) => s + x * x, 0)) + 1e-12;
  const dvec = flattenGrads({
    dW1: state.dW1,
    db1: state.db1,
    dW2: state.dW2,
    db2: state.db2,
    dWo: state.dWo,
    dbo: state.dbo,
  });
  const ddn = Math.sqrt(dvec.reduce((s, x) => s + x * x, 0)) + 1e-12;
  const dot = g.reduce((s, x, i) => s + (x / dn) * (dvec[i] / ddn), 0);
  const conf = (1 + Math.max(-1, Math.min(1, dot))) / 2;
  const scaleFactor = Math.pow(conf, gamma);

  const update2d = (W: number[][], g: number[][], m: number[][], v: number[][], d: number[][]) => {
    for (let i = 0; i < W.length; i++)
      for (let j = 0; j < W[0].length; j++) {
        const gij = g[i][j];
        m[i][j] = 0.9 * m[i][j] + 0.1 * gij;
        v[i][j] = beta2 * v[i][j] + (1 - beta2) * gij * gij;
        const mhat = m[i][j] / (1 - Math.pow(0.9, t));
        const vhat = v[i][j] / (1 - Math.pow(beta2, t));
        W[i][j] -= (lr * scaleFactor * mhat) / (Math.sqrt(vhat) + eps);
        d[i][j] = beta3 * d[i][j] + (1 - beta3) * (gij / (dn || 1));
      }
  };
  const update1d = (b: number[], g: number[], m: number[], v: number[], d: number[]) => {
    for (let j = 0; j < b.length; j++) {
      const gj = g[j];
      m[j] = 0.9 * m[j] + 0.1 * gj;
      v[j] = beta2 * v[j] + (1 - beta2) * gj * gj;
      const mhat = m[j] / (1 - Math.pow(0.9, t));
      const vhat = v[j] / (1 - Math.pow(beta2, t));
      b[j] -= (lr * scaleFactor * mhat) / (Math.sqrt(vhat) + eps);
      d[j] = beta3 * d[j] + (1 - beta3) * (gj / (dn || 1));
    }
  };
  update2d(params.W1, grads.dW1, state.mW1, state.vW1, state.dW1);
  update1d(params.b1, grads.db1, state.mb1, state.vb1, state.db1);
  if (params.W2 && grads.dW2) {
    update2d(params.W2, grads.dW2, state.mW2, state.vW2, state.dW2);
    update1d(params.b2!, grads.db2!, state.mb2, state.vb2, state.db2);
  }
  update2d(params.Wout, grads.dWo, state.mWo, state.vWo, state.dWo);
  update1d(params.bout, grads.dbo, state.mbo, state.vbo, state.dbo);
}

// SAGE-like
export function initSAGEState(params: Params, k = 20, seed = 12345) {
  const dim = flattenParamsForShape(params).length;
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
  const rng = mulberry32(seed);
  const P = Array.from({ length: k }, () =>
    Array.from({ length: dim }, () => randn(rng) / Math.sqrt(k)),
  );
  return {
    mW1: params.W1.map((r) => r.map(() => 0)),
    mb1: Array(params.b1.length).fill(0),
    mW2: params.W2 ? params.W2.map((r) => r.map(() => 0)) : undefined,
    mb2: params.b2 ? Array(params.b2.length).fill(0) : undefined,
    mWo: params.Wout.map((r) => r.map(() => 0)),
    mbo: Array(params.bout.length).fill(0),
    vW1: params.W1.map((r) => r.map(() => 0)),
    vb1: Array(params.b1.length).fill(0),
    vW2: params.W2 ? params.W2.map((r) => r.map(() => 0)) : undefined,
    vb2: params.b2 ? Array(params.b2.length).fill(0) : undefined,
    vWo: params.Wout.map((r) => r.map(() => 0)),
    vbo: Array(params.bout.length).fill(0),
    P,
    C: Array.from({ length: k }, () => Array(k).fill(0)),
    eigVecs: Array.from({ length: Math.min(3, k) }, (_, i) => {
      const v = Array(k).fill(0);
      v[i] = 1;
      return v;
    }),
    eigVals: Array(Math.min(3, k)).fill(0.1),
    gradNorm: [] as number[],
    spectralGap: [] as number[],
    t: 0,
    dim,
    k,
  };
}

export function stepSAGE(
  params: Params,
  grads: any,
  baseLr: number,
  state: any,
  updateInterval = 5,
) {
  state.t += 1;
  const t = state.t;
  const beta1 = 0.9,
    beta2 = 0.999,
    eps = 1e-8;
  const g = flattenGrads(grads);
  const gProj = projectVec(g, state.P);
  const decayC = 0.99;
  for (let i = 0; i < state.k; i++)
    for (let j = 0; j < state.k; j++)
      state.C[i][j] = decayC * state.C[i][j] + (1 - decayC) * gProj[i] * gProj[j];
  if (t % updateInterval === 0) {
    const { eigVecs, eigVals } = eigenDecomposeSmall(state.C, Math.min(3, state.k), 10);
    state.eigVecs = eigVecs;
    state.eigVals = eigVals;
    if (eigVals.length >= 2) {
      state.spectralGap.push(eigVals[0] / (eigVals[eigVals.length - 1] + 1e-10));
    }
  }
  const maxEig = Math.max(...state.eigVals, 1e-6);
  const gEigen = state.eigVecs.map((v: number[]) =>
    v.reduce((sum, vi, i) => sum + vi * gProj[i], 0),
  );
  const adaptiveScales = state.eigVals.map((lambda: number) => Math.sqrt(maxEig / (lambda + eps)));
  const gCorrected = Array(state.k).fill(0);
  for (let i = 0; i < state.eigVecs.length; i++)
    for (let j = 0; j < state.k; j++)
      gCorrected[j] += state.eigVecs[i][j] * gEigen[i] * adaptiveScales[i];
  const gFinal = backProjectVec(gCorrected, state.P);
  const gradsCorrected = rebuildVectorAsGradShapes(params, gFinal);
  const updateParam = (param: any, grad: any, m: any, v: any) => {
    for (let i = 0; i < param.length; i++) {
      if (Array.isArray(param[i])) {
        for (let j = 0; j < param[i].length; j++) {
          m[i][j] = 0.9 * m[i][j] + 0.1 * grad[i][j];
          v[i][j] = beta2 * v[i][j] + (1 - beta2) * grad[i][j] * grad[i][j];
          const mhat = m[i][j] / (1 - Math.pow(beta1, t));
          const vhat = v[i][j] / (1 - Math.pow(beta2, t));
          param[i][j] -= (baseLr * mhat) / (Math.sqrt(vhat) + eps);
        }
      } else {
        m[i] = 0.9 * m[i] + 0.1 * grad[i];
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
        const mhat = m[i] / (1 - Math.pow(beta1, t));
        const vhat = v[i] / (1 - Math.pow(beta2, t));
        param[i] -= (baseLr * mhat) / (Math.sqrt(vhat) + eps);
      }
    }
  };
  updateParam(params.W1, gradsCorrected.dW1, state.mW1, state.vW1);
  updateParam(params.b1, gradsCorrected.db1, state.mb1, state.vb1);
  if (params.W2 && gradsCorrected.dW2) {
    updateParam(params.W2, gradsCorrected.dW2, state.mW2, state.vW2);
    updateParam(params.b2!, gradsCorrected.db2!, state.mb2, state.vb2);
  }
  updateParam(params.Wout, gradsCorrected.dWo, state.mWo, state.vWo);
  updateParam(params.bout, gradsCorrected.dbo, state.mbo, state.vbo);
  const gradNorm = Math.sqrt(g.reduce((s: number, gi: number) => s + gi * gi, 0));
  state.gradNorm.push(gradNorm);
}
