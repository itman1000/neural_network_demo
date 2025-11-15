import {
  Params,
  flattenGrads,
  rebuildVectorAsGradShapes,
  flattenParamsForShape,
} from "../nn/network";
import {
  projectVec,
  backProjectVec,
  eigenDecomposeSmall,
  gramSchmidtOrthonormalize,
  vecNorm,
  vecDot,
} from "../utils/linalg";

export function initFusionState(params: Params, k = 20, seed = 12345, opal_r = 4) {
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
  const dim = flattenParamsForShape(params).length;
  const rng = mulberry32(seed);
  const P = Array.from({ length: k }, () =>
    Array.from({ length: dim }, () => randn(rng) / Math.sqrt(k)),
  );

  const st: any = {
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
    eigVals: Array(Math.min(3, k)).fill(1e-2),
    gbuf: [] as number[][],
    opal_r,
    i_k: Array(k).fill(0),
    prev_k: Array(k).fill(0),
    d_k: Array(k).fill(0),
    dirW1: Array(params.W1[0].length)
      .fill(0)
      .map(() => Array(params.W1.length).fill(0)),
    dirW2: params.W2
      ? Array(params.W2[0].length)
          .fill(0)
          .map(() => Array(params.W2!.length).fill(0))
      : undefined,
    dirWo: Array(params.Wout.length)
      .fill(0)
      .map(() => 0),
    t: 0,
  };
  return st;
}

function clipVector(vec: number[], maxNorm?: number) {
  if (!maxNorm || maxNorm <= 0) return vec;
  const n = vecNorm(vec);
  if (n <= maxNorm) return vec;
  const s = maxNorm / (n + 1e-12);
  return vec.map((v) => v * s);
}

export function stepFusion(params: Params, grads: any, baseLr: number, state: any, cfg: any) {
  state.t += 1;
  const t = state.t;
  const beta1 = cfg.beta1 ?? 0.9,
    beta2 = cfg.beta2 ?? 0.999,
    eps = cfg.eps ?? 1e-8;
  const wd = cfg.wd ?? 0.0;
  const clip = cfg.clip ?? 0;
  const enableSAGE = cfg.enableSAGE ?? true;
  const enableOPAL = cfg.enableOPAL ?? true;
  const enableDEO = cfg.enableDEO ?? true;
  const enableApollo = cfg.enableApollo ?? true;
  const enableSGDS = cfg.enableSGDS ?? true;

  let g = flattenGrads(grads);
  if (clip > 0) {
    g = clipVector(g, clip);
  }

  const gk = projectVec(g, state.P);
  const decayC = cfg.sageDecay ?? 0.99;
  for (let i = 0; i < state.C.length; i++)
    for (let j = 0; j < state.C[0].length; j++)
      state.C[i][j] = decayC * state.C[i][j] + (1 - decayC) * gk[i] * gk[j];
  if (t % (cfg.updateInterval ?? 5) === 0) {
    const { eigVecs, eigVals } = eigenDecomposeSmall(state.C, Math.min(3, state.C.length), 10);
    state.eigVecs = eigVecs;
    state.eigVals = eigVals;
  }
  let gk_sage: number[];
  if (enableSAGE) {
    const maxEig = Math.max(...state.eigVals, 1e-6);
    gk_sage = Array(state.C.length).fill(0);
    for (let i = 0; i < state.eigVecs.length; i++) {
      const v = state.eigVecs[i];
      let coef = 0;
      for (let j = 0; j < v.length; j++) coef += v[j] * gk[j];
      const scale = Math.sqrt(maxEig / (state.eigVals[i] + 1e-8));
      for (let j = 0; j < v.length; j++) gk_sage[j] += v[j] * (coef * scale);
    }
    const approx = Array(gk.length).fill(0);
    for (let i = 0; i < state.eigVecs.length; i++) {
      const v = state.eigVecs[i];
      let coef = 0;
      for (let j = 0; j < v.length; j++) coef += v[j] * gk[j];
      for (let j = 0; j < v.length; j++) approx[j] += v[j] * coef;
    }
    const residual = gk.map((x: number, i: number) => x - approx[i]);
    for (let j = 0; j < gk.length; j++) gk_sage[j] += residual[j];
  } else {
    gk_sage = gk.slice();
  }

  if (enableOPAL && state.gbuf.length >= 1) {
    const rcols = gramSchmidtOrthonormalize(state.gbuf);
    const parallel = Array(gk_sage.length).fill(0);
    for (const q of rcols) {
      const c = vecDot(q, gk_sage);
      for (let i = 0; i < parallel.length; i++) parallel[i] += q[i] * c;
    }
    const resid = gk_sage.map((x: number, i: number) => x - parallel[i]);
    const aPara = cfg.opalAlphaPara ?? 1.0;
    const aOrth = cfg.opalAlphaOrth ?? 1.0;
    gk_sage = gk_sage.map((_: number, i: number) => aPara * parallel[i] + aOrth * resid[i]);
  }
  if (enableOPAL) {
    state.gbuf.push(gk.slice());
    if (state.gbuf.length > (state.opal_r ?? 4)) state.gbuf.shift();
  } else if (state.gbuf.length) {
    state.gbuf = [];
  }

  let deok: number[];
  if (enableDEO) {
    const kp = cfg.kp ?? 1.0;
    const ki = cfg.ki ?? 0.3;
    const kd = cfg.kd ?? 0.1;
    const iDecay = cfg.iDecay ?? 0.95;
    for (let i = 0; i < state.i_k.length; i++) state.i_k[i] = iDecay * state.i_k[i] + gk_sage[i];
    const diff = gk_sage.map((x: number, i: number) => x - state.prev_k[i]);
    deok = gk_sage.map((x: number, i: number) => kp * x + ki * state.i_k[i] + kd * diff[i]);
    state.prev_k = gk_sage.slice();
  } else {
    deok = gk_sage.slice();
    state.prev_k = gk_sage.slice();
    if (state.i_k.some((v: number) => v !== 0)) state.i_k.fill(0);
  }

  let scaleFactor = 1;
  if (enableApollo) {
    const dn = vecNorm(state.d_k);
    const gn = vecNorm(deok);
    let conf = 0.5;
    if (dn > 0 && gn > 0) {
      const dot = deok.reduce(
        (s: number, val: number, i: number) => s + (state.d_k[i] / dn) * (val / gn),
        0,
      );
      conf = (1 + Math.max(-1, Math.min(1, dot))) / 2;
    }
    scaleFactor = Math.pow(conf, cfg.gamma ?? 1.0);
    const beta3 = cfg.beta3 ?? 0.9;
    if (gn > 0) {
      const unit = deok.map((x: number) => x / gn);
      for (let i = 0; i < state.d_k.length; i++)
        state.d_k[i] = beta3 * state.d_k[i] + (1 - beta3) * unit[i];
    }
  } else if (state.d_k.some((v: number) => v !== 0)) {
    state.d_k.fill(0);
  }

  const gFinal = backProjectVec(deok, state.P).map((x) => x * scaleFactor);
  const gradsCorrected = rebuildVectorAsGradShapes(params, gFinal);

  const decoupledWeightDecay = (arr: any) => {
    for (let i = 0; i < arr.length; i++) {
      if (Array.isArray(arr[i])) {
        for (let j = 0; j < arr[i].length; j++) {
          arr[i][j] -= baseLr * wd * arr[i][j];
        }
      } else {
        arr[i] -= baseLr * wd * arr[i];
      }
    }
  };

  const update2dGrouped = (
    W: number[][],
    gW: number[][],
    mW: number[][],
    vW: number[][],
    dirCols: number[][],
    useConsensus: boolean,
  ) => {
    const rows = W.length,
      cols = W[0].length;
    for (let j = 0; j < cols; j++) {
      const mcol = Array(rows).fill(0);
      for (let i = 0; i < rows; i++) {
        mW[i][j] = beta1 * mW[i][j] + (1 - beta1) * gW[i][j];
        vW[i][j] = beta2 * vW[i][j] + (1 - beta2) * gW[i][j] * gW[i][j];
        mcol[i] = mW[i][j];
      }
      let Sg = 1;
      if (useConsensus) {
        const prev = dirCols[j];
        const mcN = vecNorm(mcol);
        const pvN = vecNorm(prev);
        let cos = 0.0;
        if (mcN > 0 && pvN > 0) {
          let dot = 0;
          for (let i = 0; i < rows; i++) dot += (mcol[i] / mcN) * (prev[i] / pvN);
          cos = Math.max(-1, Math.min(1, dot));
        }
        Sg = (1 + cos) / 2;
        const betaG = 0.9;
        if (mcN > 0) {
          for (let i = 0; i < rows; i++) {
            const u = mcol[i] / mcN;
            dirCols[j][i] = betaG * dirCols[j][i] + (1 - betaG) * u;
          }
        }
      }
      const b1t = 1 - Math.pow(beta1, t);
      const b2t = 1 - Math.pow(beta2, t);
      for (let i = 0; i < rows; i++) {
        const mhat = mW[i][j] / (b1t || 1e-8);
        const vhat = vW[i][j] / (b2t || 1e-8);
        W[i][j] -= baseLr * Sg * (mhat / (Math.sqrt(vhat) + eps));
      }
    }
  };
  const update1d = (b: number[], gb: number[], mb: number[], vb: number[]) => {
    for (let j = 0; j < b.length; j++) {
      mb[j] = beta1 * mb[j] + (1 - beta1) * gb[j];
      vb[j] = beta2 * vb[j] + (1 - beta2) * gb[j] * gb[j];
      const mhat = mb[j] / (1 - Math.pow(beta1, t));
      const vhat = vb[j] / (1 - Math.pow(beta2, t));
      b[j] -= baseLr * (mhat / (Math.sqrt(vhat) + eps));
    }
  };

  update2dGrouped(params.W1, gradsCorrected.dW1, state.mW1, state.vW1, state.dirW1, enableSGDS);
  update1d(params.b1, gradsCorrected.db1, state.mb1, state.vb1);
  if (params.W2 && gradsCorrected.dW2 && state.mW2 && state.vW2 && state.dirW2) {
    update2dGrouped(params.W2, gradsCorrected.dW2, state.mW2, state.vW2, state.dirW2, enableSGDS);
    update1d(params.b2!, gradsCorrected.db2!, state.mb2, state.vb2);
  }
  // For output layer, reuse a simple fresh dir array (small layer)
  const dirOut = Array(params.Wout[0].length)
    .fill(0)
    .map(() => Array(params.Wout.length).fill(0));
  update2dGrouped(params.Wout, gradsCorrected.dWo, state.mWo, state.vWo, dirOut, enableSGDS);
  update1d(params.bout, gradsCorrected.dbo, state.mbo, state.vbo);

  if (wd > 0) {
    decoupledWeightDecay(params.W1);
    if (params.W2) decoupledWeightDecay(params.W2);
    decoupledWeightDecay(params.Wout);
  }
}
