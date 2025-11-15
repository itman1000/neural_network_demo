import React, { useCallback, useEffect, useMemo, useState } from "react";
import { makeDataset } from "./data/datasets";
import { mulberry32, shuffled } from "./utils/linalg";
import { initParams, backprop, forward } from "./nn/network";
import { LineChart } from "./components/Charts";
import { ConfMatrix } from "./components/ConfusionMatrix";
import { DecisionCanvas } from "./components/DecisionCanvas";
import { stepApoLLO, initSAGEState, stepSAGE } from "./optimizers/apollo_sage";
import { initFusionState, stepFusion } from "./optimizers/fusion";

export default function App() {
  const [dataset, setDataset] = useState("SPIRAL");
  const [dataSeed, setDataSeed] = useState(42);
  const [splitSeed, setSplitSeed] = useState(12345);
  const [initSeed, setInitSeed] = useState(7);
  const [shuffleKey, setShuffleKey] = useState(555);
  const [valRatio, setValRatio] = useState(0.2);
  const dataAll = useMemo(() => makeDataset(dataset, dataSeed), [dataset, dataSeed]);

  const prng = useMemo(() => mulberry32(splitSeed), [splitSeed]);
  const idx = useMemo(() => shuffled(dataAll.X.length, prng), [dataAll, prng]);
  const split = Math.floor(dataAll.X.length * (1 - valRatio));
  const trainIdx = idx.slice(0, split);
  const valIdx = idx.slice(split);
  const Xtrain = trainIdx.map((i) => dataAll.X[i]);
  const ytrain = trainIdx.map((i) => dataAll.y[i]);
  const Xval = valIdx.map((i) => dataAll.X[i]);
  const yval = valIdx.map((i) => dataAll.y[i]);

  const [h1, setH1] = useState(10);
  const [h2, setH2] = useState(8);
  const [act, setAct] = useState("tanh");
  const [drop1, setDrop1] = useState(0.1);
  const [drop2, setDrop2] = useState(0.1);
  const [lambda, setLambda] = useState(0.001);

  const [opt, setOpt] = useState("fusion");
  const [lr, setLr] = useState(0.12);
  const [momentum, setMomentum] = useState(0.9);

  const [k, setK] = useState(20);
  const [wd, setWd] = useState(0.0);
  const [opalR, setOpalR] = useState(4);
  const [opalAlphaPara, setOpalAlphaPara] = useState(1.0);
  const [opalAlphaOrth, setOpalAlphaOrth] = useState(1.0);
  const [kp, setKp] = useState(1.0);
  const [ki, setKi] = useState(0.3);
  const [kd, setKd] = useState(0.1);
  const [iDecay, setIDecay] = useState(0.95);
  const [beta3, setBeta3] = useState(0.9);
  const [gamma, setGamma] = useState(1.0);
  const [sageDecay, setSageDecay] = useState(0.99);
  const [updateInterval, setUpdateInterval] = useState(5);
  const [clip, setClip] = useState(3.0);
  const [batch, setBatch] = useState(32);
  const [patience, setPatience] = useState(20);
  const [enableOpal, setEnableOpal] = useState(true);
  const [enableDeo, setEnableDeo] = useState(true);
  const [enableApollo, setEnableApollo] = useState(true);
  const [enableSage, setEnableSage] = useState(true);
  const [enableSgds, setEnableSgds] = useState(true);

  const [params, setParams] = useState(() => initParams(2, h1, h2, initSeed, act as any));
  const [fusionState, setFusionState] = useState(() =>
    initFusionState(initParams(2, h1, h2, initSeed, act as any), k, 1337, opalR),
  );
  const [apolloState, setApolloState] = useState<any>({});
  const [sageState, setSageState] = useState(() => initSAGEState(params, k));

  const [epoch, setEpoch] = useState(0);
  const [trainLoss, setTrainLoss] = useState(0);
  const [trainAcc, setTrainAcc] = useState(0);
  const [valLoss, setValLoss] = useState(0);
  const [valAcc, setValAcc] = useState(0);
  const [histLoss, setHistLoss] = useState<number[]>([]);
  const [histValLoss, setHistValLoss] = useState<number[]>([]);
  const [histAcc, setHistAcc] = useState<number[]>([]);
  const [histValAcc, setHistValAcc] = useState<number[]>([]);
  const [bestVal, setBestVal] = useState(Infinity);
  const [wait, setWait] = useState(0);
  const [auto, setAuto] = useState(false);

  const resetTrainingState = useCallback(() => {
    const p = initParams(2, h1, h2, initSeed, act as any);
    setParams(p);
    setFusionState(initFusionState(p, k, 1337, opalR));
    setApolloState({});
    setSageState(initSAGEState(p, k));
    setEpoch(0);
    setHistLoss([]);
    setHistValLoss([]);
    setHistAcc([]);
    setHistValAcc([]);
    setBestVal(Infinity);
    setWait(0);
    setTrainLoss(0);
    setValLoss(0);
    setTrainAcc(0);
    setValAcc(0);
    setAuto(false);
  }, [h1, h2, initSeed, act, k, opalR]);

  useEffect(() => {
    resetTrainingState();
  }, [resetTrainingState]);

  useEffect(() => {
    if (opt === "fusion") {
      setFusionState(initFusionState(params, k, 1337, opalR));
    } else if (opt === "apollo") {
      setApolloState({});
    } else if (opt === "sage") {
      setSageState(initSAGEState(params, k));
    }
  }, [opt]);

  function trainNEpochs(n: number, useEarlyStop = false) {
    const deepCopyParams = (p: any) => ({
      W1: p.W1.map((r: number[]) => [...r]),
      b1: [...p.b1],
      W2: p.W2 ? p.W2.map((r: number[]) => [...r]) : undefined,
      b2: p.b2 ? [...p.b2] : undefined,
      Wout: p.Wout.map((r: number[]) => [...r]),
      bout: [...p.bout],
    });
    const cloneOptState = (s: any) => JSON.parse(JSON.stringify(s || {}));

    const p = deepCopyParams(params);
    const fstate = cloneOptState(fusionState);
    const apstate = cloneOptState(apolloState);
    const sstate = cloneOptState(sageState);
    const e0 = epoch;
    let localBest = bestVal;
    let localWait = wait;
    const hTrainLoss: number[] = [],
      hValLoss: number[] = [],
      hTrainAcc: number[] = [],
      hValAcc: number[] = [];
    let done = 0;

    for (let t = 0; t < n; t++) {
      const ep = e0 + t;
      const rnd = mulberry32(shuffleKey + ep * 101 + 7);
      const order = shuffled(Xtrain.length, rnd);
      for (let start = 0; start < order.length; start += batch) {
        const end = Math.min(order.length, start + batch);
        const idxs = order.slice(start, end);
        const Xb = idxs.map((i) => Xtrain[i]);
        const yb = idxs.map((i) => ytrain[i]);
        const grads = backprop(Xb, yb, p, act as any, { p1: drop1, p2: drop2 }, lambda, rnd, true);

        if (opt === "fusion") {
          stepFusion(p, grads, lr, fstate, {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            wd,
            k,
            sageDecay,
            updateInterval,
            opalAlphaPara,
            opalAlphaOrth,
            kp,
            ki,
            kd,
            iDecay,
            beta3,
            gamma,
            clip,
            enableOPAL: enableOpal,
            enableDEO: enableDeo,
            enableApollo,
            enableSAGE: enableSage,
            enableSGDS: enableSgds,
          });
        } else if (opt === "adam") {
          if (!apstate.vW1) {
            apstate.vW1 = p.W1.map((r: number[]) => r.map(() => 0));
            apstate.vb1 = Array(p.b1.length).fill(0);
            apstate.vWo = p.Wout.map((r: number[]) => r.map(() => 0));
            apstate.vbo = Array(p.bout.length).fill(0);
            apstate.mW1 = p.W1.map((r: number[]) => r.map(() => 0));
            apstate.mb1 = Array(p.b1.length).fill(0);
            apstate.mWo = p.Wout.map((r: number[]) => r.map(() => 0));
            apstate.mbo = Array(p.bout.length).fill(0);
            if (p.W2) {
              apstate.vW2 = p.W2.map((r: number[]) => r.map(() => 0));
              apstate.vb2 = Array(p.b2.length).fill(0);
              apstate.mW2 = p.W2.map((r: number[]) => r.map(() => 0));
              apstate.mb2 = Array(p.b2.length).fill(0);
            }
            apstate.t = 0;
          }
          apstate.t += 1;
          const t1 = apstate.t;
          const beta2 = 0.999,
            eps = 1e-8;
          const apply2d = (W: number[][], g: number[][], v: number[][], m: number[][]) => {
            for (let i = 0; i < W.length; i++)
              for (let j = 0; j < W[0].length; j++) {
                m[i][j] = 0.9 * m[i][j] + 0.1 * g[i][j];
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * g[i][j] * g[i][j];
                const mhat = m[i][j] / (1 - Math.pow(0.9, t1));
                const vhat = v[i][j] / (1 - Math.pow(beta2, t1));
                W[i][j] -= (lr * mhat) / (Math.sqrt(vhat) + eps);
              }
          };
          const apply1d = (b: number[], g: number[], v: number[], m: number[]) => {
            for (let j = 0; j < b.length; j++) {
              m[j] = 0.9 * m[j] + 0.1 * g[j];
              v[j] = beta2 * v[j] + (1 - beta2) * g[j] * g[j];
              const mhat = m[j] / (1 - Math.pow(0.9, t1));
              const vhat = v[j] / (1 - Math.pow(beta2, t1));
              b[j] -= (lr * mhat) / (Math.sqrt(vhat) + eps);
            }
          };
          apply2d(p.W1, grads.dW1, apstate.vW1, apstate.mW1);
          apply1d(p.b1, grads.db1, apstate.vb1, apstate.mb1);
          if (p.W2 && grads.dW2) {
            apply2d(p.W2, grads.dW2, apstate.vW2, apstate.mW2);
            apply1d(p.b2, grads.db2, apstate.vb2, apstate.mb2);
          }
          apply2d(p.Wout, grads.dWo, apstate.vWo, apstate.mWo);
          apply1d(p.bout, grads.dbo, apstate.vbo, apstate.mbo);
        } else if (opt === "momentum") {
          if (!apstate.vW1) {
            apstate.vW1 = p.W1.map((r: number[]) => r.map(() => 0));
            apstate.vb1 = Array(p.b1.length).fill(0);
            apstate.vWo = p.Wout.map((r: number[]) => r.map(() => 0));
            apstate.vbo = Array(p.bout.length).fill(0);
            if (p.W2) {
              apstate.vW2 = p.W2.map((r: number[]) => r.map(() => 0));
              apstate.vb2 = Array(p.b2.length).fill(0);
            }
          }
          const apply2d = (W: number[][], g: number[][], v: number[][]) => {
            for (let i = 0; i < W.length; i++)
              for (let j = 0; j < W[0].length; j++) {
                v[i][j] = momentum * v[i][j] + (1 - momentum) * g[i][j];
                W[i][j] -= lr * v[i][j];
              }
          };
          const apply1d = (b: number[], g: number[], v: number[]) => {
            for (let j = 0; j < b.length; j++) {
              v[j] = momentum * v[j] + (1 - momentum) * g[j];
              b[j] -= lr * v[j];
            }
          };
          apply2d(p.W1, grads.dW1, apstate.vW1);
          apply1d(p.b1, grads.db1, apstate.vb1);
          if (p.W2 && grads.dW2) {
            apply2d(p.W2, grads.dW2, apstate.vW2);
            apply1d(p.b2, grads.db2, apstate.vb2);
          }
          apply2d(p.Wout, grads.dWo, apstate.vWo);
          apply1d(p.bout, grads.dbo, apstate.vbo);
        } else if (opt === "sgd") {
          const apply2d = (W: number[][], g: number[][]) => {
            for (let i = 0; i < W.length; i++)
              for (let j = 0; j < W[0].length; j++) {
                W[i][j] -= lr * g[i][j];
              }
          };
          const apply1d = (b: number[], g: number[]) => {
            for (let j = 0; j < b.length; j++) {
              b[j] -= lr * g[j];
            }
          };
          apply2d(p.W1, grads.dW1);
          apply1d(p.b1, grads.db1);
          if (p.W2 && grads.dW2) {
            apply2d(p.W2, grads.dW2);
            apply1d(p.b2, grads.db2);
          }
          apply2d(p.Wout, grads.dWo);
          apply1d(p.bout, grads.dbo);
        } else if (opt === "apollo") {
          stepApoLLO(p, grads, lr, apstate, { beta2: 0.999, eps: 1e-8, beta3, gamma });
        } else if (opt === "sage") {
          stepSAGE(p, grads, lr, sstate, updateInterval);
        }
      }
      const gTrain = backprop(
        Xtrain,
        ytrain,
        p,
        act as any,
        { p1: 0, p2: 0 },
        lambda,
        mulberry32(999),
        false,
      );
      const gVal = backprop(
        Xval,
        yval,
        p,
        act as any,
        { p1: 0, p2: 0 },
        lambda,
        mulberry32(999),
        false,
      );
      hTrainLoss.push(gTrain.loss);
      hValLoss.push(gVal.loss);
      hTrainAcc.push(gTrain.acc);
      hValAcc.push(gVal.acc);
      if (gVal.loss + 1e-9 < localBest) {
        localBest = gVal.loss;
        localWait = 0;
      } else {
        localWait++;
      }
      done++;
      if (useEarlyStop && localWait >= patience) break;
    }

    setParams(p);
    setFusionState(fstate);
    setApolloState(apstate);
    setSageState(sstate);
    setEpoch((e) => e + done);
    if (hTrainLoss.length) {
      setTrainLoss(hTrainLoss[hTrainLoss.length - 1]);
      setValLoss(hValLoss[hValLoss.length - 1]);
      setTrainAcc(hTrainAcc[hTrainAcc.length - 1]);
      setValAcc(hValAcc[hValAcc.length - 1]);
      setHistLoss((h) => [...h, ...hTrainLoss]);
      setHistValLoss((h) => [...h, ...hValLoss]);
      setHistAcc((h) => [...h, ...hTrainAcc]);
      setHistValAcc((h) => [...h, ...hValAcc]);
    }
    setBestVal(localBest);
    setWait(localWait);
  }

  const trainOneEpoch = () => trainNEpochs(1, false);

  useEffect(() => {
    if (!auto) return;
    if (wait >= patience) {
      setAuto(false);
      return;
    }
    const id = setTimeout(() => trainNEpochs(1, true), 50);
    return () => clearTimeout(id);
  }, [
    auto,
    epoch,
    wait,
    patience,
    lr,
    act,
    drop1,
    drop2,
    lambda,
    clip,
    batch,
    momentum,
    Xtrain,
    ytrain,
    shuffleKey,
    opt,
    k,
    wd,
    opalR,
    opalAlphaPara,
    opalAlphaOrth,
    kp,
    ki,
    kd,
    iDecay,
    beta3,
    gamma,
    sageDecay,
    updateInterval,
    enableOpal,
    enableDeo,
    enableApollo,
    enableSage,
    enableSgds,
  ]);

  const cm = useMemo(() => {
    const { Yhat } = forward(Xval, params, act as any, { p1: 0, p2: 0 }, false, mulberry32(1234));
    const m = [
      [0, 0],
      [0, 0],
    ];
    for (let i = 0; i < yval.length; i++) {
      const p = Yhat[i][0] >= 0.5 ? 1 : 0;
      m[yval[i]][p]++;
    }
    return m;
  }, [Xval, yval, params, act]);

  return (
    <div className="w-full min-h-screen bg-slate-50 text-slate-900">
      <div className="max-w-7xl mx-auto p-4 md:p-6">
        <header className="mb-4 flex items-center justify-between gap-4">
          <h1 className="text-2xl md:text-3xl font-bold">
            Fusion Optimizer Lab (OPAL + DEO + ApoLLO + SAGE + SGD-S)
          </h1>
          <div className="text-sm text-slate-600">Dropout / L2 / Mini-batch / EarlyStop / Viz</div>
        </header>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
          <section className="xl:col-span-1 bg-white rounded-2xl shadow p-4 space-y-4">
            <h2 className="font-semibold text-lg">Controls</h2>
            <div className="space-y-2">
              <label className="block text-sm">Dataset</label>
              <select
                className="w-full rounded-xl border p-2"
                value={dataset}
                onChange={(e) => {
                  setDataset(e.target.value);
                  setEpoch(0);
                }}
              >
                <option>SPIRAL</option>
                <option>CIRCLES</option>
                <option>XOR</option>
                <option>AND</option>
                <option>OR</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm">Hidden 1: {h1}</label>
                <input
                  type="range"
                  min={1}
                  max={24}
                  value={h1}
                  onChange={(e) => setH1(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm">Hidden 2: {h2} (0=off)</label>
                <input
                  type="range"
                  min={0}
                  max={24}
                  value={h2}
                  onChange={(e) => setH2(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm">Activation</label>
                <select
                  className="w-full rounded-xl border p-2"
                  value={act}
                  onChange={(e) => setAct(e.target.value)}
                >
                  <option value="relu">ReLU</option>
                  <option value="silu">SiLU</option>
                  <option value="gelu">GELU</option>
                  <option value="tanh">tanh</option>
                  <option value="sigmoid">Sigmoid</option>
                </select>
              </div>
              <div>
                <label className="block text-sm">Val ratio: {(valRatio * 100).toFixed(0)}%</label>
                <input
                  type="range"
                  min={0.1}
                  max={0.5}
                  step={0.05}
                  value={valRatio}
                  onChange={(e) => setValRatio(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm">Dropout h1: {drop1.toFixed(2)}</label>
                <input
                  type="range"
                  min={0}
                  max={0.6}
                  step={0.05}
                  value={drop1}
                  onChange={(e) => setDrop1(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm">Dropout h2: {drop2.toFixed(2)}</label>
                <input
                  type="range"
                  min={0}
                  max={0.6}
                  step={0.05}
                  value={drop2}
                  onChange={(e) => setDrop2(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm">L2 lambda: {lambda.toFixed(4)}</label>
                <input
                  type="range"
                  min={0}
                  max={0.01}
                  step={0.0001}
                  value={lambda}
                  onChange={(e) => setLambda(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm">Clip norm: {clip.toFixed(1)}</label>
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={0.5}
                  value={clip}
                  onChange={(e) => setClip(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm">Optimizer</label>
                <select
                  className="w-full rounded-xl border p-2"
                  value={opt}
                  onChange={(e) => setOpt(e.target.value)}
                >
                  <option value="fusion">Fusion</option>
                  <option value="adam">Adam</option>
                  <option value="momentum">Momentum</option>
                  <option value="sgd">SGD</option>
                  <option value="apollo">ApoLLO</option>
                  <option value="sage">SAGE</option>
                </select>
              </div>
              <div>
                <label className="block text-sm">LR: {lr.toFixed(3)}</label>
                <input
                  type="range"
                  min={0.001}
                  max={0.5}
                  step={0.001}
                  value={lr}
                  onChange={(e) => setLr(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>

            {opt === "fusion" && (
              <div className="space-y-3 p-2 border rounded-xl bg-indigo-50">
                <h3 className="text-sm font-semibold text-indigo-800">Fusion knobs</h3>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={enableOpal}
                      onChange={(e) => setEnableOpal(e.target.checked)}
                    />
                    <span>OPAL</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={enableDeo}
                      onChange={(e) => setEnableDeo(e.target.checked)}
                    />
                    <span>DEO</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={enableApollo}
                      onChange={(e) => setEnableApollo(e.target.checked)}
                    />
                    <span>ApoLLO</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={enableSage}
                      onChange={(e) => setEnableSage(e.target.checked)}
                    />
                    <span>SAGE</span>
                  </label>
                  <label className="flex items-center gap-2 col-span-2 sm:col-span-1">
                    <input
                      type="checkbox"
                      checked={enableSgds}
                      onChange={(e) => setEnableSgds(e.target.checked)}
                    />
                    <span>SGD-S</span>
                  </label>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs">SAGE k: {k}</label>
                    <input
                      type="range"
                      min={5}
                      max={50}
                      value={k}
                      onChange={(e) => setK(parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs">Weight Decay: {wd.toFixed(4)}</label>
                    <input
                      type="range"
                      min={0}
                      max={0.02}
                      step={0.0005}
                      value={wd}
                      onChange={(e) => setWd(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs">OPAL r: {opalR}</label>
                    <input
                      type="range"
                      min={1}
                      max={8}
                      value={opalR}
                      onChange={(e) => setOpalR(parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs">
                      OPAL alpha parallel: {opalAlphaPara.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min={0.2}
                      max={2.0}
                      step={0.05}
                      value={opalAlphaPara}
                      onChange={(e) => setOpalAlphaPara(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs">
                      OPAL alpha orth: {opalAlphaOrth.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min={0.2}
                      max={2.0}
                      step={0.05}
                      value={opalAlphaOrth}
                      onChange={(e) => setOpalAlphaOrth(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="block text-xs">kp: {kp.toFixed(2)}</label>
                    <input
                      type="range"
                      min={0}
                      max={3}
                      step={0.05}
                      value={kp}
                      onChange={(e) => setKp(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs">ki: {ki.toFixed(2)}</label>
                    <input
                      type="range"
                      min={0}
                      max={2}
                      step={0.05}
                      value={ki}
                      onChange={(e) => setKi(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs">kd: {kd.toFixed(2)}</label>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.02}
                      value={kd}
                      onChange={(e) => setKd(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="block text-xs">iDecay: {iDecay.toFixed(2)}</label>
                    <input
                      type="range"
                      min={0.8}
                      max={0.999}
                      step={0.001}
                      value={iDecay}
                      onChange={(e) => setIDecay(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs">beta3: {beta3.toFixed(2)}</label>
                    <input
                      type="range"
                      min={0.5}
                      max={0.99}
                      step={0.01}
                      value={beta3}
                      onChange={(e) => setBeta3(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs">gamma: {gamma.toFixed(2)}</label>
                    <input
                      type="range"
                      min={0.5}
                      max={2.0}
                      step={0.05}
                      value={gamma}
                      onChange={(e) => setGamma(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs">SAGE decay: {sageDecay.toFixed(2)}</label>
                    <input
                      type="range"
                      min={0.9}
                      max={0.999}
                      step={0.001}
                      value={sageDecay}
                      onChange={(e) => setSageDecay(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs">eig update interval: {updateInterval}</label>
                    <input
                      type="range"
                      min={1}
                      max={20}
                      value={updateInterval}
                      onChange={(e) => setUpdateInterval(parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm">Batch: {batch}</label>
                <input
                  type="range"
                  min={8}
                  max={128}
                  step={8}
                  value={batch}
                  onChange={(e) => setBatch(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm">EarlyStop patience: {patience}</label>
                <input
                  type="range"
                  min={5}
                  max={50}
                  step={1}
                  value={patience}
                  onChange={(e) => setPatience(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>

            <div className="flex items-center gap-2 flex-wrap">
              <button
                className="px-4 py-2 rounded-xl bg-slate-200 text-slate-900"
                onClick={resetTrainingState}
              >
                Reset
              </button>
              <button
                className="px-4 py-2 rounded-xl bg-indigo-600 text-white"
                onClick={trainOneEpoch}
              >
                1 epoch
              </button>
              <button
                className="px-4 py-2 rounded-xl bg-indigo-600 text-white"
                onClick={() => trainNEpochs(10, false)}
              >
                10
              </button>
              <button
                className="px-4 py-2 rounded-xl bg-indigo-600 text-white"
                onClick={() => trainNEpochs(100, false)}
              >
                100
              </button>
              <button
                className={`px-4 py-2 rounded-xl ${auto ? "bg-orange-600" : "bg-orange-500"} text-white`}
                onClick={() => setAuto((v) => !v)}
              >
                {auto ? "Pause auto" : "Auto"}
              </button>
            </div>

            <div className="bg-blue-50 rounded-xl p-3 text-sm space-y-1">
              <div className="font-semibold text-blue-900">Metrics (epoch {epoch})</div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  Train Loss: <b className="text-blue-900">{trainLoss.toFixed(4)}</b>
                </div>
                <div>
                  Train Acc: <b className="text-blue-900">{(trainAcc * 100).toFixed(1)}%</b>
                </div>
                <div>
                  Val Loss: <b className="text-green-900">{valLoss.toFixed(4)}</b>
                </div>
                <div>
                  Val Acc: <b className="text-green-900">{(valAcc * 100).toFixed(1)}%</b>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-2">
              <LineChart
                series={[
                  { name: "train", data: histLoss },
                  { name: "val", data: histValLoss },
                ]}
                yLabel="Loss"
              />
              <LineChart
                series={[
                  { name: "train", data: histAcc },
                  { name: "val", data: histValAcc },
                ]}
                yLabel="Accuracy"
              />
            </div>
          </section>

          <section className="xl:col-span-2 space-y-4">
            <div className="bg-white rounded-2xl shadow p-4">
              <h3 className="font-semibold mb-2">Decision boundary & data</h3>
              <DecisionCanvas
                params={params}
                act={act}
                X={Xtrain.concat(Xval)}
                y={ytrain.concat(yval)}
              />
            </div>

            <div className="bg-white rounded-2xl shadow p-4">
              <h3 className="font-semibold mb-2">Confusion matrix (val)</h3>
              <ConfMatrix cm={cm} />
            </div>
          </section>
        </div>

        <footer className="mt-6 text-xs text-slate-500 text-center">
          Fusion Optimizer Lab — OPAL + DEO + ApoLLO + SAGE + SGD-S
        </footer>
      </div>
    </div>
  );
}
