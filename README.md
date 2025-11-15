# Fusion Optimizer Lab (React + TS + Vite)

This demo implements a playground for a **fusion optimizer** that combines ideas from OPAL, DEO (PID in gradient-space), ApoLLO (directional confidence), SAGE (low-rank eigenspace preconditioning), and SGD-S style group-consensus for matrix-shaped weights.

## Quick start

```bash
pnpm i   # or npm i / yarn
pnpm dev # open the shown URL
```

All knobs are exposed in the left panel. Switch between `Fusion`, `Adam`, `Momentum`, `SGD`, `ApoLLO`, and `SAGE` from the Optimizer dropdown.
