# Ant-grown brains

Ants on a grid grow a brain. Every ant carries the same tiny neural network (the DNA), which decides
where it walks, where it drops neurons and how strong the wires it lays are. Evolution, not
backprop, improves the DNA. The goal is DNA whose grown wiring matches a real fly connectome, and
then the smallest DNA that still does. See [`PLAN.md`](PLAN.md) for the full plan.

This folder is self-contained; the rest of the repository is an unrelated project.

| Path | What's there |
|---|---|
| `PLAN.md` | The step-by-step plan and what was learned so far |
| `RUNNING.md` | Setup and run commands for the GPU machine, Colab and CPU |
| `antbrains/` | The library: growth, brains, evolution (`growlab.py`) |
| `experiments/` | One script per step, e.g. `step0_reliable_growth.py` |
| `results/` | Output of each step, plus `LOG.md` with a short summary of each |
| `archive/` | The original notebook and earlier experiments, for reference |

Status: Step 0 is ready for the full GPU run.
