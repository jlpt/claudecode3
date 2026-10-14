# Archive

The files from the original bundle, kept for reference. They are not part of the
step-by-step pipeline in `PLAN.md` and are not maintained.

| File | What it is |
|---|---|
| `ant_brains_gpu.ipynb` | The Colab notebook for the reliable-growth experiment. Its code is now `antbrains/growlab.py` and `experiments/step0_reliable_growth.py`. |
| `neural_dna.py` | CPU version comparing rule-table DNA vs. neural DNA (finding 4). Imports `genome_sweep.py`, so run it from this folder. |
| `genome_sweep.py` | Rule-table genome-size sweep with/without Hebbian lifetime learning (finding 3). |
| `evolve_everything.py` | Ants set neurons, wiring and weights; compared with backprop (finding 2). |
| `evaluate_wiring.py` | Grown wiring vs. random vs. ring wiring on a graph of tiny MLPs (finding 1). |
| `graph_of_mlps.py` | The original prototype: 256 tiny MLPs in a sparse, two-way, small-world graph. |
| `turmite_lab.html` | Interactive browser page: watch rule-table ants grow a network, mutate and evolve rules. Open it in a browser. |
