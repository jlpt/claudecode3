# Ant-Grown Brains: Project Plan

A plan for Claude Code. Read this whole file before starting, then work through the steps in order. Each step ends with a check that must pass before moving on.

## The idea in one paragraph

A brain is not designed directly. It is *grown* by ants on a grid. Every ant carries the same tiny neural network (the **DNA**). Each step, an ant senses its surroundings and the DNA network decides what it does: turn, paint a colour, drop a neuron, split into two ants, stop, and how strong to make the wire it lays. Wires form when an ant walks from one neuron to another; repeated trips strengthen the wire, like extra synapses. Evolution (no backprop) improves the DNA. The new goal: **evolve DNA whose grown wiring matches a real fly connectome, then find the smallest DNA that still does**. That measures how much information a brain's wiring really needs.

## What we learned so far (don't relearn these)

1. **Grown wiring can help learning.** On a bit-counting task, 3 of 4 turmite-grown networks beat randomly rewired networks with the same wire counts (75–88% vs ~45% accuracy). They had 50–80% two-way wires vs ~5% for random.
2. **Ants can set weights too, but evolution is weak at it.** Ants setting neurons, wiring and weights reached MSE 2.2 on counting (guessing = 4.0). Backprop on the *same* wiring reached 0.10. The wiring was capable; precise weights were the bottleneck.
3. **Bigger genomes didn't help at small budgets.** Rule tables from 4 to 128 entries all landed around MSE 1.9–3.1. Run-to-run noise was larger than any effect. The 96-neuron cap also meant bigger genomes couldn't build bigger brains.
4. **The core problem is chaotic growth.** Tiny DNA changes flip a near-tied ant decision, which cascades into a completely different brain. Switching from a rule table to neural-network DNA did *not* fix it: even nudging weights by 0.002 changed most children's performance. Growth amplifies small differences.
5. **The proposed fix: select for reliable growth.** Grow each genome several times with small noise in the ants' decisions and score the average, so evolution favours DNA that grows the same good brain consistently (biology calls this canalization). This is implemented in `ant_brains_gpu.ipynb` but **has not been run yet**.

## Files in this bundle

| File | What it is |
|---|---|
| `growlab.py` | GPU-batched growth + brain + evolution. Neural-network DNA, dense wiring, 96-neuron cap. **Start from this.** |
| `ant_brains_gpu.ipynb` | Notebook version of `growlab.py` running the reliable-growth experiment (single growth vs. 4 noisy growths). |
| `neural_dna.py` | CPU version comparing rule-table DNA vs. neural DNA (finding 4). |
| `genome_sweep.py` | Rule-table genome-size sweep with/without Hebbian lifetime learning (finding 3). |
| `evolve_everything.py` | Ants set neurons, wiring and weights; compared with backprop (finding 2). |
| `evaluate_wiring.py` | Grown wiring vs. random vs. ring wiring on a graph of tiny MLPs (finding 1). |
| `graph_of_mlps.py` | The original prototype: 256 tiny MLPs in a sparse, two-way, small-world graph. |
| `turmite_lab.html` | Interactive browser page: watch rule-table ants grow a network, mutate and evolve rules. |

## Where things run

Claude Code works in a **cloud session with no GPU and limited internet**. Heavy runs happen on the user's **RTX 4080 (16 GB)** or on Google Colab. So the workflow is:

1. **Cloud (Claude Code):** write code, run small CPU tests (quick mode, tiny populations, few generations), commit and push to the GitHub repo.
2. **GPU machine (the user):** `git pull`, run the full experiment script, commit the files in `results/`, push.
3. **Cloud (Claude Code):** pull, read the results, report them, plan the next change.

Rules that follow from this:
- Every experiment is a **script with command-line options** (`--quick`, `--seed`, `--config`), not something run by hand. Each one writes all its output to `results/<step>/` (CSV logs, a summary `.md`, plots as `.png`, checkpoints).
- Every script must **run on CPU in `--quick` mode** in a few minutes, so it can be tested in the cloud before the user spends GPU time.
- Code picks the device automatically (`cuda` if available, else `cpu`). Keep everything batched on the GPU: all brains in a generation grow at once, every ant in every brain moving in one tensor operation per step.
- When handing off a GPU run, tell the user the exact commands to run, roughly how long it should take (from the benchmark), and which files to push back.
- Keep a `requirements.txt` and a short `RUNNING.md` with setup commands for the GPU machine (CUDA PyTorch install) and for Colab.
- Don't download data from the cloud session; it will usually be blocked. See Step 2.

---

## Step 0: Set up and reproduce

1. Organise the repo: code in `antbrains/`, experiment scripts in `experiments/`, results in `results/`. Add `requirements.txt` (torch, numpy, networkx, scipy, pandas, matplotlib) and `RUNNING.md`.
2. Turn the code in `ant_brains_gpu.ipynb` into `experiments/step0_reliable_growth.py` with `--quick`. Test it on CPU in the cloud with `--quick`.
3. Hand off to the user: they run it on the 4080 (first `python -c "import torch; print(torch.cuda.get_device_name(0))"` to confirm the GPU), then the full run, and push `results/step0/`.

**Check:** quick mode runs end to end on CPU; the user's full GPU run produces the summary. Save the summary to `results/step0_reliable_growth.txt`. Report whether condition B (reliable growth) produced more children within 0.25 of their parent and a lower median than condition A. This result decides how much noise-averaging to use in later steps.

## Step 1: Make it scale (sparse wiring + config)

The current code stores each brain as a full N × N table (96% zeros). At fly scale that would be ~90 GB per brain, so switch to sparse.

1. Put every size setting in one config object: grid width/height, colours, max ants, max neurons, growth steps, DNA hidden size, population, elite count, repeats, noise, mutation size.
2. Replace the dense `Wm` with a sparse wire list per brain: `(src, dst, weight, synapse_count)`. Accumulate repeat trips into `synapse_count` and `weight`. Use scatter/gather (or `torch.sparse`) for running brains.
3. Keep a dense fallback for small brains (dense is faster under ~1,000 neurons on a GPU). Choose automatically by size.
4. Write `benchmark.py`: for neurons in [96, 1k, 3k, 10k, 30k] and population in [32, 128, 512], report growth time per generation and peak GPU memory. Stop a row when it runs out of memory.

**Check:** with the same seed, sparse and dense versions produce identical wiring and identical brain outputs at 96 neurons. Save the benchmark table to `results/step1_benchmark.md`.

## Step 2: Get the target connectomes

Use one standard format for every connectome: a CSV with columns `pre, post, synapses` (one row per connected neuron pair, `synapses` = number of synapses), plus optional per-neuron columns (`id, x, y, z, cell_type`).

1. **Warm-up target: *C. elegans*** (302 neurons). Use the hermaphrodite chemical-synapse connectome (for example Cook et al. 2019 / WormWiring).
2. **Main target: larval *Drosophila*** (~3,000 neurons, ~550,000 synapses), from Winding et al., *Science* 2023, "The connectome of an insect brain". The connectivity matrices are in the paper's supplementary data.
3. **Later target: adult *Drosophila*** (FlyWire, ~140,000 neurons, ~50 million synapses), from the FlyWire Codex download page.
4. Write `load_connectome.py` that converts each source into the standard CSV, and prints neuron count, connection count, synapse count, and the fraction of two-way connections.

The cloud session can't download these. Tell the user exactly which file to get and where to put it (`data/raw/<dataset>/`). Small converted CSVs (worm, larval fly) can be committed to the repo. The adult FlyWire data is several GB: keep it out of git (add it to `.gitignore`) and have the loader run on the GPU machine, or use Git LFS. For cloud testing, include a small synthetic connectome generator so the pipeline can be tested without real data.

**Check:** the printed counts roughly match the published numbers for each dataset.

## Step 3: The "fly-likeness" score

Matching neuron-by-neuron is intractable (it requires solving which grown neuron corresponds to which real one). Match **statistics** instead. Write `connectome_stats.py` computing, for any wire list:

1. In-degree and out-degree distributions (compare with a Kolmogorov–Smirnov or Wasserstein distance on log-scaled degrees, so hubs matter)
2. Reciprocity: fraction of connections that are two-way
3. Clustering coefficient (directed or undirected, but be consistent)
4. Three-neuron motif counts (the 16 directed triad types), normalised against degree-preserving random graphs
5. Synapses-per-connection distribution
6. Connection density (connections / possible pairs)
7. Optional, when the target has positions: connection probability vs. distance

Combine them into one score: a weighted sum of normalised distances, where 0 = identical statistics.

**Check (validation, do before any evolution):**
- The target scored against itself = 0.
- A degree-preserving shuffle of the target scores worse than the target but better than a random graph.
- A random (Erdős–Rényi) graph with the same neuron and connection counts scores clearly worst.
- Two different random seeds of the same random-graph model score similarly (the metric isn't too noisy).

Save these numbers to `results/step3_metric_validation.md`. If the ordering is wrong, fix the metric before continuing.

## Step 4: Phase 1, evolve DNA that grows fly-like wiring

1. Fitness = fly-likeness score of the grown wiring (lower is better). Add a penalty if the neuron count misses the target size.
2. Use reliable growth from Step 0: grow each genome R times with decision noise, average the score. Pick R and noise from Step 0's results (default R = 4, noise = 0.3).
3. Warm up on *C. elegans* first. Set the neuron cap and grid so the right number of neurons fits.
4. Then the larval fly: neuron cap ~3,000, larger grid, more ants and steps. Use the Step 1 benchmark to choose a population size that gives generations of under ~30 seconds.
5. Log every generation to CSV (best, median, each statistic's distance, neurons, connections). Save the best genome every 25 generations as a checkpoint. Make runs resumable.
6. Baselines to report alongside: a random graph, and a simple 2-rule generative model (connect by distance cost + "similar neighbours connect"), fitted to the same target.

**Check:** the best evolved genome beats the random-graph baseline on the combined score, and the result holds across 3 seeds. Report which statistics matched well and which didn't.

## Step 5: Phase 2, find the smallest DNA

1. Starting from the best Phase 1 genome, search for smaller DNA that still matches:
   - shrink the hidden layer (16 → 12 → 8 → 4 units), retraining by evolution each time, starting from the larger solution where possible
   - add a mutation that zeros out a weight, plus a fitness penalty per nonzero weight
2. Measure DNA size as the number of nonzero genes, and also as bits (nonzero genes × bits needed per gene after rounding to the coarsest precision that keeps the match).
3. Plot match quality vs. DNA size (a Pareto curve). Mark the size of the target's raw wiring list in bits for comparison: that ratio is the **compression factor**.

**Check:** `results/step5_pareto.png` and a table of DNA size vs. match score for 3 seeds.

## Step 6: Grow big

1. Take the best larval-fly DNA and grow it larger (bigger grid, more ants, more steps), without re-evolving. Does the wiring stay fly-like as it grows?
2. Then compare against the adult FlyWire statistics, and fine-tune with a few generations at full size if needed.

## Optional extensions

- **LIF neurons:** leaky integrate-and-fire neurons, with the DNA setting each neuron's leak and threshold. Evolution doesn't need gradients, so spikes cause no trouble.
- **Function test:** do fly-like grown brains learn a task better than random brains with the same counts? (Repeat finding 1 at fly scale.)
- **3D growth:** grow in a 3D volume instead of a 2D grid.

## Ground rules

- Keep runs reproducible: fixed seeds, settings saved with every result.
- Run small first (quick mode) before any long run, and estimate the run time before starting it.
- Don't overclaim. Report run-to-run spread, use at least 3 seeds for conclusions, and say plainly when a result is inconclusive.
- Put results in `results/`, with a short summary of each step in `results/LOG.md`.
