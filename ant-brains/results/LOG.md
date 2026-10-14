# Results log

One short entry per step. Details are in each step's folder.

## Step 0: reliable growth

- 2026-10-14: The notebook is now `experiments/step0_reliable_growth.py`. The packaged code gives
  bit-identical genomes, curves and test results to the notebook code for the same seed
  (checked on CPU with growth shortened to 300 steps). `--quick` (10 generations, population 16,
  2 seeds) ran end to end on the cloud CPU (4 cores) in 2 min 39 s. The full GPU run is pending.
