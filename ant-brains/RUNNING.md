# Running the experiments

Code is written and smoke-tested on CPU in the cloud. Full runs happen on the RTX 4080 (or Colab).
The results get pushed back to GitHub. See "Where things run" in `PLAN.md`.

All commands below are run from the `ant-brains/` folder.

## GPU machine: one-time setup

You need git, Python 3.10 or newer, and an NVIDIA driver from 2024 or later (`nvidia-smi` shows the version).

**Windows (PowerShell):**

```powershell
git clone -b claude/magical-johnson-gz2geq https://github.com/jlpt/claudecode3.git
cd claudecode3\ant-brains
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

If you already have a clone of this repository, use `git fetch origin` then
`git checkout claude/magical-johnson-gz2geq` in it instead of `git clone`.

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
In a later session, only `cd` into `ant-brains` and run `.\.venv\Scripts\Activate.ps1`.

**Linux:** the same, except:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

The last command must print `NVIDIA GeForce RTX 4080`. If it errors with "no CUDA GPUs" or
`Torch not compiled with CUDA enabled`, the CPU build of PyTorch got installed. Fix it with
`pip uninstall -y torch` and then the `--index-url` install again. Install PyTorch before
`requirements.txt`. On Windows, the plain `pip install torch` from PyPI is the CPU build.
`cu126` works with any 2024-or-later driver. `cu130` also exists but needs driver 580 or newer.

## Before every run

```bash
git pull
```

## Step 0: reliable growth

```bash
python experiments/step0_reliable_growth.py --estimate   # about 1 minute: prints the expected run time
python experiments/step0_reliable_growth.py              # the full run: 2 conditions x 3 seeds x 200 generations
```

The script writes everything to `results/step0/` and `results/step0_reliable_growth.txt`. If it gets
interrupted, run the same command again: finished runs are skipped. Keep the PC from going to sleep
during the run.

Then push the results:

```bash
git add results/step0 results/step0_reliable_growth.txt
git commit -m "Step 0 results (RTX 4080)"
git push
```

Options (all scripts follow the same pattern):

| Option | Meaning |
|---|---|
| `--quick` | Tiny run (10 generations, population 16, 2 seeds) to check the code works. Writes to `results/step0_quick/`, which git ignores. |
| `--estimate` | Time one generation of each condition, print the total, and exit. |
| `--seed S --n-seeds K` | Use seeds S, S+1, …, S+K-1 (default 0 and 3). |
| `--config file.json` | Read settings from a JSON file, e.g. `{"generations": 400, "pop": 128}`. Command-line options override it. |
| `--generations`, `--pop`, `--elite`, `--repeats-b`, `--noise-b`, `--sigma` | Override single settings. `--help` lists the defaults. |
| `--out DIR` | Write somewhere other than `results/step0/`. |
| `--fresh` | Redo runs even if a finished checkpoint exists. |

## Google Colab

Runtime → Change runtime type → GPU. Then in a cell:

```python
!git clone -b claude/magical-johnson-gz2geq https://github.com/jlpt/claudecode3.git
%cd claudecode3/ant-brains
!pip install -q -r requirements.txt   # Colab's PyTorch already has CUDA
!python experiments/step0_reliable_growth.py
```

If the repository is private, put a GitHub token in the clone URL:
`https://<token>@github.com/jlpt/claudecode3.git`.
To get the results out, either `git push` from Colab (with the token), or run
`!zip -r step0.zip results/step0 results/step0_reliable_growth.txt`, then download the zip and
commit it from your own machine. Colab disconnects idle sessions, so keep the tab open.

## CPU (cloud session or any machine)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python experiments/step0_reliable_growth.py --quick      # about 3 minutes on 4 cores
```

Tested in the cloud with Python 3.13, torch 2.14.1 (CPU), numpy 2.5, matplotlib 3.11.

## Reproducibility

Each run uses a fixed seed, and every setting is saved in `config.json` next to the results.
The CPU and the GPU use different random number streams, so the same seed gives different runs
on the two. On the GPU, a few operations (for example two ants painting the same square in the
same step) are not guaranteed to be bit-identical between reruns. Growth is chaotic, so a rerun
can come out differently. This is why conclusions need several seeds.
