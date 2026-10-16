#!/usr/bin/env python3
"""Replay an evolved brain: score it on the training trials and on held-out ones, and export an HTML viewer.

    python replay.py runs/demo/champion.json --html runs/demo/replay.html
    python replay.py runs/demo/champion.json --held-out 20 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hnarm.brain import TRACTS, BrainSubstrate
from hnarm.evaluate import TRAIN_TRIALS, noisy_runs, random_trials, region_rates, simulate
from hnarm.hyperneat import build_brain
from hnarm.io import load_champion
from hnarm.viewer import write_page


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("champion", help="champion.json written by train.py")
    ap.add_argument("--html", help="write the viewer to this file")
    ap.add_argument("--held-out", type=int, default=6, help="number of random unseen ball/target placements")
    ap.add_argument("--seed", type=int, default=1, help="seed for the held-out placements")
    ap.add_argument("--noise-seed", type=int, default=7, help="background-noise seed of the recorded trials")
    ap.add_argument("--robustness", type=int, default=10, help="also replay every placement this many times with "
                    "different noise and report how often the ball lands on target (0 to skip)")
    ap.add_argument("--fragment", action="store_true", help="omit the doctype/head (for hosts that add their own)")
    args = ap.parse_args()

    genome, cfg, generation = load_champion(Path(args.champion))
    sub = BrainSubstrate(cfg.sensors, cfg.brain)
    w, b, expressed = build_brain(genome, sub, cfg.brain)
    held = random_trials(args.held_out, seed=args.seed) if args.held_out else ()
    trials = tuple(TRAIN_TRIALS) + tuple(held)
    seeds = [(args.noise_seed, 0, t) for t in range(len(trials))]
    res = simulate(sub, cfg, w[None], b[None], trials, record_every=10, noise_seeds=seeds)

    print(f"Champion from generation {generation}, training fitness {genome.fitness:.3f}")
    print(f"{len(sub.pre)} candidate synapses, {int(expressed.sum())} expressed:")
    for i, t in enumerate(TRACTS):
        m = sub.tract == i
        print(f"  {t:12s} {int(expressed[m].sum()):5d} / {int(m.sum())}")
    print("\n  set       ball   target   final error   result")
    for k, (bx, tx) in enumerate(trials):
        kind = "train" if k < len(TRAIN_TRIALS) else "held-out"
        ok = "on target" if res["success"][0, k] else "missed"
        print(f"  {kind:8s} {bx * 100:+5.0f}cm {tx * 100:+6.0f}cm   {res['error'][0, k] * 100:6.1f} cm     {ok}")
    n = len(TRAIN_TRIALS)
    print(f"\n  training on target: {int(res['success'][0, :n].sum())}/{n}", end="")
    if held:
        print(f"   held-out on target: {int(res['success'][0, n:].sum())}/{len(held)}", end="")
    if args.robustness:
        runs, rseeds = noisy_runs(trials, args.noise_seed + 1, args.robustness)
        rob = simulate(sub, cfg, w[None], b[None], runs, noise_seeds=rseeds)["success"].reshape(args.robustness, -1)
        print(f"\n  over {args.robustness} noise repeats per placement: training {rob[:, :n].mean():.0%} on target", end="")
        if held:
            print(f", held-out {rob[:, n:].mean():.0%}", end="")
    print("\n\n  mean firing rate per region (Hz): " +
          ", ".join(f"{k} {v:.0f}" for k, v in region_rates(sub, res["rate_hz"]).items()))

    if args.html:
        hidden, conns = genome.size()
        meta = {"generation": generation, "fitness": genome.fitness, "neurons": sub.N, "synapses": int(expressed.sum()),
                "cppn_hidden": hidden, "cppn_conns": conns, "n_train": n}
        write_page(Path(args.html), sub, cfg, res["recording"], res, trials, meta, expressed,
                   standalone=not args.fragment)
        print(f"\nWrote {args.html}")


if __name__ == "__main__":
    main()
