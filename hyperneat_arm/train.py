#!/usr/bin/env python3
"""Evolve a spiking sensorimotor brain with HyperNEAT to push a ball onto a target with a two-link arm.

    python train.py --generations 300 --workers 4 --out runs/demo
    python train.py --resume runs/demo --generations 100      # continue from the last checkpoint

Every generation sees fresh background noise and plays each placement --repeats times, so a brain cannot score by
memorising one noise pattern; it has to push the ball reliably. Every --validate-every generations the best few
genomes are re-scored on held-back validation noise, and the best of those is the champion.

Writes to the output folder: log.csv (one row per generation), champion.json (the validated champion, with the config
it was evolved under) and checkpoint.pkl (the whole population, for --resume).
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import time
from pathlib import Path

from hnarm.brain import N_CPPN_INPUTS, N_CPPN_OUTPUTS, BrainSubstrate
from hnarm.config import Config
from hnarm.evaluate import Evaluator
from hnarm.io import load_checkpoint, save_champion, save_checkpoint
from hnarm.neat import Population

VALIDATION_SEED = 10**6  # never used for training noise (those seeds are generation numbers)
LOG_FIELDS = ["generation", "best", "mean", "success", "error", "species", "threshold", "cppn_hidden",
              "cppn_conns", "champion", "champion_success", "seconds"]


def validate(pop: Population, evaluator: Evaluator, cfg_out: tuple, top: int, repeats: int) -> None:
    out, cfg, gen = cfg_out
    cands = sorted(pop.genomes, key=lambda g: g.fitness, reverse=True)[:top]
    for g, (score, info) in zip(cands, evaluator.run(cands, VALIDATION_SEED, repeats)):
        if score > pop.champion_score:
            pop.champion_score = score
            pop.champion = copy.deepcopy(g)
            pop.champion.fitness = score
            pop.champion.info = dict(info, validated_generation=gen)
            save_champion(out / "champion.json", pop.champion, cfg, gen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/demo", help="output folder")
    ap.add_argument("--resume", metavar="RUN", help="continue the run in this folder")
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--population", type=int, default=None, help="population size (default from config: 150)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)), help="evaluation processes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=None, help="seconds per trial (default 3)")
    ap.add_argument("--repeats", type=int, default=2, help="noise repeats of every placement per generation")
    ap.add_argument("--validate-every", type=int, default=5)
    ap.add_argument("--validate-top", type=int, default=3)
    ap.add_argument("--validate-repeats", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    args = ap.parse_args()

    if args.resume:
        out = Path(args.resume)
        pop, cfg = load_checkpoint(out / "checkpoint.pkl")
        print(f"Resuming {out} at generation {pop.generation}")
    else:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        cfg = Config()
        if args.population:
            cfg.neat.population = args.population
        if args.duration:
            cfg.env.duration = args.duration
        # Seed the topographic Gaussian on z_pre (input 0) and z_post (input 3).
        pop = Population(cfg.neat, N_CPPN_INPUTS, N_CPPN_OUTPUTS, seed=args.seed, seed_topographic_inputs=(0, 3))
        pop.champion, pop.champion_score = None, -math.inf

    print(BrainSubstrate(cfg.sensors, cfg.brain).describe())
    log_path = out / "log.csv"
    new_log = not log_path.exists()
    evaluator = Evaluator(cfg, workers=args.workers)
    try:
        with open(log_path, "a", newline="") as f:
            log = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new_log:
                log.writeheader()
            end = pop.generation + args.generations
            while pop.generation < end:
                t0 = time.time()
                gen = pop.generation
                for g in pop.genomes:  # fresh noise: elites are re-scored too
                    g.fitness = None
                evaluator.evaluate(pop.genomes, noise_seed=gen + 1, repeats=args.repeats)
                best = max(pop.genomes, key=lambda g: g.fitness)
                mean = sum(g.fitness for g in pop.genomes) / len(pop.genomes)
                if gen % args.validate_every == 0 or gen == end - 1:
                    validate(pop, evaluator, (out, cfg, gen), args.validate_top, args.validate_repeats)
                hidden, conns = best.size()
                champ = pop.champion
                row = {
                    "generation": gen, "best": round(best.fitness, 4), "mean": round(mean, 4),
                    "success": round(best.info["success"], 3), "error": round(best.info["error"], 4),
                    "cppn_hidden": hidden,
                    "cppn_conns": conns, "champion": round(pop.champion_score, 4),
                    "champion_success": round(champ.info["success"], 3) if champ else 0.0,
                }
                pop.evolve()  # speciates and breeds the next generation
                row.update(species=len(pop.species), threshold=round(pop.threshold, 2),
                           seconds=round(time.time() - t0, 1))
                log.writerow(row)
                f.flush()
                print(f"gen {gen:4d}  best {row['best']:+.3f}  mean {row['mean']:+.3f}  success {row['success']:.2f}"
                      f"  err {row['error']:.3f} m  species {row['species']:2d}  cppn {hidden}h/{conns}c"
                      f"  | champion {row['champion']:+.3f} ({row['champion_success']:.2f} on target)"
                      f"  {row['seconds']:.1f}s", flush=True)
                if pop.generation % args.checkpoint_every == 0 or pop.generation == end:
                    save_checkpoint(out / "checkpoint.pkl", pop, cfg)
    finally:
        evaluator.close()
    print(f"Done. Champion validation fitness {pop.champion_score:.3f}, saved to {out / 'champion.json'}")
    print(f"View it with: python replay.py {out / 'champion.json'} --html {out / 'replay.html'}")


if __name__ == "__main__":
    main()
