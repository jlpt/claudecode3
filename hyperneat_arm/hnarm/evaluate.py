"""Runs genomes on the ball-pushing task and scores them.

A trial: the arm starts hanging still, the ball rests on the floor at ``ball_x`` and the target is a spot on the
floor at ``target_x``. The brain has ``duration`` seconds to push the ball onto the target and leave it there.

Fitness per trial (averaged over trials):
    w_progress   * clip(1 - final_error / initial_error, -1, 1)          how much closer the ball ended up
  + w_precision  * exp(-(final_error / tolerance)^2)                     bonus for landing on the target
  + w_direction  * clip(displacement toward target / initial_error, -1, 1)  moved the right way, even if too far
  + w_reach      * exp(-closest_gap / 5 cm)                              shaping: get the arm to the ball at all
  - w_effort     * mean squared muscle activation                        metabolic cost
  - w_saturation * mean over cortical neurons of clip((rate - saturation_hz) / saturation_hz, 0, 1)
where final_error and the final position are averaged over the last ``settle_window`` seconds.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass

import numpy as np

from .brain import BrainSubstrate
from .config import Config
from .hyperneat import build_brains
from .periphery import Muscles, Receptors
from .snn import SpikingNetwork
from .world import ArmBallWorld

# (ball_x, target_x), in contrast pairs: every ball position appears twice, once with the target further out and once
# with it back toward the body. A brain that ignores the target can win at most one of each pair, so any sensitivity
# to where the target is pays off directly. (Without pairs, evolution found swings that only react to where the ball
# is, a few centimetres changing how a fixed swing hits it.)
TRAIN_TRIALS = (
    (-0.25, -0.40), (-0.25, 0.05),
    (-0.15, -0.35), (-0.15, 0.20),
    (0.15, 0.35), (0.15, -0.20),
    (0.25, 0.40), (0.25, -0.05),
)


def random_trials(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        b = rng.uniform(0.15, 0.3) * rng.choice([-1.0, 1.0])
        t = rng.uniform(-0.4, 0.4)
        if abs(t - b) >= 0.15:
            out.append((float(b), float(t)))
    return tuple(out)


@dataclass
class Recording:
    frame_dt: float
    q: np.ndarray  # (F, B, 2)
    ball: np.ndarray  # (F, B, 2)
    act: np.ndarray  # (F, B, 4)
    spike_step: np.ndarray  # (S,) step index of every spike
    spike_batch: np.ndarray  # (S,) which simulation
    spike_neuron: np.ndarray  # (S,) which neuron


def simulate(sub: BrainSubstrate, cfg: Config, weights_t: np.ndarray, bias: np.ndarray, trials, record_every: int = 0,
             noise_seeds=None):
    """Simulate P brains x T trials in one batch. Returns a dict of (P, T) metrics, plus a Recording if asked.
    noise_seeds gives each trial's background-noise seed (default: the trial's index)."""
    env = cfg.env
    P, T = weights_t.shape[0], len(trials)
    B = P * T
    ball_x = np.tile(np.array([t[0] for t in trials]), P)
    target_x = np.tile(np.array([t[1] for t in trials]), P)

    world = ArmBallWorld(env, B)
    world.reset(ball_x, target_x)
    receptors = Receptors(sub, cfg.sensors, env)
    muscles = Muscles(sub, env, B)
    net = SpikingNetwork(sub, cfg.brain, weights_t, bias, T, dt_ms=env.dt * 1000.0, noise_seeds=noise_seeds)

    steps = int(round(env.duration / env.dt))
    settle = int(round(env.settle_window / env.dt))
    err_sum = np.zeros(B)
    x_sum = np.zeros(B)
    min_gap = np.full(B, np.inf)
    effort = np.zeros(B)
    spike_count = np.zeros((P, T, sub.N), np.int32)
    frames_q, frames_ball, frames_act, sp_step, sp_b, sp_n = [], [], [], [], [], []
    act = np.zeros((B, 4))

    for k in range(steps):
        if record_every and k % record_every == 0:
            frames_q.append(world.q.copy()); frames_ball.append(world.ball.copy()); frames_act.append(act.copy())
        spikes = net.step(receptors.currents(world))
        spike_count += spikes
        flat = spikes.reshape(B, sub.N)
        if record_every:
            bi, ni = np.nonzero(flat)
            sp_step.append(np.full(bi.size, k, np.int32)); sp_b.append(bi.astype(np.int32)); sp_n.append(ni.astype(np.int32))
        act = muscles.update(flat[:, sub.motor_slice].astype(np.float32))
        world.step(muscles.torque(act, world.qd))
        effort += (act * act).mean(1)
        min_gap = np.minimum(min_gap, world.gaps.min(1))
        if k >= steps - settle:
            err_sum += np.abs(world.ball[:, 0] - world.target_x)
            x_sum += world.ball[:, 0]

    d0 = np.abs(ball_x - target_x)
    err = err_sum / settle
    progress = np.clip(1.0 - err / d0, -1.0, 1.0)
    precision = np.exp(-((err / env.target_tolerance) ** 2))
    direction = np.clip(np.sign(target_x - ball_x) * (x_sum / settle - ball_x) / d0, -1.0, 1.0)
    reach = np.exp(-np.maximum(min_gap, 0.0) / 0.05)
    effort = effort / steps
    rate_hz = spike_count / env.duration
    excess = np.clip((rate_hz[..., sub.cortical] - cfg.saturation_hz) / cfg.saturation_hz, 0.0, 1.0)
    saturation = excess.mean(-1).reshape(B)
    fitness = (cfg.w_progress * progress + cfg.w_precision * precision + cfg.w_direction * direction
               + cfg.w_reach * reach - cfg.w_effort * effort - cfg.w_saturation * saturation)
    shape = (P, T)
    result = {
        "fitness": fitness.reshape(shape),
        "error": err.reshape(shape),
        "progress": progress.reshape(shape),
        "reach": reach.reshape(shape),
        "effort": effort.reshape(shape),
        "direction": direction.reshape(shape),
        "saturation": saturation.reshape(shape),
        "success": (err < env.target_tolerance).reshape(shape),
        "rate_hz": rate_hz,
    }
    if record_every:
        result["recording"] = Recording(
            frame_dt=record_every * env.dt,
            q=np.array(frames_q), ball=np.array(frames_ball), act=np.array(frames_act),
            spike_step=np.concatenate(sp_step), spike_batch=np.concatenate(sp_b), spike_neuron=np.concatenate(sp_n),
        )
    return result


def region_rates(sub: BrainSubstrate, rate_hz: np.ndarray) -> dict:
    """Mean firing rate of each region, averaged over genomes and trials."""
    r = rate_hz.reshape(-1, sub.N).mean(0)
    return {key: float(r[sub.region_of == i].mean()) for i, key in enumerate(sub.region_keys)}


def noisy_runs(trials, noise_seed: int, repeats: int):
    """Each placement `repeats` times, each run with its own background noise drawn from `noise_seed`."""
    runs = [tr for _ in range(repeats) for tr in trials]
    seeds = [(noise_seed, r, t) for r in range(repeats) for t in range(len(trials))]
    return runs, seeds


def score(genomes, sub: BrainSubstrate, cfg: Config, trials=TRAIN_TRIALS, noise_seed: int = 0, repeats: int = 1):
    """Fitness and summary info for every genome, averaged over all placements and noise repeats."""
    weights_t, bias = build_brains(genomes, sub, cfg.brain)
    runs, seeds = noisy_runs(trials, noise_seed, repeats)
    res = simulate(sub, cfg, weights_t, bias, runs, noise_seeds=seeds)
    out = []
    for i in range(len(genomes)):
        info = {
            "success": float(res["success"][i].mean()),
            "error": float(res["error"][i].mean()),
            "reach": float(res["reach"][i].mean()),
            "effort": float(res["effort"][i].mean()),
            "saturation": float(res["saturation"][i].mean()),
            "rates": region_rates(sub, res["rate_hz"][i : i + 1]),
        }
        out.append((float(res["fitness"][i].mean()), info))
    return out


# --------------------------------------------------------------------------- parallel evaluation

_WORKER = {}


def _init_worker(cfg_dict):
    cfg = Config.from_dict(cfg_dict)
    _WORKER["cfg"] = cfg
    _WORKER["sub"] = BrainSubstrate(cfg.sensors, cfg.brain)


def _score_chunk(task):
    genomes, trials, noise_seed, repeats = task
    return score(genomes, _WORKER["sub"], _WORKER["cfg"], trials, noise_seed, repeats)


class Evaluator:
    """Scores genomes, split over worker processes."""

    def __init__(self, cfg: Config, workers: int = 1, trials=TRAIN_TRIALS, chunk: int = 40):
        self.cfg = cfg
        self.trials = tuple(trials)
        self.workers = workers
        self.chunk = chunk
        self.sub = BrainSubstrate(cfg.sensors, cfg.brain)
        self.pool = None
        if workers > 1:
            # One BLAS thread per worker; the workers already use every core.
            for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            ctx = mp.get_context("spawn")
            self.pool = ctx.Pool(workers, initializer=_init_worker, initargs=(cfg.to_dict(),))

    def run(self, genomes, noise_seed: int = 0, repeats: int = 1, trials=None) -> list:
        """[(fitness, info)] for each genome, without storing anything on the genomes."""
        trials = self.trials if trials is None else tuple(trials)
        if not genomes:
            return []
        n = max(1, min(self.chunk, -(-len(genomes) // max(self.workers, 1))))
        tasks = [(genomes[i : i + n], trials, noise_seed, repeats) for i in range(0, len(genomes), n)]
        if self.pool is None:
            parts = [score(g, self.sub, self.cfg, t, s, r) for g, t, s, r in tasks]
        else:
            parts = self.pool.map(_score_chunk, tasks)
        return [r for part in parts for r in part]

    def evaluate(self, genomes, noise_seed: int = 0, repeats: int = 1) -> None:
        """Sets fitness and info on every genome that has no fitness yet."""
        todo = [g for g in genomes if g.fitness is None]
        for g, (f, info) in zip(todo, self.run(todo, noise_seed, repeats)):
            g.fitness, g.info = f, info

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
