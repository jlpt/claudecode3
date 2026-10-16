"""Batched spiking network of Izhikevich neurons with exponentially decaying, current-based synapses.

Shapes: P genomes (each its own weight matrix) x T trials x N neurons. Afferents (the first n_inputs neurons) are
driven by receptor currents instead of synapses. Each step:

    I     = tonic bias + W . x + noise              (x = synaptic trace of every presynaptic neuron)
    v'    = 0.04 v^2 + 5 v + 140 - u + I            (two 0.5 ms half steps, as in Izhikevich 2003)
    u'    = a (b v - u)
    spike when v >= 30 mV, then v <- c, u <- u + d
    x     <- x * exp(-dt / tau) + spike             (tau 5 ms for excitatory cells, 10 ms for inhibitory)
"""

from __future__ import annotations

import numpy as np

from .brain import BrainSubstrate
from .config import BrainConfig


class SpikingNetwork:
    def __init__(self, sub: BrainSubstrate, cfg: BrainConfig, weights_t: np.ndarray, bias: np.ndarray, trials: int,
                 dt_ms: float = 1.0, noise_seeds=None):
        """weights_t (P, N, n_post): presynaptic x postsynaptic. bias (P, n_post): tonic current.
        noise_seeds: one seed (an int or a tuple of ints) per trial for its background noise."""
        self.sub = sub
        self.cfg = cfg
        P = weights_t.shape[0]
        self.P, self.T, self.N = P, trials, sub.N
        self.n_in = sub.n_inputs
        self.dt = dt_ms
        self.W = weights_t.astype(np.float32)
        self.bias = bias.astype(np.float32)[:, None, :]
        f32 = np.float32
        self.a, self.b, self.c, self.d = (x.astype(f32) for x in (sub.izh_a, sub.izh_b, sub.izh_c, sub.izh_d))
        tau = np.where(sub.inhibitory, cfg.tau_inh, cfg.tau_exc)
        self.decay = np.exp(-dt_ms / tau).astype(f32)
        # Background noise. Every genome gets the same noise in a given trial (common random numbers), so differences
        # in fitness within a generation come from the networks, not from luck. Each trial's noise depends only on
        # its own seed, never on which other trials share the batch.
        if noise_seeds is None:
            noise_seeds = range(trials)
        self.noise = np.empty((cfg.noise_bank, trials, sub.N), f32)
        for t, seed in enumerate(noise_seeds):
            seed = list(seed) if isinstance(seed, (tuple, list)) else seed
            self.noise[:, t] = cfg.noise_std * np.random.default_rng(seed).standard_normal((cfg.noise_bank, sub.N))
        self.reset()

    def reset(self) -> None:
        shape = (self.P, self.T, self.N)
        self.v = np.full(shape, -65.0, np.float32)
        self.u = (self.b * self.v).astype(np.float32)
        self.x = np.zeros(shape, np.float32)
        self.t = 0

    def step(self, input_current: np.ndarray) -> np.ndarray:
        """input_current (P*T, n_inputs) or (P, T, n_inputs). Returns this step's spikes (P, T, N) as bool."""
        I = np.empty((self.P, self.T, self.N), np.float32)
        I[..., : self.n_in] = input_current.reshape(self.P, self.T, self.n_in)
        np.matmul(self.x, self.W, out=I[..., self.n_in:])
        I[..., self.n_in:] += self.bias
        I += self.noise[self.t % len(self.noise)]
        np.clip(I, -self.cfg.max_current, self.cfg.max_current, out=I)

        v, u = self.v, self.u
        h = np.float32(0.5 * self.dt)
        for _ in range(2):
            v += h * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
            np.clip(v, -100.0, 30.0, out=v)
        u += np.float32(self.dt) * self.a * (self.b * v - u)
        spikes = v >= 30.0
        np.copyto(v, np.broadcast_to(self.c, v.shape), where=spikes)
        np.add(u, self.d, out=u, where=spikes)
        self.x *= self.decay
        self.x += spikes
        self.t += 1
        return spikes
