"""HyperNEAT: paint the synaptic weights of the brain substrate with a CPPN.

For every synapse the connectome allows, the CPPN is queried with the coordinates of the two neurons. The output for
that synapse's tract decides whether it exists (above the expression threshold) and how strong it is. Its sign is
fixed by the presynaptic cell (Dale's principle). A separate output, queried with only the postsynaptic coordinates,
sets each neuron's tonic current.
"""

from __future__ import annotations

import numpy as np

from .brain import BIAS_OUTPUT, N_CPPN_INPUTS, N_CPPN_OUTPUTS, BrainSubstrate
from .config import BrainConfig
from .neat import CPPN, Genome


def build_brain(genome: Genome, sub: BrainSubstrate, cfg: BrainConfig):
    """Returns (weights_t (N, n_post), bias (n_post,), expressed mask over sub.pre/sub.post)."""
    cppn = CPPN(genome, N_CPPN_INPUTS, N_CPPN_OUTPUTS)
    out = cppn(sub.query)
    raw = out[np.arange(len(sub.pre)), sub.tract]
    thr = cfg.expression_threshold
    mag = np.clip((raw - thr) / (1.0 - thr), 0.0, 1.0)
    w = (mag * sub.peak_weight).astype(np.float32)
    weights_t = np.zeros((sub.N, sub.n_post), np.float32)
    weights_t[sub.pre, sub.post - sub.n_inputs] = w
    bias = (cfg.bias_gain * cppn(sub.bias_query)[:, BIAS_OUTPUT]).astype(np.float32)
    return weights_t, bias, mag > 0.0


def build_brains(genomes, sub: BrainSubstrate, cfg: BrainConfig):
    ws, bs = [], []
    for g in genomes:
        w, b, _ = build_brain(g, sub, cfg)
        ws.append(w)
        bs.append(b)
    return np.stack(ws), np.stack(bs)
