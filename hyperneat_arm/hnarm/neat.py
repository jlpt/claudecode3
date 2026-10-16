"""NEAT (NeuroEvolution of Augmenting Topologies, Stanley & Miikkulainen 2002) for evolving CPPNs.

Genomes are feed-forward graphs whose hidden nodes pick their activation function from a set (sin, Gaussian, ...),
which is what makes them CPPNs. Includes historical markings, speciation with an adaptive compatibility threshold,
fitness sharing, stagnation, elitism, crossover and structural mutation.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

import numpy as np

from .config import NeatConfig

# --------------------------------------------------------------------------- activation functions


def _clip(x):
    return np.clip(x, -30.0, 30.0)


ACTIVATIONS = {
    "tanh": np.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-4.9 * _clip(x))),
    "sin": lambda x: np.sin(_clip(x)),
    "gauss": lambda x: np.exp(-_clip(x) ** 2),
    "abs": lambda x: np.minimum(np.abs(x), 10.0),
    "identity": lambda x: np.clip(x, -10.0, 10.0),
    "relu": lambda x: np.clip(x, 0.0, 10.0),
}


@dataclass
class NodeGene:
    id: int
    kind: str  # "input", "output" or "hidden"
    act: str
    bias: float = 0.0


@dataclass
class ConnGene:
    innov: int
    src: int
    dst: int
    w: float
    on: bool = True


@dataclass
class Genome:
    id: int
    nodes: dict = field(default_factory=dict)  # id -> NodeGene
    conns: dict = field(default_factory=dict)  # innov -> ConnGene
    fitness: float | None = None
    info: dict = field(default_factory=dict)  # extra evaluation results (success rate, ...)

    def size(self) -> tuple[int, int]:
        hidden = sum(1 for n in self.nodes.values() if n.kind == "hidden")
        return hidden, sum(1 for c in self.conns.values() if c.on)


class Innovations:
    """Global historical markings so that the same structural change gets the same id in every genome."""

    def __init__(self, first_node: int):
        self.next_node = first_node
        self.next_conn = 0
        self.conn_ids: dict[tuple[int, int], int] = {}
        self.split_ids: dict[int, int] = {}

    def conn(self, src: int, dst: int) -> int:
        key = (src, dst)
        if key not in self.conn_ids:
            self.conn_ids[key] = self.next_conn
            self.next_conn += 1
        return self.conn_ids[key]

    def split(self, innov: int, genome: Genome) -> int:
        nid = self.split_ids.get(innov)
        if nid is None or nid in genome.nodes:
            nid = self.next_node
            self.next_node += 1
            if innov not in self.split_ids:
                self.split_ids[innov] = nid
        return nid


# --------------------------------------------------------------------------- compiled network


class CPPN:
    """A genome compiled for fast vectorised evaluation over many query points at once."""

    def __init__(self, genome: Genome, n_in: int, n_out: int):
        self.n_in, self.n_out = n_in, n_out
        self.outputs = list(range(n_in, n_in + n_out))
        incoming: dict[int, list[tuple[int, float]]] = {}
        for c in genome.conns.values():
            if c.on:
                incoming.setdefault(c.dst, []).append((c.src, c.w))
        # Only nodes that feed an output matter.
        needed, stack = set(), list(self.outputs)
        while stack:
            n = stack.pop()
            if n in needed:
                continue
            needed.add(n)
            stack.extend(s for s, _ in incoming.get(n, []))
        order, seen = [], set(range(n_in))

        def visit(n, path=()):
            if n in seen:
                return
            if n in path:
                raise ValueError("cycle in CPPN")
            for s, _ in incoming.get(n, []):
                visit(s, path + (n,))
            seen.add(n)
            order.append(n)

        for o in self.outputs:
            visit(o)
        self.steps = []
        for n in order:
            node = genome.nodes[n]
            srcs = [(s, w) for s, w in incoming.get(n, []) if s in needed or s < n_in]
            self.steps.append((n, ACTIVATIONS[node.act], node.bias, srcs))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        vals = {i: X[:, i] for i in range(self.n_in)}
        M = X.shape[0]
        for n, fn, bias, srcs in self.steps:
            s = np.full(M, bias, dtype=X.dtype)
            for src, w in srcs:
                s += w * vals[src]
            vals[n] = fn(s)
        return np.stack([vals[o] for o in self.outputs], axis=1)


# --------------------------------------------------------------------------- population


@dataclass
class Species:
    id: int
    representative: Genome
    members: list = field(default_factory=list)
    best_fitness: float = -math.inf
    last_improved: int = 0
    created: int = 0


class Population:
    def __init__(self, cfg: NeatConfig, n_in: int, n_out: int, seed: int = 0,
                 out_act: str = "tanh", seed_topographic_inputs: tuple[int, int] | None = None):
        self.cfg = cfg
        self.n_in, self.n_out = n_in, n_out
        self.out_act = out_act
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.innov = Innovations(first_node=n_in + n_out)
        self.next_genome = 0
        self.generation = 0
        self.species: list[Species] = []
        self.next_species = 0
        self.threshold = cfg.compat_threshold
        self.best: Genome | None = None
        self.topo_inputs = seed_topographic_inputs if cfg.seed_topographic else None
        self.genomes = [self._initial_genome() for _ in range(cfg.population)]

    # ------------------------------------------------------------- creation

    def _new_id(self) -> int:
        self.next_genome += 1
        return self.next_genome

    def _initial_genome(self) -> Genome:
        cfg = self.cfg
        g = Genome(self._new_id())
        for i in range(self.n_in):
            g.nodes[i] = NodeGene(i, "input", "identity")
        outs = range(self.n_in, self.n_in + self.n_out)
        for o in outs:
            g.nodes[o] = NodeGene(o, "output", self.out_act, float(self.np_rng.normal(0.0, 0.5)))
        for i in range(self.n_in):
            for o in outs:
                self._add_conn(g, i, o, float(self.np_rng.normal(0.0, cfg.init_weight_std)))
        if self.topo_inputs is not None:
            # Seed a Gaussian of (z_post - z_pre): topographic projections, a prior taken from the brain where
            # connected areas keep their maps in register. Node id is fixed so every genome shares it.
            zi, zj = self.topo_inputs
            nid = self.n_in + self.n_out
            self.innov.next_node = max(self.innov.next_node, nid + 1)
            g.nodes[nid] = NodeGene(nid, "hidden", "gauss", 0.0)
            self._add_conn(g, zi, nid, -2.0)
            self._add_conn(g, zj, nid, 2.0)
            for o in list(outs)[:-1]:
                self._add_conn(g, nid, o, float(abs(self.np_rng.normal(1.0, 0.5))))
        return g

    def _add_conn(self, g: Genome, src: int, dst: int, w: float) -> None:
        innov = self.innov.conn(src, dst)
        g.conns[innov] = ConnGene(innov, src, dst, w, True)

    # ------------------------------------------------------------- mutation

    def _creates_cycle(self, g: Genome, src: int, dst: int) -> bool:
        if src == dst:
            return True
        out_edges: dict[int, list[int]] = {}
        for c in g.conns.values():
            out_edges.setdefault(c.src, []).append(c.dst)
        stack, seen = [dst], set()
        while stack:
            n = stack.pop()
            if n == src:
                return True
            if n in seen:
                continue
            seen.add(n)
            stack.extend(out_edges.get(n, []))
        return False

    def _mutate(self, g: Genome) -> None:
        cfg, r = self.cfg, self.rng
        if r.random() < cfg.p_mutate_weights:
            for c in g.conns.values():
                if r.random() < cfg.p_weight_replace:
                    c.w = r.gauss(0.0, cfg.init_weight_std)
                else:
                    c.w += r.gauss(0.0, cfg.weight_perturb_std)
                c.w = max(-cfg.weight_limit, min(cfg.weight_limit, c.w))
            for n in g.nodes.values():
                if n.kind != "input" and r.random() < 0.5:
                    n.bias = max(-cfg.weight_limit, min(cfg.weight_limit, n.bias + r.gauss(0.0, cfg.weight_perturb_std)))
        if r.random() < cfg.p_add_conn:
            self._mutate_add_conn(g)
        if r.random() < cfg.p_add_node:
            self._mutate_add_node(g)
        if r.random() < cfg.p_toggle_conn and g.conns:
            # Safe: cycle checks consider disabled connections too, so re-enabling one never closes a loop.
            c = r.choice(list(g.conns.values()))
            c.on = not c.on
        if r.random() < cfg.p_change_activation:
            hidden = [n for n in g.nodes.values() if n.kind == "hidden"]
            if hidden:
                r.choice(hidden).act = r.choice(cfg.hidden_activations)

    def _mutate_add_conn(self, g: Genome) -> None:
        srcs = [n.id for n in g.nodes.values() if n.kind != "output"]
        dsts = [n.id for n in g.nodes.values() if n.kind != "input"]
        existing = {(c.src, c.dst) for c in g.conns.values()}
        for _ in range(20):
            s, d = self.rng.choice(srcs), self.rng.choice(dsts)
            if (s, d) in existing or self._creates_cycle(g, s, d):
                continue
            self._add_conn(g, s, d, self.rng.gauss(0.0, self.cfg.init_weight_std))
            return

    def _mutate_add_node(self, g: Genome) -> None:
        enabled = [c for c in g.conns.values() if c.on]
        if not enabled:
            return
        c = self.rng.choice(enabled)
        c.on = False
        nid = self.innov.split(c.innov, g)
        g.nodes[nid] = NodeGene(nid, "hidden", self.rng.choice(self.cfg.hidden_activations), 0.0)
        self._add_conn(g, c.src, nid, 1.0)
        self._add_conn(g, nid, c.dst, c.w)

    # ------------------------------------------------------------ crossover

    def _crossover(self, a: Genome, b: Genome) -> Genome:
        if (b.fitness or 0.0) > (a.fitness or 0.0):
            a, b = b, a
        r = self.rng
        child = Genome(self._new_id())
        for innov, ca in a.conns.items():
            cb = b.conns.get(innov)
            gene = copy.copy(ca if cb is None or r.random() < 0.5 else cb)
            if cb is not None and (not ca.on or not cb.on):
                gene.on = r.random() > 0.75
            child.conns[innov] = gene
        for nid, na in a.nodes.items():
            nb = b.nodes.get(nid)
            child.nodes[nid] = copy.copy(na if nb is None or r.random() < 0.5 else nb)
        return child

    # ----------------------------------------------------------- speciation

    def distance(self, a: Genome, b: Genome) -> float:
        cfg = self.cfg
        dis, wsum = 0, 0.0
        for k, ca in a.conns.items():
            cb = b.conns.get(k)
            if cb is None:
                dis += 1
            else:
                wsum += abs(ca.w - cb.w) + (ca.on != cb.on)
        dis += sum(1 for k in b.conns if k not in a.conns)
        nc = max(len(a.conns), len(b.conns), 1)
        d = (cfg.compat_disjoint * dis + cfg.compat_weight * wsum) / nc
        ndis, nsum = 0, 0.0
        for k, na in a.nodes.items():
            if na.kind == "input":
                continue
            nb = b.nodes.get(k)
            if nb is None:
                ndis += 1
            else:
                nsum += abs(na.bias - nb.bias) + (na.act != nb.act)
        ndis += sum(1 for k, n in b.nodes.items() if n.kind != "input" and k not in a.nodes)
        nn = max(len(a.nodes), len(b.nodes), 1) - self.n_in
        return d + (cfg.compat_disjoint * ndis + cfg.compat_weight * nsum) / max(nn, 1)

    def speciate(self) -> None:
        for s in self.species:
            s.members = []
        for g in self.genomes:
            for s in self.species:
                if self.distance(g, s.representative) < self.threshold:
                    s.members.append(g)
                    break
            else:
                self.next_species += 1
                self.species.append(Species(self.next_species, g, [g], created=self.generation,
                                            last_improved=self.generation))
        self.species = [s for s in self.species if s.members]
        n = len(self.species)
        if n < self.cfg.target_species:
            self.threshold = max(0.3, self.threshold - self.cfg.compat_threshold_step)
        elif n > self.cfg.target_species:
            self.threshold += self.cfg.compat_threshold_step

    # --------------------------------------------------------- reproduction

    def evolve(self) -> None:
        """Produce the next generation from the evaluated current one (all genomes must have a fitness)."""
        cfg, r = self.cfg, self.rng
        for g in self.genomes:
            assert g.fitness is not None
            if self.best is None or g.fitness > self.best.fitness:
                self.best = copy.deepcopy(g)
        self.speciate()

        # Stagnation.
        for s in self.species:
            top = max(m.fitness for m in s.members)
            if top > s.best_fitness + 1e-6:
                s.best_fitness = top
                s.last_improved = self.generation
        ranked = sorted(self.species, key=lambda s: s.best_fitness, reverse=True)
        protected = {s.id for s in ranked[: cfg.species_elitism]}
        alive = [s for s in self.species
                 if s.id in protected or self.generation - s.last_improved < cfg.stagnation]

        # Offspring quotas from range-normalised mean fitness (explicit fitness sharing).
        fits = [g.fitness for s in alive for g in s.members]
        lo, hi = min(fits), max(fits)
        span = max(hi - lo, 1e-9)
        adj = np.array([np.mean([(m.fitness - lo) / span for m in s.members]) for s in alive]) + 1e-3
        quota = adj / adj.sum() * cfg.population
        sizes = np.maximum(np.round(quota).astype(int), cfg.min_species_size)
        while sizes.sum() > cfg.population:
            sizes[np.argmax(sizes)] -= 1
        while sizes.sum() < cfg.population:
            sizes[np.argmax(quota - sizes)] += 1

        children: list[Genome] = []
        for s, n in zip(alive, sizes):
            members = sorted(s.members, key=lambda g: g.fitness, reverse=True)
            for elite in members[: min(cfg.elitism, n)]:
                children.append(elite)  # unchanged, keeps its fitness
            pool = members[: max(2, int(math.ceil(cfg.survival_fraction * len(members))))]
            for _ in range(n - min(cfg.elitism, n)):
                if len(pool) > 1 and r.random() < cfg.p_crossover:
                    if r.random() < cfg.p_interspecies and len(alive) > 1:
                        other = r.choice(r.choice(alive).members)
                        child = self._crossover(r.choice(pool), other)
                    else:
                        p1, p2 = r.sample(pool, 2)
                        child = self._crossover(p1, p2)
                else:
                    child = copy.deepcopy(r.choice(pool))
                    child.id = self._new_id()
                self._mutate(child)
                child.fitness = None
                child.info = {}
                children.append(child)
            s.representative = r.choice(s.members)
        self.species = alive
        self.genomes = children
        self.generation += 1

    def compile(self, g: Genome) -> CPPN:
        return CPPN(g, self.n_in, self.n_out)
