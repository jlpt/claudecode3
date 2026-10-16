"""Saving and loading genomes (JSON) and whole runs (pickle checkpoints)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from .config import Config
from .neat import ConnGene, Genome, NodeGene


def genome_to_dict(g: Genome) -> dict:
    return {
        "id": g.id,
        "fitness": g.fitness,
        "info": g.info,
        "nodes": [[n.id, n.kind, n.act, n.bias] for n in g.nodes.values()],
        "conns": [[c.innov, c.src, c.dst, c.w, c.on] for c in g.conns.values()],
    }


def genome_from_dict(d: dict) -> Genome:
    g = Genome(d["id"], fitness=d.get("fitness"), info=d.get("info", {}))
    for nid, kind, act, bias in d["nodes"]:
        g.nodes[nid] = NodeGene(nid, kind, act, bias)
    for innov, src, dst, w, on in d["conns"]:
        g.conns[innov] = ConnGene(innov, src, dst, w, on)
    return g


def save_champion(path: Path, genome: Genome, cfg: Config, generation: int) -> None:
    data = {"generation": generation, "config": cfg.to_dict(), "genome": genome_to_dict(genome)}
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def load_champion(path: Path) -> tuple[Genome, Config, int]:
    data = json.loads(Path(path).read_text())
    return genome_from_dict(data["genome"]), Config.from_dict(data["config"]), data.get("generation", 0)


def save_checkpoint(path: Path, population, cfg: Config) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"population": population, "config": cfg.to_dict()}, f)
    tmp.replace(path)


def load_checkpoint(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["population"], Config.from_dict(data["config"])
