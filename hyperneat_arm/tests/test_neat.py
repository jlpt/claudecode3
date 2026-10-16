import dataclasses
import random
import unittest

import numpy as np

from hnarm.config import NeatConfig
from hnarm.neat import CPPN, Population

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float64)
XOR_Y = np.array([0, 1, 1, 0], np.float64)


class NeatTest(unittest.TestCase):
    def test_mutations_keep_the_cppn_acyclic(self):
        cfg = dataclasses.replace(NeatConfig(), population=10, p_add_node=0.5, p_add_conn=0.9, p_toggle_conn=0.3)
        pop = Population(cfg, 4, 2, seed=1)
        g = pop.genomes[0]
        for _ in range(300):
            pop._mutate(g)
        self.assertGreater(g.size()[0], 20)
        out = CPPN(g, 4, 2)(np.random.default_rng(0).normal(size=(50, 4)))
        self.assertEqual(out.shape, (50, 2))
        self.assertTrue(np.isfinite(out).all())

    def test_crossover_of_divergent_parents_compiles(self):
        cfg = dataclasses.replace(NeatConfig(), population=10, p_add_node=0.4, p_add_conn=0.8)
        pop = Population(cfg, 3, 1, seed=2)
        a, b = pop.genomes[0], pop.genomes[1]
        for _ in range(100):
            pop._mutate(a)
            pop._mutate(b)
        a.fitness, b.fitness = 1.0, 0.5
        child = pop._crossover(a, b)
        self.assertTrue(set(child.conns) <= set(a.conns))  # structure comes from the fitter parent
        CPPN(child, 3, 1)(np.zeros((4, 3)))

    def test_solves_xor(self):
        # NEAT's classic sanity check: XOR needs a hidden node, so it only works if structural mutation does.
        random.seed(0)
        cfg = dataclasses.replace(NeatConfig(), population=150, seed_topographic=False, target_species=8)
        pop = Population(cfg, 3, 1, seed=4, out_act="sigmoid")  # third input is a constant bias
        X = np.hstack([XOR_X, np.ones((4, 1))])
        solved = False
        for _ in range(200):
            for g in pop.genomes:
                y = CPPN(g, 3, 1)(X)[:, 0]
                g.fitness = 4.0 - float(((y - XOR_Y) ** 2).sum())
            best = max(pop.genomes, key=lambda g: g.fitness)
            if all(abs(v - t) < 0.4 for v, t in zip(CPPN(best, 3, 1)(X)[:, 0], XOR_Y)):
                solved = True
                break
            pop.evolve()
        self.assertTrue(solved, f"best XOR fitness {best.fitness:.3f} after {pop.generation} generations")


if __name__ == "__main__":
    unittest.main()
