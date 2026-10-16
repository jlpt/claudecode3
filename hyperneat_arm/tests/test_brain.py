import unittest

import numpy as np

from hnarm.brain import N_CPPN_INPUTS, N_CPPN_OUTPUTS, BrainSubstrate
from hnarm.config import BrainConfig, Config
from hnarm.hyperneat import build_brain
from hnarm.neat import Population
from hnarm.snn import SpikingNetwork


def single_neuron_rate(sub, cell_key, current, ms=1000):
    """Firing rate of the neurons of one population driven by a constant current, with no synapses or noise."""
    cfg = BrainConfig(noise_std=0.0)
    net = SpikingNetwork(sub, cfg, np.zeros((1, sub.N, sub.n_post), np.float32), np.zeros((1, sub.n_post)), 1)
    pop = sub.populations[cell_key]
    count = 0
    for _ in range(ms):
        # Afferents get their current as input; for other neurons inject it through the tonic bias.
        net.bias[:] = current
        spikes = net.step(np.full((1, sub.n_inputs), current, np.float32))
        count += spikes[0, 0, pop.slice].sum()
    return count / pop.n / (ms / 1000)


class SubstrateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sub = BrainSubstrate()

    def test_connectome_respects_anatomy(self):
        sub = self.sub
        self.assertTrue((sub.post >= sub.n_inputs).all(), "afferents receive no synapses")
        self.assertFalse((sub.pre == sub.post).any(), "no autapses")
        region = np.array(sub.region_keys)[sub.region_of]
        inh = sub.inhibitory[sub.pre]
        # Inhibitory cells reach only their own region, except spinal interneurons onto motor neurons (same region SC).
        self.assertTrue((region[sub.pre][inh] == region[sub.post][inh]).all())
        # Motor neurons are the only cells that drive muscles, and M1 reaches them directly (corticomotoneuronal).
        m1_l5 = sub.populations["M1.L5"].slice
        mn = sub.populations["SC.MN"].slice
        self.assertTrue(((sub.pre >= m1_l5.start) & (sub.pre < m1_l5.stop) & (sub.post >= mn.start) & (sub.post < mn.stop)).any())

    def test_somatotopy_is_shared(self):
        # The shoulder sits medial (z < 0) and the elbow lateral (z > 0) in the afferents, cortex and spinal cord.
        P = self.sub.populations
        mn = P["SC.MN"]
        z = mn.coords[:, 0]
        self.assertTrue((z[mn.meta["muscle"] < 2] < 0).all() and (z[mn.meta["muscle"] >= 2] > 0).all())
        ang = P["DRG.angle"]
        self.assertTrue((ang.coords[ang.meta["joint"] == 0, 0] < 0).all())

    def test_dales_principle_and_connectome_mask(self):
        sub = self.sub
        pop = Population(Config().neat, N_CPPN_INPUTS, N_CPPN_OUTPUTS, seed=3, seed_topographic_inputs=(0, 3))
        allowed = np.zeros((sub.N, sub.n_post), bool)
        allowed[sub.pre, sub.post - sub.n_inputs] = True
        for g in pop.genomes[:10]:
            w, b, expressed = build_brain(g, sub, BrainConfig())
            self.assertFalse(w[~allowed].any(), "synapse outside the connectome")
            self.assertTrue((w[sub.inhibitory] <= 0).all(), "inhibitory cells must only inhibit")
            self.assertTrue((w[~sub.inhibitory] >= 0).all(), "excitatory cells must only excite")
            self.assertTrue(0 < expressed.sum() < len(sub.pre))
            self.assertEqual(b.shape, (sub.n_post,))

    def test_cell_types_fire_like_their_models(self):
        sub = self.sub
        rs = single_neuron_rate(sub, "M1.L23", 10.0)
        fs = single_neuron_rate(sub, "M1.INT", 10.0)
        self.assertGreater(rs, 5)
        self.assertLess(rs, 60)
        self.assertGreater(fs, 1.5 * rs, "fast-spiking interneurons outpace regular-spiking pyramidal cells")
        self.assertEqual(single_neuron_rate(sub, "M1.L23", 2.0), 0.0, "below rheobase stays silent")

    def test_trial_noise_does_not_depend_on_the_batch(self):
        sub = self.sub
        cfg = BrainConfig()
        z = np.zeros((1, sub.N, sub.n_post), np.float32), np.zeros((1, sub.n_post))
        a = SpikingNetwork(sub, cfg, *z, 2, noise_seeds=[(5, 0, 0), (5, 0, 1)])
        b = SpikingNetwork(sub, cfg, *z, 3, noise_seeds=[(9, 9, 9), (5, 0, 1), (5, 0, 0)])
        np.testing.assert_array_equal(a.noise[:, 0], b.noise[:, 2])
        np.testing.assert_array_equal(a.noise[:, 1], b.noise[:, 1])


if __name__ == "__main__":
    unittest.main()
