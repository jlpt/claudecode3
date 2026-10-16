import dataclasses
import unittest

import numpy as np

from hnarm.config import EnvConfig
from hnarm.world import ArmBallWorld


class PhysicsTest(unittest.TestCase):
    def test_passive_double_pendulum_conserves_energy(self):
        # No friction, no limits, no contacts: a chaotic double pendulum must keep its energy. A wrong mass matrix,
        # Coriolis or gravity term shows up as drift.
        env = dataclasses.replace(EnvConfig(), joint_damping=0.0, dt=1e-4, shoulder_range=(-50, 50), elbow_range=(-50, 50))
        w = ArmBallWorld(env, 3)
        w.reset(0.5, 0.0, q0=np.array([[1.0, 0.5], [2.0, 1.0], [0.3, 2.5]]))
        e0 = w.energy().copy()
        for _ in range(10000):
            w.step(np.zeros((3, 2)), contacts=False)
        self.assertLess(np.abs(w.energy() - e0).max(), 0.05)
        self.assertGreater(np.abs(w.q - [[1.0, 0.5], [2.0, 1.0], [0.3, 2.5]]).max(), 0.5)  # it really moved

    def test_hanging_arm_and_resting_ball_stay_put(self):
        w = ArmBallWorld(EnvConfig(), 2)
        w.reset(np.array([-0.3, 0.3]), 0.0)
        for _ in range(1000):
            w.step(np.zeros((2, 2)))
        np.testing.assert_allclose(w.q, 0.0, atol=1e-6)
        np.testing.assert_allclose(w.ball[:, 0], [-0.3, 0.3], atol=1e-6)
        np.testing.assert_allclose(w.ball[:, 1], EnvConfig().ball_radius, atol=1e-3)

    def test_released_arm_swings_into_ball_and_pushes_it(self):
        # Raised backward and let go, the arm falls forward under gravity and knocks the ball toward +x.
        w = ArmBallWorld(EnvConfig(), 1)
        w.reset(0.2, 0.0, q0=np.array([[-1.2, 0.0]]))
        touched = False
        for _ in range(1500):
            w.step(np.zeros((1, 2)))
            touched |= bool((w.touch > 0).any())
        self.assertTrue(touched)
        self.assertGreater(w.ball[0, 0], 0.3)

    def test_rolling_resistance_brings_ball_to_rest(self):
        w = ArmBallWorld(EnvConfig(), 1)
        w.reset(-0.8, 0.0)  # the arm hangs still at x = 0, out of the ball's way
        w.ball_v[0, 0] = 0.6
        for _ in range(3000):
            w.step(np.zeros((1, 2)))
        self.assertLess(abs(w.ball_v[0, 0]), 0.01)
        self.assertGreater(w.ball[0, 0], -0.7)  # it rolled ...
        self.assertLess(w.ball[0, 0], -0.4)  # ... and stopped well short of the arm

    def test_elbow_cannot_hyperextend(self):
        w = ArmBallWorld(EnvConfig(), 1)
        w.reset(0.6, 0.0, q0=np.array([[1.0, 1.0]]))
        for _ in range(2000):
            w.step(np.array([[0.0, -12.0]]))  # full extensor torque at the elbow
        self.assertGreater(w.q[0, 1], -0.06)


if __name__ == "__main__":
    unittest.main()
