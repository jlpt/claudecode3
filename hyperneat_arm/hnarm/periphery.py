"""The peripheral nervous system: receptors that turn the world into afferent currents, and muscles that turn motor
neuron spikes into joint torque."""

from __future__ import annotations

import numpy as np

from .brain import MUSCLE_JOINT, MUSCLE_SIDE, BrainSubstrate
from .config import EnvConfig, SensorConfig
from .world import ArmBallWorld


class Receptors:
    """Computes the input current of every afferent neuron from the state of the world.

    * Vision: three retinotopic maps (ball, target, hand) of Gaussian receptive fields tiling the visual field.
    * Joint-angle afferents: Gaussian tuning curves tiling each joint's range (a population code).
    * Ia spindle afferents: one per muscle, driven by how stretched the muscle is and how fast it is lengthening.
    * Touch afferents: contact force on the upper arm and on the forearm/hand.
    """

    def __init__(self, sub: BrainSubstrate, sensors: SensorConfig, env: EnvConfig):
        self.cfg = sensors
        self.n = sub.n_inputs
        P = sub.populations
        self.vis = {k: (P[f"VIS.{k}"].slice, P[f"VIS.{k}"].meta) for k in ("ball", "target", "hand")}
        dx = (sensors.vision_x_range[1] - sensors.vision_x_range[0]) / max(sensors.vision_nx - 1, 1)
        dy = (sensors.vision_y_range[1] - sensors.vision_y_range[0]) / max(sensors.vision_ny - 1, 1)
        self.sig_x, self.sig_y = 0.6 * dx, 0.6 * dy

        ang = P["DRG.angle"]
        self.ang_slice = ang.slice
        lo = np.array([env.shoulder_range[0], env.elbow_range[0]])
        hi = np.array([env.shoulder_range[1], env.elbow_range[1]])
        joint = ang.meta["joint"]
        self.ang_joint = joint
        self.ang_pref = lo[joint] + (ang.meta["pref"] + 1.0) / 2.0 * (hi - lo)[joint]
        self.ang_sigma = (hi - lo)[joint] / sensors.angle_cells * 0.6
        self.lo, self.hi = lo, hi
        self.ia_slice = P["DRG.Ia"].slice
        self.touch_slice = P["DRG.touch"].slice
        self.ball_radius = env.ball_radius

    def _vision(self, out, key, x, y):
        sl, meta = self.vis[key]
        ex = (x[:, None] - meta["rf_x"][None]) / self.sig_x
        ey = (y[:, None] - meta["rf_y"][None]) / self.sig_y
        out[:, sl] = np.exp(-0.5 * (ex * ex + ey * ey))

    def activations(self, world: ArmBallWorld) -> np.ndarray:
        """Receptor activation in [0, 1] for every afferent, (B, n_inputs)."""
        act = np.zeros((world.B, self.n))
        _, hand = world.joints()
        self._vision(act, "ball", world.ball[:, 0], world.ball[:, 1])
        self._vision(act, "target", world.target_x, np.full(world.B, self.ball_radius))
        self._vision(act, "hand", hand[:, 0], hand[:, 1])

        q = world.q[:, self.ang_joint]
        e = (q - self.ang_pref[None]) / self.ang_sigma[None]
        act[:, self.ang_slice] = np.exp(-0.5 * e * e)

        # Muscle length in [0, 1]: a flexor is longest with the joint fully extended, an extensor the other way round.
        flex_len = (self.hi - world.q) / (self.hi - self.lo)
        ia = np.empty((world.B, 4))
        for m in range(4):
            j = MUSCLE_JOINT[m]
            length = flex_len[:, j] if MUSCLE_SIDE[m] > 0 else 1.0 - flex_len[:, j]
            lengthening = -MUSCLE_SIDE[m] * world.qd[:, j]
            ia[:, m] = 0.25 * np.clip(length, 0.0, 1.0) + 0.75 * np.tanh(np.maximum(lengthening, 0.0) / self.cfg.velocity_scale)
        act[:, self.ia_slice] = ia
        act[:, self.touch_slice] = np.tanh(world.touch / self.cfg.touch_scale)
        return act

    def currents(self, world: ArmBallWorld) -> np.ndarray:
        return self.cfg.input_offset + self.cfg.input_gain * self.activations(world)


class Muscles:
    """Agonist / antagonist muscle pairs driven by motor neuron pools.

    Each muscle low-pass filters the mean firing rate of its pool (twitch summation) and saturates. Joint torque is
    flexor minus extensor force times the joint's strength, and co-contraction stiffens the joint by adding damping.
    """

    def __init__(self, sub: BrainSubstrate, env: EnvConfig, batch: int):
        self.cfg = env
        self.motor_slice = sub.motor_slice
        pool = np.zeros((sub.motor_muscle.size, 4), dtype=np.float32)
        pool[np.arange(sub.motor_muscle.size), sub.motor_muscle] = 1.0
        self.pool = pool / pool.sum(0, keepdims=True)  # spikes -> mean spikes per neuron of each pool
        self.rate = np.zeros((batch, 4))
        self.alpha = env.dt / env.muscle_tau
        self.max_torque = np.array(env.max_torque)

    def reset(self) -> None:
        self.rate[:] = 0.0

    def update(self, motor_spikes: np.ndarray) -> np.ndarray:
        """motor_spikes (B, n_motor) of this step -> muscle activation (B, 4) in [0, 1)."""
        inst = (motor_spikes @ self.pool) / self.cfg.dt  # Hz
        self.rate += self.alpha * (inst - self.rate)
        return 1.0 - np.exp(-self.rate / self.cfg.muscle_rate_ref)

    def torque(self, act: np.ndarray, qd: np.ndarray) -> np.ndarray:
        flex = act[:, [0, 2]]
        ext = act[:, [1, 3]]
        return self.max_torque * (flex - ext) - self.cfg.muscle_viscosity * (flex + ext) * qd
