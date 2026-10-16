"""Batched physics of a two-link arm (a driven double pendulum) that pushes a ball along the floor.

Side view, gravity along -y, floor at y = 0. The shoulder is a fixed pivot above the floor. Each link is a uniform
rod, so the arm follows the compound double pendulum equations

    M(q) q'' + C(q, q') + G(q) = tau

where q = (shoulder, elbow). The ball is a rolling sphere that touches the floor, two walls and the two arm links
(capsules). All contacts are penalty springs with damping and smoothed Coulomb friction. Every array has a leading
batch dimension B so many trials and many genomes are simulated in one set of numpy calls.

Integration is semi-implicit (symplectic) Euler at the network's 1 ms step.
"""

from __future__ import annotations

import numpy as np

from .config import EnvConfig


class ArmBallWorld:
    def __init__(self, cfg: EnvConfig, batch: int):
        self.cfg = cfg
        self.B = batch
        l1, l2 = cfg.link_length
        m1, m2 = cfg.link_mass
        self.l1, self.l2, self.m1, self.m2 = l1, l2, m1, m2
        lc1, lc2 = l1 / 2.0, l2 / 2.0
        i1, i2 = m1 * l1 * l1 / 12.0, m2 * l2 * l2 / 12.0
        # Constant parts of the mass matrix.
        self._ma = i1 + m1 * lc1 * lc1 + i2 + m2 * (l1 * l1 + lc2 * lc2)
        self._mb = m2 * l1 * lc2
        self._md = i2 + m2 * lc2 * lc2
        self._g1 = (m1 * lc1 + m2 * l1) * cfg.gravity
        self._g2 = m2 * lc2 * cfg.gravity
        self.shoulder = np.array([0.0, cfg.shoulder_height])
        self.lo = np.array([cfg.shoulder_range[0], cfg.elbow_range[0]])
        self.hi = np.array([cfg.shoulder_range[1], cfg.elbow_range[1]])

        self.q = np.zeros((batch, 2))
        self.qd = np.zeros((batch, 2))
        self.ball = np.zeros((batch, 2))
        self.ball_v = np.zeros((batch, 2))
        self.target_x = np.zeros(batch)
        # Magnitude of the contact force on each link from the last step, read by the touch sensors.
        self.touch = np.zeros((batch, 2))
        # Distance between the ball's surface and each link's surface (negative while touching).
        self.gaps = np.zeros((batch, 2))
        self.time = 0.0

    # ------------------------------------------------------------------ setup

    def reset(self, ball_x, target_x, q0=None) -> None:
        self.q[:] = 0.0 if q0 is None else q0
        self.qd[:] = 0.0
        self.ball[:, 0] = ball_x
        self.ball[:, 1] = self.cfg.ball_radius
        self.ball_v[:] = 0.0
        self.target_x[:] = target_x
        self.touch[:] = 0.0
        self.time = 0.0

    # ------------------------------------------------------------ kinematics

    def _angles(self):
        # phi = absolute direction of each link, measured from +x. Hanging straight down is -pi/2.
        phi1 = self.q[:, 0] - np.pi / 2.0
        phi12 = phi1 + self.q[:, 1]
        return phi1, phi12

    def joints(self):
        """Elbow and hand positions, both (B, 2)."""
        phi1, phi12 = self._angles()
        elbow = np.empty((self.B, 2))
        elbow[:, 0] = self.shoulder[0] + self.l1 * np.cos(phi1)
        elbow[:, 1] = self.shoulder[1] + self.l1 * np.sin(phi1)
        hand = np.empty((self.B, 2))
        hand[:, 0] = elbow[:, 0] + self.l2 * np.cos(phi12)
        hand[:, 1] = elbow[:, 1] + self.l2 * np.sin(phi12)
        return elbow, hand

    def mass_matrix(self):
        c2 = np.cos(self.q[:, 1])
        m11 = self._ma + 2.0 * self._mb * c2
        m12 = self._md + self._mb * c2
        return m11, m12, np.full_like(m11, self._md)

    def energy(self):
        """Kinetic + potential energy of the arm (J), used to validate the equations of motion."""
        m11, m12, m22 = self.mass_matrix()
        w1, w2 = self.qd[:, 0], self.qd[:, 1]
        ke = 0.5 * (m11 * w1 * w1 + 2.0 * m12 * w1 * w2 + m22 * w2 * w2)
        phi1, phi12 = self._angles()
        pe = self._g1 * np.sin(phi1) + self._g2 * np.sin(phi12)
        return ke + pe

    # --------------------------------------------------------------- contacts

    @staticmethod
    def _point_jacobian(seg, s, phi1, phi12, l1):
        """Columns of the 2x2 Jacobian of a point at distance s along link `seg` (0 upper arm, 1 forearm)."""
        if seg == 0:
            j1 = np.stack([-s * np.sin(phi1), s * np.cos(phi1)], axis=1)
            j2 = np.zeros_like(j1)
        else:
            j2 = np.stack([-s * np.sin(phi12), s * np.cos(phi12)], axis=1)
            j1 = np.stack([-l1 * np.sin(phi1), l1 * np.cos(phi1)], axis=1) + j2
        return j1, j2

    def _ball_link_contact(self, seg, a, b, phi1, phi12, tau, f_ball):
        cfg = self.cfg
        ab = b - a
        t = np.clip(((self.ball - a) * ab).sum(1) / (ab * ab).sum(1), 0.0, 1.0)
        c = a + t[:, None] * ab
        d = self.ball - c
        dist = np.sqrt((d * d).sum(1)) + 1e-9
        n = d / dist[:, None]
        pen = cfg.ball_radius + cfg.link_radius - dist
        self.gaps[:, seg] = -pen
        hit = pen > 0.0
        if not hit.any():
            self.touch[:, seg] = 0.0
            return
        s = t * (self.l1 if seg == 0 else self.l2)
        j1, j2 = self._point_jacobian(seg, s, phi1, phi12, self.l1)
        v_point = j1 * self.qd[:, 0:1] + j2 * self.qd[:, 1:2]
        v_rel = self.ball_v - v_point
        vn = (v_rel * n).sum(1)
        fn = np.where(hit, np.maximum(cfg.contact_stiffness * pen - cfg.contact_damping * vn, 0.0), 0.0)
        tang = np.stack([-n[:, 1], n[:, 0]], axis=1)
        vt = (v_rel * tang).sum(1)
        ft = -cfg.contact_friction * fn * np.tanh(vt / 0.02)
        f = fn[:, None] * n + ft[:, None] * tang
        f_ball += f
        # Equal and opposite force on the arm, mapped to joint torques by the Jacobian transpose.
        tau[:, 0] -= (j1 * f).sum(1)
        tau[:, 1] -= (j2 * f).sum(1)
        self.touch[:, seg] = np.sqrt((f * f).sum(1))

    def _arm_floor_contact(self, phi1, phi12, elbow, hand, tau):
        """Keeps the elbow and hand above the floor. Inactive with the default shoulder height."""
        cfg = self.cfg
        for seg, p, s in ((0, elbow, self.l1), (1, hand, self.l2)):
            pen = cfg.link_radius - p[:, 1]
            hit = pen > 0.0
            if not hit.any():
                continue
            j1, j2 = self._point_jacobian(seg, np.full(self.B, s), phi1, phi12, self.l1)
            v = j1 * self.qd[:, 0:1] + j2 * self.qd[:, 1:2]
            fy = np.where(hit, np.maximum(cfg.floor_stiffness * pen - cfg.floor_damping * v[:, 1], 0.0), 0.0)
            fx = -cfg.contact_friction * fy * np.tanh(v[:, 0] / 0.02)
            tau[:, 0] += j1[:, 0] * fx + j1[:, 1] * fy
            tau[:, 1] += j2[:, 0] * fx + j2[:, 1] * fy

    # ------------------------------------------------------------------- step

    def step(self, torque: np.ndarray, contacts: bool = True) -> None:
        """Advance one dt. `torque` (B, 2) is the muscle torque at the shoulder and elbow."""
        cfg = self.cfg
        dt = cfg.dt
        phi1, phi12 = self._angles()
        elbow, hand = self.joints()

        tau = torque - cfg.joint_damping * self.qd
        # Soft joint limits.
        below = self.lo - self.q
        above = self.q - self.hi
        tau += np.where(below > 0.0, cfg.limit_stiffness * below - cfg.limit_damping * np.minimum(self.qd, 0.0), 0.0)
        tau -= np.where(above > 0.0, cfg.limit_stiffness * above + cfg.limit_damping * np.maximum(self.qd, 0.0), 0.0)

        f_ball = np.zeros((self.B, 2))
        if contacts:
            shoulder = np.broadcast_to(self.shoulder, (self.B, 2))
            self._ball_link_contact(0, shoulder, elbow, phi1, phi12, tau, f_ball)
            self._ball_link_contact(1, elbow, hand, phi1, phi12, tau, f_ball)
            self._arm_floor_contact(phi1, phi12, elbow, hand, tau)

        # Arm dynamics: M q'' = tau - C - G.
        s2 = np.sin(self.q[:, 1])
        w1, w2 = self.qd[:, 0], self.qd[:, 1]
        h = self._mb * s2
        r1 = tau[:, 0] + h * (2.0 * w1 * w2 + w2 * w2) - self._g1 * np.cos(phi1) - self._g2 * np.cos(phi12)
        r2 = tau[:, 1] - h * w1 * w1 - self._g2 * np.cos(phi12)
        m11, m12, m22 = self.mass_matrix()
        det = m11 * m22 - m12 * m12
        qdd1 = (m22 * r1 - m12 * r2) / det
        qdd2 = (m11 * r2 - m12 * r1) / det
        self.qd[:, 0] += qdd1 * dt
        self.qd[:, 1] += qdd2 * dt
        self.q += self.qd * dt

        if contacts:
            self._ball_step(f_ball)
        self.time += dt

    def _ball_step(self, f: np.ndarray) -> None:
        cfg = self.cfg
        r = cfg.ball_radius
        x, y = self.ball[:, 0], self.ball[:, 1]
        vx, vy = self.ball_v[:, 0], self.ball_v[:, 1]
        fx = f[:, 0] - cfg.ball_air_drag * vx
        fy = f[:, 1] - cfg.ball_air_drag * vy - cfg.ball_mass * cfg.gravity

        pen = r - y
        on_floor = pen > 0.0
        normal = np.where(on_floor, np.maximum(cfg.floor_stiffness * pen - cfg.floor_damping * vy, 0.0), 0.0)
        fy = fy + normal
        fx = fx - cfg.rolling_resistance * normal * np.tanh(vx / 0.01)

        pen_r = x + r - cfg.wall_x
        pen_l = -cfg.wall_x - (x - r)
        fx = fx - np.where(pen_r > 0.0, np.maximum(cfg.floor_stiffness * pen_r + cfg.floor_damping * vx, 0.0), 0.0)
        fx = fx + np.where(pen_l > 0.0, np.maximum(cfg.floor_stiffness * pen_l - cfg.floor_damping * vx, 0.0), 0.0)

        # Rolling without slipping makes the ball heavier to push sideways while it is on the floor.
        mx = cfg.ball_mass * np.where(pen > -0.002, cfg.rolling_factor, 1.0)
        self.ball_v[:, 0] += fx / mx * cfg.dt
        self.ball_v[:, 1] += fy / cfg.ball_mass * cfg.dt
        self.ball += self.ball_v * cfg.dt
