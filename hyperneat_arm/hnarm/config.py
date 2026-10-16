"""All tunable settings, grouped by subsystem.

Every config is a plain dataclass so a run can be saved next to its checkpoints (``to_dict``) and rebuilt exactly
(``Config.from_dict``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """The body and the world: a two-link arm (a double pendulum) hanging over a floor with a ball on it."""

    dt: float = 0.001  # s, shared by physics and the spiking network (1 ms, the usual SNN step)
    duration: float = 3.0  # s per trial
    settle_window: float = 0.5  # s at the end of a trial over which the ball's final error is averaged
    gravity: float = 9.81

    # Arm. Angles: shoulder is measured from hanging straight down, counterclockwise positive, so positive is a
    # forward swing toward +x. The elbow is measured relative to the upper arm and, like a human elbow, only flexes
    # (bends forward).
    shoulder_height: float = 0.73  # m above the floor. Arm length + link radius + 5 mm, so the hand skims the floor.
    link_length: tuple = (0.35, 0.35)  # upper arm, forearm + hand (m)
    link_mass: tuple = (2.0, 1.5)  # kg, uniform rods
    link_radius: float = 0.025  # m, capsule radius used for collisions
    joint_damping: float = 0.05  # N*m*s/rad, passive joint friction
    shoulder_range: tuple = (-2.0, 2.0)  # rad
    elbow_range: tuple = (0.0, 2.6)  # rad
    limit_stiffness: float = 300.0  # N*m/rad, soft joint limits
    limit_damping: float = 3.0

    # Muscles: one flexor and one extensor per joint.
    max_torque: tuple = (20.0, 12.0)  # N*m at full activation (shoulder, elbow)
    muscle_tau: float = 0.04  # s, twitch / excitation-contraction time constant
    muscle_rate_ref: float = 40.0  # Hz of mean motor-pool firing that gives 63% activation
    muscle_viscosity: float = 0.5  # N*m*s/rad of extra joint damping per unit of total activation (co-contraction)

    # Ball.
    ball_radius: float = 0.1
    ball_mass: float = 0.4
    rolling_factor: float = 5.0 / 3.0  # 1 + I/(m r^2) for a hollow sphere rolling without slipping
    rolling_resistance: float = 0.15  # coefficient of rolling resistance, makes a pushed ball come to rest
    ball_air_drag: float = 0.05  # N*s/m
    wall_x: float = 0.9  # walls at +-wall_x keep the ball in the arena

    # Penalty contacts.
    floor_stiffness: float = 2.0e4
    floor_damping: float = 40.0
    contact_stiffness: float = 4000.0  # arm <-> ball
    contact_damping: float = 15.0
    contact_friction: float = 0.4

    # Task.
    target_tolerance: float = 0.05  # m, ball centre within this of the target counts as a success


@dataclass
class SensorConfig:
    """How the world is turned into input currents (the 'retina' and the proprioceptors)."""

    vision_nx: int = 7
    vision_ny: int = 3
    vision_x_range: tuple = (-0.6, 0.6)  # m, horizontal extent of the visual field
    vision_y_range: tuple = (0.1, 0.9)  # m, vertical extent of the visual field
    angle_cells: int = 5  # population-coded joint-angle afferents per joint
    input_gain: float = 14.0  # input current at full receptor activation
    input_offset: float = 1.0  # resting input current of every afferent
    velocity_scale: float = 2.0  # rad/s of muscle lengthening that half-saturates a spindle (Ia) afferent
    touch_scale: float = 5.0  # N of contact force that saturates a touch afferent


@dataclass
class BrainConfig:
    """The spiking network that HyperNEAT paints onto the brain substrate."""

    expression_threshold: float = 0.2  # CPPN outputs below this do not create a synapse
    # Peak synaptic weight of each tract before fan-in normalisation (weight = gain / sqrt(presynaptic population size)).
    tract_gain: dict = field(
        default_factory=lambda: {
            "afferent": 20.0,
            "feedforward": 10.0,
            "feedback": 8.0,
            "local_exc": 6.0,
            "local_inh": 16.0,
            "cerebellar": 8.0,
            "descending": 12.0,
            "spinal": 12.0,
        }
    )
    bias_gain: float = 4.0  # tonic current = bias_gain * CPPN bias output
    tau_exc: float = 5.0  # ms, AMPA-like decay of excitatory synaptic current
    tau_inh: float = 10.0  # ms, GABA_A-like decay of inhibitory synaptic current
    noise_std: float = 1.5  # std of background current noise (same noise for every genome, so fitness is fair)
    noise_bank: int = 251  # steps of precomputed noise, cycled
    max_current: float = 100.0


@dataclass
class NeatConfig:
    """NEAT settings for evolving the CPPN."""

    population: int = 150
    hidden_activations: tuple = ("tanh", "sigmoid", "sin", "gauss", "abs", "identity", "relu")
    seed_topographic: bool = True  # start every CPPN with a Gaussian of (z_post - z_pre): topographic maps
    init_weight_std: float = 1.0
    weight_limit: float = 5.0
    p_crossover: float = 0.6
    p_interspecies: float = 0.02
    p_mutate_weights: float = 0.8
    p_weight_replace: float = 0.1
    weight_perturb_std: float = 0.35
    p_add_conn: float = 0.15
    p_add_node: float = 0.06
    p_toggle_conn: float = 0.02
    p_change_activation: float = 0.05
    compat_disjoint: float = 1.0
    compat_weight: float = 0.5
    compat_threshold: float = 1.0  # starting value, adapted to hit target_species
    compat_threshold_step: float = 0.05
    target_species: int = 10
    survival_fraction: float = 0.3
    elitism: int = 1  # champions copied unchanged per species
    stagnation: int = 20  # generations without improvement before a species is dropped
    species_elitism: int = 2  # best species protected from stagnation
    min_species_size: int = 2


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    neat: NeatConfig = field(default_factory=NeatConfig)
    # Fitness weights.
    w_progress: float = 1.0
    w_precision: float = 1.0
    w_reach: float = 0.25
    w_effort: float = 0.2

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        def build(dc_type, values):
            kwargs = {}
            for f in dataclasses.fields(dc_type):
                if f.name not in values:
                    continue
                v = values[f.name]
                if isinstance(v, list):  # JSON turns tuples into lists
                    v = tuple(v)
                kwargs[f.name] = v
            return dc_type(**kwargs)

        top = {k: v for k, v in d.items() if k not in ("env", "sensors", "brain", "neat")}
        return cls(
            env=build(EnvConfig, d.get("env", {})),
            sensors=build(SensorConfig, d.get("sensors", {})),
            brain=build(BrainConfig, d.get("brain", {})),
            neat=build(NeatConfig, d.get("neat", {})),
            **top,
        )
