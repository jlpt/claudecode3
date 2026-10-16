"""The HyperNEAT substrate: a small spiking sensorimotor brain laid out like the real one.

What is "brain-like" about it:

* Regions and wiring follow the primate reaching circuit: vision -> posterior parietal cortex (PPC) -> dorsal premotor
  cortex (PMd) -> primary motor cortex (M1) -> spinal cord -> muscles, with proprioception entering through the
  dorsal root ganglia (DRG) to the spinal cord, cerebellum (CB) and primary somatosensory cortex (S1), and the
  cerebellum looping back to the motor cortex through the thalamus.
* Each cortical area has layers: layer 2/3 pyramidal cells (cortico-cortical), layer 5 pyramidal cells (output to
  the cerebellum and spinal cord, and feedback) and fast-spiking interneurons. The cerebellum has granule cells,
  Purkinje cells and deep cerebellar nuclei. The spinal cord has Ia inhibitory interneurons, Renshaw cells,
  excitatory interneurons and alpha motor neuron pools for a flexor and an extensor of each joint.
* Dale's principle: every neuron is excitatory or inhibitory and all its synapses share that sign. Only local
  circuits (interneurons, Purkinje cells, spinal interneurons) are inhibitory, as in the brain.
* Topographic maps share one axis, ``z``: the somatotopic body map (shoulder medial, elbow lateral) in S1, M1, the
  cerebellum, the spinal cord and the afferents, and the horizontal retinotopic axis in vision. HyperNEAT can exploit
  that, for example with one Gaussian of (z_post - z_pre) that wires matching body parts together.
* Different cell types use different Izhikevich spiking dynamics (regular spiking, bursting, fast spiking, ...).

Coordinates. Every neuron has local coordinates (z, a, b) in [-1, 1]:
  z  topographic position (somatotopy or horizontal retinotopy)
  a  secondary map (retinotopic elevation, preferred joint angle, flexor (+1) / extensor (-1) side, column)
  b  laminar depth or dorsoventral lamina (superficial / dorsal = +1)
and its region sits at an anatomical position (x posterior->anterior, y ventral->dorsal) in a normalised sagittal
frame. The CPPN is queried with (z, a, b) of both neurons plus the displacement between their regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import BrainConfig, SensorConfig

TRACTS = (
    "afferent",  # sensory afferents and thalamic relays into cortex
    "feedforward",  # cortico-cortical, layer 2/3 of a lower area to a higher one
    "feedback",  # cortico-cortical, layer 5 of a higher area back to layer 2/3 of a lower one
    "local_exc",  # recurrent excitation inside an area (pyramidal -> pyramidal and interneurons)
    "local_inh",  # interneurons inside an area
    "cerebellar",  # mossy fibres, parallel fibres, Purkinje -> nuclei, cerebello-thalamo-cortical
    "descending",  # corticospinal
    "spinal",  # reflex arcs and spinal interneuron circuits
)
TRACT_INDEX = {t: i for i, t in enumerate(TRACTS)}
BIAS_OUTPUT = len(TRACTS)  # CPPN output used for each neuron's tonic current
N_CPPN_INPUTS = 8  # z1 a1 b1 z2 a2 b2 dx dy
N_CPPN_OUTPUTS = len(TRACTS) + 1
CPPN_INPUT_NAMES = ("z_pre", "a_pre", "b_pre", "z_post", "a_post", "b_post", "dx", "dy")
CPPN_OUTPUT_NAMES = tuple(f"w_{t}" for t in TRACTS) + ("bias",)

# Izhikevich (2003) parameters (a, b, c, d).
IZHIKEVICH = {
    "RS": (0.02, 0.2, -65.0, 8.0),  # regular spiking: pyramidal cells
    "IB": (0.02, 0.2, -55.0, 4.0),  # intrinsically bursting: layer 5 pyramidal cells
    "FS": (0.1, 0.2, -65.0, 2.0),  # fast spiking: basket cells, Purkinje cells, spinal interneurons
    "TC": (0.02, 0.25, -65.0, 0.05),  # thalamocortical-like rebound bursting: deep cerebellar nuclei
    "TONIC": (0.02, 0.2, -65.0, 2.0),  # weakly adapting tonic firing: afferents and motor neurons
}

JOINTS = ("shoulder", "elbow")
JOINT_Z = (-0.5, 0.5)  # somatotopy: shoulder medial, elbow lateral
MUSCLES = ("shoulder flexor", "shoulder extensor", "elbow flexor", "elbow extensor")
MUSCLE_JOINT = (0, 0, 1, 1)
MUSCLE_SIDE = (1.0, -1.0, 1.0, -1.0)  # +1 flexor, -1 extensor


@dataclass
class Region:
    key: str
    name: str
    center: tuple  # (x, y) in the sagittal brain frame
    role: str


@dataclass
class Population:
    key: str
    region: str
    label: str
    inhibitory: bool
    cell: str
    coords: np.ndarray  # (n, 3) local (z, a, b)
    meta: dict = field(default_factory=dict)  # per-neuron arrays used by sensors / motors
    start: int = 0

    @property
    def n(self) -> int:
        return len(self.coords)

    @property
    def slice(self) -> slice:
        return slice(self.start, self.start + self.n)


@dataclass
class Projection:
    src: str
    dst: str
    tract: str
    note: str


def _centres(n: int, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    return lo + (np.arange(n) + 0.5) * (hi - lo) / n


def _sheet(nz: int, na: int, b: float) -> np.ndarray:
    """A small cortical sheet: nz positions along the topographic axis times na columns, at laminar depth b."""
    z, a = np.meshgrid(_centres(nz), _centres(na, -0.5, 0.5), indexing="ij")
    return np.stack([z.ravel(), a.ravel(), np.full(z.size, b)], axis=1)


REGIONS = (
    Region("VIS", "Visual cortex (dorsal stream input)", (-0.95, 0.30), "Retinotopic maps of the ball, target and hand"),
    Region("DRG", "Dorsal root ganglia", (-0.45, -0.95), "Proprioceptive and touch afferents from the arm"),
    Region("S1", "Primary somatosensory cortex", (-0.15, 0.90), "Body sense: where the arm is and what it touches"),
    Region("PPC", "Posterior parietal cortex", (-0.50, 0.75), "Combines vision and body sense into a reach plan"),
    Region("PMd", "Dorsal premotor cortex", (0.50, 0.80), "Selects and prepares the movement"),
    Region("M1", "Primary motor cortex", (0.15, 0.90), "Somatotopic motor commands"),
    Region("CB", "Cerebellum", (-0.65, -0.35), "Predicts and corrects movement from efference copy and proprioception"),
    Region("SC", "Spinal cord (cervical enlargement)", (-0.15, -0.90), "Reflexes, interneurons and motor neurons"),
)


class BrainSubstrate:
    """Neurons, their coordinates and cell types, the connectome, and the CPPN query for every allowed synapse."""

    def __init__(self, sensors: SensorConfig | None = None, brain: BrainConfig | None = None):
        self.sensor_cfg = sensors or SensorConfig()
        self.brain_cfg = brain or BrainConfig()
        self.regions = {r.key: r for r in REGIONS}
        self.populations: dict[str, Population] = {}
        self._build_populations()
        self._index_neurons()
        self.projections = self._build_projections()
        self._build_queries()

    # ------------------------------------------------------------ populations

    def _add(self, pop: Population) -> None:
        self.populations[pop.key] = pop

    def _build_populations(self) -> None:
        s = self.sensor_cfg
        # ---- Inputs (afferents). They come first so inputs are neurons [0, n_inputs).
        gx = np.linspace(s.vision_x_range[0], s.vision_x_range[1], s.vision_nx)
        gy = np.linspace(s.vision_y_range[0], s.vision_y_range[1], s.vision_ny)
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        vz = np.interp(X.ravel(), s.vision_x_range, (-1.0, 1.0))
        va = np.interp(Y.ravel(), s.vision_y_range, (-1.0, 1.0))
        for b, channel in ((-0.6, "ball"), (0.0, "target"), (0.6, "hand")):
            coords = np.stack([vz, va, np.full(vz.size, b)], axis=1)
            self._add(
                Population(
                    f"VIS.{channel}", "VIS", f"Retinotopic {channel} map", False, "TONIC", coords,
                    {"rf_x": X.ravel().copy(), "rf_y": Y.ravel().copy()},
                )
            )

        n_ang = s.angle_cells
        z, a, joint, pref = [], [], [], []
        for j in range(2):
            for k, u in enumerate(_centres(n_ang)):
                z.append(JOINT_Z[j]); a.append(u); joint.append(j); pref.append(u)
        self._add(
            Population(
                "DRG.angle", "DRG", "Joint-angle afferents (group II, population coded)", False, "TONIC",
                np.stack([z, a, np.full(len(z), -0.6)], axis=1).astype(float),
                {"joint": np.array(joint), "pref": np.array(pref)},
            )
        )
        self._add(
            Population(
                "DRG.Ia", "DRG", "Muscle spindle Ia afferents (stretch velocity)", False, "TONIC",
                np.array([[JOINT_Z[MUSCLE_JOINT[m]], MUSCLE_SIDE[m], 0.0] for m in range(4)]),
                {"muscle": np.arange(4)},
            )
        )
        self._add(
            Population(
                "DRG.touch", "DRG", "Cutaneous touch afferents (upper arm, forearm/hand)", False, "TONIC",
                np.array([[JOINT_Z[0], 0.0, 0.6], [JOINT_Z[1], 0.0, 0.6]]),
                {"segment": np.arange(2)},
            )
        )

        # ---- Cortex: layer 2/3 pyramidal (RS), interneurons (FS), layer 5 pyramidal (IB).
        cortex = {"S1": (4, 2, 2), "PPC": (6, 3, 2), "PMd": (4, 3, 2), "M1": (4, 6, 2)}  # (L2/3 nz, L5 nz, int nz)
        for key, (n23, n5, ni) in cortex.items():
            self._add(Population(f"{key}.L23", key, "Layer 2/3 pyramidal cells", False, "RS", _sheet(n23, 2, 0.6)))
            self._add(Population(f"{key}.INT", key, "Fast-spiking interneurons", True, "FS", _sheet(ni, 2, 0.0)))
            self._add(Population(f"{key}.L5", key, "Layer 5 pyramidal cells", False, "IB", _sheet(n5, 2, -0.6)))

        # ---- Cerebellum.
        self._add(Population("CB.GR", "CB", "Granule cells", False, "RS", _sheet(4, 2, 0.6)))
        self._add(Population("CB.PC", "CB", "Purkinje cells", True, "FS", _sheet(2, 2, 0.0)))
        self._add(Population("CB.DCN", "CB", "Deep cerebellar nuclei", False, "TC", _sheet(2, 2, -0.6)))

        # ---- Spinal cord: one Ia interneuron, Renshaw cell and excitatory interneuron per muscle, then motor pools.
        def per_muscle(b):
            return np.array([[JOINT_Z[MUSCLE_JOINT[m]], MUSCLE_SIDE[m], b] for m in range(4)])

        self._add(Population("SC.EIN", "SC", "Excitatory premotor interneurons", False, "RS", per_muscle(0.8), {"muscle": np.arange(4)}))
        self._add(Population("SC.IaIN", "SC", "Ia inhibitory interneurons", True, "FS", per_muscle(0.4), {"muscle": np.arange(4)}))
        self._add(Population("SC.RC", "SC", "Renshaw cells", True, "FS", per_muscle(0.0), {"muscle": np.arange(4)}))
        pool = 4
        mn, muscle = [], []
        for m in range(4):
            for k, b in enumerate(np.linspace(-0.4, -1.0, pool)):
                mn.append([JOINT_Z[MUSCLE_JOINT[m]], MUSCLE_SIDE[m], b]); muscle.append(m)
        self._add(Population("SC.MN", "SC", "Alpha motor neuron pools", False, "TONIC", np.array(mn), {"muscle": np.array(muscle)}))

    def _index_neurons(self) -> None:
        start = 0
        for p in self.populations.values():
            p.start = start
            start += p.n
        self.N = start
        pops = list(self.populations.values())
        self.coords = np.concatenate([p.coords for p in pops]).astype(np.float32)
        self.pop_of = np.concatenate([np.full(p.n, i) for i, p in enumerate(pops)])
        self.pop_keys = [p.key for p in pops]
        self.region_keys = list(self.regions)
        self.region_of = np.array([self.region_keys.index(pops[i].region) for i in self.pop_of])
        self.inhibitory = np.concatenate([np.full(p.n, p.inhibitory) for p in pops])
        izh = np.array([IZHIKEVICH[pops[i].cell] for i in self.pop_of])
        self.izh_a, self.izh_b, self.izh_c, self.izh_d = (izh[:, k].astype(np.float32) for k in range(4))
        self.input_keys = [k for k, p in self.populations.items() if p.region in ("VIS", "DRG")]
        self.n_inputs = sum(self.populations[k].n for k in self.input_keys)
        assert all(self.populations[k].start < self.n_inputs for k in self.input_keys)
        self.n_post = self.N - self.n_inputs
        mn = self.populations["SC.MN"]
        self.motor_slice = mn.slice
        self.motor_muscle = mn.meta["muscle"]

    # ------------------------------------------------------------ connectome

    def _build_projections(self) -> list[Projection]:
        P = []

        def add(src, dsts, tract, note):
            for s in src if isinstance(src, (list, tuple)) else [src]:
                for d in dsts if isinstance(dsts, (list, tuple)) else [dsts]:
                    P.append(Projection(s, d, tract, note))

        vis = ["VIS.ball", "VIS.target", "VIS.hand"]
        proprio = ["DRG.angle", "DRG.Ia"]
        # Afferents and thalamic relays.
        add(proprio + ["DRG.touch"], "S1.L23", "afferent", "dorsal column - medial lemniscus - VPL thalamus - S1")
        add(vis, "PPC.L23", "afferent", "dorsal visual stream to parietal cortex")
        # Cortico-cortical feedforward (L2/3 -> L2/3 and L5 of the next area).
        add("S1.L23", "PPC.L23", "feedforward", "S1 -> area 5")
        add("S1.L23", ["M1.L23", "M1.L5"], "feedforward", "areas 3a/2 -> area 4")
        add("PPC.L23", ["PMd.L23", "PMd.L5"], "feedforward", "parieto-frontal reach pathway")
        add("PPC.L23", "M1.L5", "feedforward", "area 5 -> M1")
        add("PMd.L23", ["M1.L23", "M1.L5"], "feedforward", "premotor -> motor")
        # Cortico-cortical feedback.
        add("M1.L5", ["PMd.L23", "S1.L23"], "feedback", "motor feedback / efference copy")
        add("PMd.L5", "PPC.L23", "feedback", "frontal -> parietal feedback")
        # Local microcircuits of every cortical area.
        for area in ("S1", "PPC", "PMd", "M1"):
            exc = [f"{area}.L23", f"{area}.L5"]
            add(exc, exc + [f"{area}.INT"], "local_exc", "recurrent pyramidal collaterals")
            add(f"{area}.INT", exc + [f"{area}.INT"], "local_inh", "basket-cell inhibition")
        # Cerebellum.
        add(proprio, ["CB.GR", "CB.DCN"], "cerebellar", "spinocerebellar mossy fibres")
        add(["M1.L5", "PPC.L5"], ["CB.GR", "CB.DCN"], "cerebellar", "cortico-ponto-cerebellar mossy fibres")
        add("CB.GR", "CB.PC", "cerebellar", "parallel fibres")
        add("CB.PC", "CB.DCN", "cerebellar", "Purkinje inhibition of the nuclei")
        add("CB.DCN", ["M1.L23", "M1.L5", "PMd.L23"], "cerebellar", "cerebello-thalamo-cortical (via VL thalamus)")
        # Descending.
        add("M1.L5", ["SC.MN", "SC.EIN", "SC.IaIN"], "descending", "corticospinal / corticomotoneuronal")
        add("PMd.L5", "SC.EIN", "descending", "premotor corticospinal")
        # Spinal circuits.
        add("DRG.Ia", "SC.MN", "spinal", "monosynaptic stretch reflex")
        add("DRG.Ia", "SC.IaIN", "spinal", "disynaptic reciprocal inhibition")
        add(["DRG.angle", "DRG.touch"], "SC.EIN", "spinal", "group II and cutaneous reflex pathways")
        add("SC.EIN", "SC.MN", "spinal", "premotor interneurons")
        add("SC.IaIN", ["SC.MN", "SC.IaIN"], "spinal", "reciprocal inhibition")
        add("SC.MN", "SC.RC", "spinal", "motor axon collaterals")
        add("SC.RC", ["SC.MN", "SC.IaIN"], "spinal", "recurrent (Renshaw) inhibition")

        for p in P:
            src, dst = self.populations[p.src], self.populations[p.dst]
            assert dst.region not in ("VIS", "DRG"), "afferents receive no synapses"
            if src.inhibitory:
                assert p.tract in ("local_inh", "cerebellar", "spinal"), "inhibitory cells only project locally"
        return P

    def _build_queries(self) -> None:
        pre, post, tract, proj = [], [], [], []
        for i, p in enumerate(self.projections):
            src, dst = self.populations[p.src], self.populations[p.dst]
            s_idx = np.arange(src.start, src.start + src.n)
            d_idx = np.arange(dst.start, dst.start + dst.n)
            S, D = np.meshgrid(s_idx, d_idx, indexing="ij")
            keep = S != D  # no autapses
            pre.append(S[keep]); post.append(D[keep])
            tract.append(np.full(keep.sum(), TRACT_INDEX[p.tract])); proj.append(np.full(keep.sum(), i))
        self.pre = np.concatenate(pre)
        self.post = np.concatenate(post)
        self.tract = np.concatenate(tract)
        self.proj = np.concatenate(proj)
        assert len(set(zip(self.pre.tolist(), self.post.tolist()))) == len(self.pre), "duplicate synapse in connectome"

        centers = np.array([self.regions[k].center for k in self.region_keys], dtype=np.float32)
        rc = centers[self.region_of]
        self.query = np.concatenate(
            [self.coords[self.pre], self.coords[self.post], rc[self.post] - rc[self.pre]], axis=1
        ).astype(np.float32)
        hidden = np.arange(self.n_inputs, self.N)
        self.bias_query = np.concatenate(
            [np.zeros((hidden.size, 3), np.float32), self.coords[hidden], np.zeros((hidden.size, 2), np.float32)], axis=1
        )

        # Peak weight of each synapse: tract gain normalised by the size of the presynaptic population, signed by
        # Dale's principle.
        gains = self.brain_cfg.tract_gain
        src_size = np.array([self.populations[p.src].n for p in self.projections])
        peak = np.array([gains[p.tract] for p in self.projections]) / np.sqrt(src_size)
        sign = np.where(self.inhibitory[self.pre], -1.0, 1.0)
        self.peak_weight = (peak[self.proj] * sign).astype(np.float32)

    # ------------------------------------------------------------- summaries

    def describe(self) -> str:
        lines = [f"Brain substrate: {self.N} neurons ({self.n_inputs} afferent), {len(self.pre)} candidate synapses"]
        for r in self.regions.values():
            pops = [p for p in self.populations.values() if p.region == r.key]
            parts = ", ".join(f"{p.key.split('.')[1]} {p.n}{' (inh)' if p.inhibitory else ''}" for p in pops)
            lines.append(f"  {r.key:4s} {r.name}: {parts}")
        counts = np.bincount(self.tract, minlength=len(TRACTS))
        lines.append("  candidate synapses per tract: " + ", ".join(f"{t} {c}" for t, c in zip(TRACTS, counts)))
        return "\n".join(lines)
