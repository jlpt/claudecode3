"""
Does wiring grown by turmite rules help a graph of tiny MLPs learn?

For each turmite genome:
  1. Grow a network (capped at 128 neurons) exactly like the Turmite Wiring Lab.
  2. Turn every neuron into a tiny MLP and every wire into a message link
     (wire a -> b means b reads a's output), as in graph_of_mlps.py.
  3. Train it on the bit-counting task: bits enter the 16 earliest-grown neurons,
     the answer is read from the 16 latest-grown neurons.
  4. Compare against controls with the SAME neurons and the SAME number of
     incoming wires per neuron, but with the sources picked at random.
"""
import math, random, statistics, sys, time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)

# ----------------------------------------------------------------- turmites
W, H = 120, 90
DX, DY = [0, 1, 0, -1], [-1, 0, 1, 0]
CODE_ACT = {'-': None, 'n': 'neuron', 's': 'split', 'x': 'stop'}


def parse(genome):
    return [[(int(t[0]), t[1], int(t[2]), CODE_ACT[t[3]]) for t in row.split()]
            for row in genome.split('|')]


def turn(d, t):
    return {'L': (d + 3) % 4, 'R': (d + 1) % 4, 'U': (d + 2) % 4}.get(t, d)


def grow(genome, steps=3000, max_neurons=128, max_ants=48):
    rule = parse(genome)
    grid = [0] * (W * H)
    at = {}
    neurons, edges = [], set()

    def add(x, y):
        at[(x, y)] = len(neurons); neurons.append((x, y)); return len(neurons) - 1

    n0 = add(W // 2, H // 2)
    ants = [[W // 2, H // 2, 0, 0, n0, True]]           # x, y, dir, state, last, alive
    for _ in range(steps):
        if not any(a[5] for a in ants):
            break
        born = []
        for a in ants:
            if not a[5]:
                continue
            x, y, d, s, last, _ = a
            i = y * W + x
            write, t, nxt, act = rule[s][grid[i]]
            grid[i] = write
            here = at.get((x, y), -1)
            if here >= 0 and here != last:
                edges.add((last, here)); last = here
            if act == 'neuron' and here < 0 and len(neurons) < max_neurons:
                nid = add(x, y); edges.add((last, nid)); last = nid
            elif act == 'split' and len(ants) + len(born) < max_ants:
                m = {'L': 'R', 'R': 'L', 'N': 'U', 'U': 'N'}[t]
                born.append([x, y, turn(d, m), nxt, last, True])
            elif act == 'stop':
                a[5] = False; a[4] = last; continue
            d = turn(d, t)
            a[:] = [(x + DX[d]) % W, (y + DY[d]) % H, d, nxt, last, True]
        for b in born:
            b[0] = (b[0] + DX[b[2]]) % W; b[1] = (b[1] + DY[b[2]]) % H
            ants.append(b)
    edges.discard(None)
    return len(neurons), sorted(e for e in edges if e[0] != e[1])


# ------------------------------------------------------------ graph helpers
def in_lists(n, edges):
    src = [[] for _ in range(n)]
    for a, b in edges:
        src[b].append(a)
    return src


def random_rewire(n, edges, rng):
    """Same in-degree for every neuron, but each source chosen at random."""
    src = in_lists(n, edges)
    new = []
    for b in range(n):
        picks = rng.sample([i for i in range(n) if i != b], min(len(src[b]), n - 1))
        new += [(a, b) for a in picks]
    return new


def ring_small_world(n, n_edges, rng):
    """The prototype's design: two-way local ring links plus random shortcuts,
    scaled to the same total number of wires."""
    k = max(1, round(n_edges / n))
    local = max(2, int(k * 0.75) // 2 * 2)
    edges = set()
    for b in range(n):
        for o in range(1, local // 2 + 1):
            edges.add(((b + o) % n, b)); edges.add(((b - o) % n, b))
    while len(edges) < n_edges:
        a, b = rng.randrange(n), rng.randrange(n)
        if a != b:
            edges.add((a, b))
    return sorted(edges)


def hop_stats(n, edges, sources, targets):
    out = [[] for _ in range(n)]
    for a, b in edges:
        out[a].append(b)
    dist = [-1] * n
    q = deque(sources)
    for s in sources:
        dist[s] = 0
    while q:
        u = q.popleft()
        for v in out[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1; q.append(v)
    reach = [dist[t] for t in targets if dist[t] >= 0]
    return len(reach), (max(reach) if reach else None)


def two_way(edges):
    s = set(edges)
    return sum((b, a) in s for a, b in edges) / max(1, len(edges))


# ------------------------------------------------------------------ model
class GraphOfMLPs(nn.Module):
    def __init__(self, n, edges, d=16, hidden=32, steps=6, active_frac=0.25, n_io=16):
        super().__init__()
        src = in_lists(n, edges)
        k = max(1, max(len(s) for s in src))
        nbrs = torch.zeros(n, k, dtype=torch.long)
        mask = torch.zeros(n, k)
        for b, s in enumerate(src):
            if s:
                nbrs[b, :len(s)] = torch.tensor(s); mask[b, :len(s)] = 1
        self.register_buffer('nbrs', nbrs); self.register_buffer('mask', mask)
        self.edge_logits = nn.Parameter(torch.zeros(n, k))
        self.W1 = nn.Parameter(torch.randn(n, 2 * d, hidden) / math.sqrt(2 * d))
        self.b1 = nn.Parameter(torch.zeros(n, hidden))
        self.W2 = nn.Parameter(torch.randn(n, hidden, d) / math.sqrt(hidden) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(n, d))
        self.ns, self.ni = nn.LayerNorm(d), nn.LayerNorm(d)
        self.n, self.d, self.steps = n, d, steps
        self.n_active = max(1, int(active_frac * n))
        self.in_nodes = list(range(n_io))                 # earliest-grown neurons
        self.out_nodes = list(range(n - n_io, n))         # latest-grown neurons
        self.bit = nn.Embedding(2, d)
        self.pos = nn.Parameter(torch.randn(n_io, d) * 0.1)
        self.readout = nn.Linear(d, 17)

    def forward(self, bits):
        B = bits.shape[0]
        h = torch.zeros(B, self.n, self.d)
        h[:, self.in_nodes] = self.bit(bits) + self.pos
        logits = self.edge_logits.masked_fill(self.mask == 0, -1e9)
        w = F.softmax(logits, -1) * self.mask
        for _ in range(self.steps):
            inbox = torch.einsum('nk,bnkd->bnd', w, h[:, self.nbrs])
            x = torch.cat([self.ns(h), self.ni(inbox)], -1)
            z = F.gelu(torch.einsum('bni,nih->bnh', x, self.W1) + self.b1)
            u = torch.einsum('bnh,nhd->bnd', z, self.W2) + self.b2
            st = u.norm(dim=-1)
            cut = st.topk(self.n_active, -1).values[:, -1:]
            h = h + (st >= cut).float().unsqueeze(-1) * u
        return self.readout(h[:, self.out_nodes].mean(1))


def train(n, edges, seed, iters=250):
    torch.manual_seed(seed)
    m = GraphOfMLPs(n, edges)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(iters):
        bits = torch.randint(0, 2, (32, 16))
        loss = F.cross_entropy(m(bits), bits.sum(1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        bits = torch.randint(0, 2, (2000, 16))
        acc = (m(bits).argmax(-1) == bits.sum(1)).float().mean().item()
    return statistics.mean(losses), acc     # mean training loss = area under the curve


# ------------------------------------------------------------------- main
GENOMES = {
    'Branching web': '1R0- 2U1- 0U2- | 1L2s 2L1- 2R2- | 2U2- 0U2n 2L1-',
    'Lattice':       '0L2- 1N2s 0N1- 2L0s | 0N0- 1R1- 3U0- 2L0- | 3N1n 3N2- 2R2s 3R1s',
    'Cross':         '2U2- 0U0- 1R0n | 0R2s 0R2- 2N0n | 0L1n 1R2- 2U0-',
    'Spindle':       '0N2n 2L1s 1N2- 3L0s | 3L0- 0N2- 2N2- 2L0n | 1N2n 2U2- 2N1s 1L1-',
}
SEEDS = [0, 1]

if __name__ == '__main__':
    rows = []
    for name, g in GENOMES.items():
        n, edges = grow(g)
        rng = random.Random(7)
        variants = {
            'grown by turmites': edges,
            'random, same in-degree': random_rewire(n, edges, rng),
            'ring + shortcuts': ring_small_world(n, len(edges), rng),
        }
        for kind, e in variants.items():
            reach, far = hop_stats(n, e, list(range(16)), list(range(n - 16, n)))
            t0 = time.time()
            res = [train(n, e, s) for s in SEEDS]
            loss = statistics.mean(r[0] for r in res)
            acc = statistics.mean(r[1] for r in res)
            rows.append((name, kind, n, len(e), two_way(e), reach, far, loss, acc))
            print(f'{name:14s} {kind:24s} n={n:3d} wires={len(e):4d} two-way={two_way(e):4.0%} '
                  f'outputs reached={reach:2d}/16 max hops={far}  '
                  f'mean loss={loss:.3f} test acc={acc:6.1%}  ({time.time()-t0:.0f}s)', flush=True)
