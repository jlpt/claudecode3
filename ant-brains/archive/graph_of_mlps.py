"""
A sparse graph of tiny feed-forward networks that talk to each other.

Each node is its own small MLP. Nodes are wired in a sparse "small-world" graph:
mostly local, two-way links (so signals can flow forward AND back), plus a few
random long-range links. The graph runs for several steps, so information can
loop around, and only a fraction of nodes "fire" each step (activity sparsity).

Toy task: 16 bits go into 16 input nodes; 16 output nodes far away in the graph
must report how many bits were 1. They can only do that if information travels
across the graph through the node-to-node connections.
"""
import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. THE WIRING: who listens to whom
# ---------------------------------------------------------------------------
def build_small_world(n_nodes, n_local, n_long, gen):
    """Return nbrs[i] = list of nodes that node i READS from.

    Local links are symmetric (i hears i+1 and i+1 hears i), which creates
    feedback loops: a node's output comes back to it via its neighbors.
    Long-range links are random shortcuts that make the graph mix quickly.
    """
    half = n_local // 2
    offsets = [o for o in range(-half, half + 1) if o != 0]          # e.g. -3..-1, 1..3
    nbrs = []
    for i in range(n_nodes):
        local = [(i + o) % n_nodes for o in offsets]
        long_range = torch.randint(0, n_nodes, (n_long,), generator=gen).tolist()
        nbrs.append(local + long_range)
    return torch.tensor(nbrs)                                           # [N, k]


def hops(nbrs, sources, targets):
    """Shortest number of steps for a signal to get from any source to each target."""
    n = nbrs.shape[0]
    out_edges = [[] for _ in range(n)]
    for i in range(n):
        for j in nbrs[i].tolist():
            out_edges[j].append(i)          # j's output flows to i
    dist = [-1] * n
    q = deque(sources)
    for s in sources:
        dist[s] = 0
    while q:
        u = q.popleft()
        for v in out_edges[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return max(dist[t] for t in targets)


# ---------------------------------------------------------------------------
# 2. THE MODEL
# ---------------------------------------------------------------------------
class GraphOfMLPs(nn.Module):
    def __init__(self, n_nodes=256, d=32, hidden=64, n_local=6, n_long=2,
                 steps=5, active_frac=0.25, n_in=16, n_out=16, n_classes=17, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.N, self.d, self.steps = n_nodes, d, steps
        self.n_active = int(active_frac * n_nodes)

        # Wiring (fixed, not learned) and learned strength of each connection
        self.register_buffer("nbrs", build_small_world(n_nodes, n_local, n_long, gen))
        k = self.nbrs.shape[1]
        self.edge_logits = nn.Parameter(torch.zeros(n_nodes, k))       # "synapse strengths"

        # Every node gets its OWN MLP. Stored as stacked weights so all
        # N little networks run in one batched operation.
        #   input to each MLP = [its own state, what it heard from neighbors]
        self.W1 = nn.Parameter(torch.randn(n_nodes, 2 * d, hidden) / math.sqrt(2 * d))
        self.b1 = nn.Parameter(torch.zeros(n_nodes, hidden))
        self.W2 = nn.Parameter(torch.randn(n_nodes, hidden, d) / math.sqrt(hidden) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(n_nodes, d))
        self.norm_self = nn.LayerNorm(d)
        self.norm_inbox = nn.LayerNorm(d)

        # Input nodes at one side of the ring, output nodes on the opposite side
        self.in_nodes = list(range(n_in))
        self.out_nodes = list(range(n_nodes // 2, n_nodes // 2 + n_out))
        self.bit_embed = nn.Embedding(2, d)
        self.in_pos = nn.Parameter(torch.randn(n_in, d) * 0.1)
        self.readout = nn.Linear(d, n_classes)

    def step(self, h):
        """One round of every node listening, thinking, and (maybe) speaking."""
        # --- THE CONNECTION: each node gathers its neighbors' current outputs ---
        msgs = h[:, self.nbrs]                                   # [B, N, k, d]
        w = F.softmax(self.edge_logits, dim=-1)                  # [N, k]
        inbox = torch.einsum("nk,bnkd->bnd", w, msgs)            # weighted sum of what it heard

        # --- Each node runs its own private MLP on (self, inbox) ---
        x = torch.cat([self.norm_self(h), self.norm_inbox(inbox)], dim=-1)   # [B, N, 2d]
        z = F.gelu(torch.einsum("bni,nih->bnh", x, self.W1) + self.b1)
        update = torch.einsum("bnh,nhd->bnd", z, self.W2) + self.b2          # [B, N, d]

        # --- Activity sparsity: only the most strongly driven nodes fire ---
        strength = update.norm(dim=-1)                                       # [B, N]
        cutoff = strength.topk(self.n_active, dim=-1).values[:, -1:]
        fired = (strength >= cutoff).float().unsqueeze(-1)                   # [B, N, 1]
        h = h + fired * update        # silent nodes keep their old state
        return h, fired.squeeze(-1)

    def forward(self, bits):
        B = bits.shape[0]
        h = torch.zeros(B, self.N, self.d, device=bits.device)
        h[:, self.in_nodes] = self.bit_embed(bits) + self.in_pos   # inject the input

        firing = []
        for _ in range(self.steps):                                # unrolled loop: signals
            h, fired = self.step(h)                                # can bounce back and forth
            firing.append(fired)
        out = h[:, self.out_nodes].mean(dim=1)
        return self.readout(out), torch.stack(firing, 1)           # logits, [B, steps, N]


# ---------------------------------------------------------------------------
# 3. TRAINING ON THE TOY TASK
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    model = GraphOfMLPs()
    n_params = sum(p.numel() for p in model.parameters())
    nb = model.nbrs
    recip = sum(int(i in nb[j].tolist()) for i in range(model.N) for j in nb[i].tolist())
    print(f"{model.N} tiny MLPs, {nb.shape[1]} incoming links each, {n_params:,} params total")
    print(f"{recip / nb.numel():.0%} of links are two-way (feedback loops)")
    print(f"hops from input nodes to farthest output node: "
          f"{hops(nb, model.in_nodes, model.out_nodes)} (model runs {model.steps} steps)\n")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for it in range(1, 601):
        bits = torch.randint(0, 2, (128, 16))
        target = bits.sum(dim=1)                         # how many 1s (0..16)
        logits, firing = model(bits)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 100 == 0:
            acc = (logits.argmax(-1) == target).float().mean()
            print(f"step {it:4d}  loss {loss.item():.3f}  acc {acc.item():.0%}")

    # Do nodes specialize? See how often each node fires across many inputs.
    with torch.no_grad():
        _, firing = model(torch.randint(0, 2, (1000, 16)))
        rate = firing.mean(dim=(0, 1))
    print(f"\nnodes that never fire: {(rate == 0).sum().item()}, "
          f"always fire: {(rate == 1).sum().item()}, "
          f"sometimes (input-dependent): {((rate > 0) & (rate < 1)).sum().item()}")


if __name__ == "__main__":
    main()
