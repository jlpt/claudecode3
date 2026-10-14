"""
Two questions in one sweep:
  1. Does a BIGGER genome (more ant states x colours = more rules) build a better brain?
  2. Does adding an evolved LEARNING RULE (lifetime Hebbian learning) help?

Every genome size is evolved twice:
  fixed    - ants set neurons, wiring and final connection strengths (as before)
  learning - ants set neurons, wiring and STARTING strengths; the genome also holds
             a Hebbian rule  dW = lr * (A*pre*post + B*pre + C*post + D)  that adjusts
             existing wires during a short lifetime. During the lifetime a teacher
             holds the output neurons at the right answer; at test time it is gone.
No backpropagation anywhere. Same compute budget for every size.
"""
import copy, random, statistics, sys, time
from multiprocessing import Pool
import numpy as np

W_GRID, H_GRID = 120, 90
DX, DY = [0, 1, 0, -1], [-1, 0, 1, 0]
TURNS = 'LRNU'
ACTIONS = [None, 'neuron', 'split', 'stop']
N_IO, T_STEPS = 16, 6
MAX_NEURONS, MAX_ANTS, GROW_STEPS = 96, 48, 1500
LIFE_ROUNDS, LIFE_BATCH = 10, 64
SIZES = [(2, 2), (3, 3), (4, 4), (8, 4), (8, 8), (16, 8)]     # (states, colours)
POP, ELITE, GENERATIONS, SEEDS = 24, 5, 70, [0, 1]


def turn(d, t):
    return {'L': (d + 3) % 4, 'R': (d + 1) % 4, 'U': (d + 2) % 4}.get(t, d)


def random_genome(rng, S, C):
    rule = [[[rng.randrange(C), rng.choice(TURNS), rng.randrange(S),
              rng.choices(ACTIONS, [55, 20, 17, 8])[0], rng.uniform(-1, 1)]
             for _ in range(C)] for _ in range(S)]
    return {'rule': rule, 'bias': [rng.uniform(-0.5, 0.5) for _ in range(C)],
            'hebb': [rng.uniform(-1, 1) for _ in range(4)] + [rng.uniform(0, 0.3)]}


def n_genes(S, C, learning):
    return S * C * 5 + C + (5 if learning else 0)


def mutate(g, rng, learning):
    g = copy.deepcopy(g)
    S, C = len(g['rule']), len(g['rule'][0])
    for _ in range(rng.choice([1, 2, 3])):
        r = rng.random()
        if learning and r < 0.2:
            i = rng.randrange(5)
            lo, hi = (0.0, 1.0) if i == 4 else (-1.0, 1.0)
            g['hebb'][i] = float(np.clip(g['hebb'][i] + rng.gauss(0, 0.15), lo, hi)); continue
        if r < 0.3:
            c = rng.randrange(C); g['bias'][c] = float(np.clip(g['bias'][c] + rng.gauss(0, 0.3), -2, 2)); continue
        e = g['rule'][rng.randrange(S)][rng.randrange(C)]
        f = rng.choices(range(5), [1, 1, 1, 1, 3])[0]
        if f == 0: e[0] = rng.randrange(C)
        elif f == 1: e[1] = rng.choice(TURNS)
        elif f == 2: e[2] = rng.randrange(S)
        elif f == 3: e[3] = rng.choices(ACTIONS, [55, 20, 17, 8])[0]
        else: e[4] = float(np.clip(e[4] + rng.gauss(0, 0.3), -1, 1))
    return g


def grow(g):
    rule = g['rule']
    grid = bytearray(W_GRID * H_GRID)
    at, pos, born = {}, [], []
    weights, used = {}, set()

    def add(x, y):
        at[(x, y)] = len(pos); pos.append((x, y)); born.append(grid[y * W_GRID + x]); return len(pos) - 1

    def wire(a, b, w):
        if a != b: weights[(a, b)] = weights.get((a, b), 0.0) + w

    ants = [[W_GRID // 2, H_GRID // 2, 0, 0, add(W_GRID // 2, H_GRID // 2)]]
    for _ in range(GROW_STEPS):
        if not ants: break
        new, keep = [], []
        for x, y, d, s, last in ants:
            i = y * W_GRID + x
            c = grid[i]
            write, t, nxt, act, wg = rule[s][c]
            used.add((s, c))
            grid[i] = write
            here = at.get((x, y), -1)
            if here >= 0 and here != last:
                wire(last, here, wg); last = here
            if act == 'neuron' and here < 0 and len(pos) < MAX_NEURONS:
                nid = add(x, y); wire(last, nid, wg); last = nid
            elif act == 'split' and len(ants) + len(new) < MAX_ANTS:
                d2 = turn(d, {'L': 'R', 'R': 'L', 'N': 'U', 'U': 'N'}[t])
                new.append([(x + DX[d2]) % W_GRID, (y + DY[d2]) % H_GRID, d2, nxt, last])
            elif act == 'stop':
                continue
            d = turn(d, t)
            keep.append([(x + DX[d]) % W_GRID, (y + DY[d]) % H_GRID, d, nxt, last])
        ants = keep + new
    n = len(pos)
    Wm = np.zeros((n, n), dtype=np.float32)
    for (a, b), w in weights.items():
        Wm[a, b] = w
    bias = np.array([g['bias'][c] for c in born], dtype=np.float32)
    return Wm, bias, len(used)


def forward(Wm, bias, bits, teach=None):
    n = Wm.shape[0]
    inp = np.zeros((bits.shape[0], n), dtype=np.float32)
    inp[:, :N_IO] = bits * 2 - 1
    x = np.zeros_like(inp)
    for _ in range(T_STEPS):
        x = np.tanh(x @ Wm + bias + inp)
        if teach is not None:
            x[:, n - N_IO:] = teach[:, None]          # teacher holds outputs at the answer
    return x


def predict(Wm, bias, bits):
    return 8 + 8 * forward(Wm, bias, bits)[:, -N_IO:].mean(1)


def lifetime(Wm, bias, hebb, rng):
    """Hebbian learning on existing wires only, guided by a teacher."""
    A, B, C, D, lr = hebb
    mask = (Wm != 0).astype(np.float32)
    Wm = Wm.copy()
    for _ in range(LIFE_ROUNDS):
        bits = rng.integers(0, 2, (LIFE_BATCH, 16)).astype(np.float32)
        x = forward(Wm, bias, bits, teach=(bits.sum(1) - 8) / 8)
        pre_post = x.T @ x / LIFE_BATCH
        m = x.mean(0)
        dW = A * pre_post + B * m[:, None] + C * m[None, :] + D
        Wm = np.clip(Wm + lr * dW * mask, -3, 3)
    return Wm


def build(g, learning, rng):
    Wm, bias, used = grow(g)
    if Wm.shape[0] < 2 * N_IO:
        return None, None, used
    if learning:
        Wm = lifetime(Wm, bias, g['hebb'], rng)
    return Wm, bias, used


def mse(Wm, bias, bits):
    return float(np.mean((predict(Wm, bias, bits) - bits.sum(1)) ** 2))


def run_one(args):
    S, C, learning, seed = args
    rng = random.Random(seed * 1000 + S * 10 + C)
    nrng = np.random.default_rng(seed)
    pop = [random_genome(rng, S, C) for _ in range(POP)]
    curve = []
    for gen in range(GENERATIONS):
        bits = nrng.integers(0, 2, (256, 16)).astype(np.float32)
        life_seed = int(nrng.integers(1 << 30))
        scored = []
        for i, g in enumerate(pop):
            Wm, bias, _ = build(g, learning, np.random.default_rng(life_seed))
            scored.append((-99.0 if Wm is None else -mse(Wm, bias, bits), i, g))
        scored.sort(key=lambda t: t[:2], reverse=True)
        curve.append(-scored[0][0])
        parents = [s[2] for s in scored[:ELITE]]
        pop = parents + [mutate(rng.choice(parents), rng, learning) for _ in range(POP - ELITE)]
    best = parents[0]
    test = np.random.default_rng(999).integers(0, 2, (5000, 16)).astype(np.float32)
    Wm, bias, used = build(best, learning, np.random.default_rng(42))   # fresh lifetime
    if Wm is None:
        return dict(S=S, C=C, learning=learning, seed=seed, mse=99.0, exact=0.0, used=used, curve=curve)
    pred = predict(Wm, bias, test)
    return dict(S=S, C=C, learning=learning, seed=seed, mse=mse(Wm, bias, test),
                exact=float(np.mean(np.round(pred) == test.sum(1))), used=used,
                neurons=Wm.shape[0], wires=int((Wm != 0).sum()), hebb=best['hebb'], curve=curve)


if __name__ == '__main__':
    jobs = [(S, C, learning, seed) for (S, C) in SIZES for learning in (False, True) for seed in SEEDS]
    jobs.sort(key=lambda j: -(j[0] * j[1]))          # biggest first for better packing
    t0 = time.time()
    results = []
    with Pool(2) as pool:
        for r in pool.imap_unordered(run_one, jobs):
            results.append(r)
            print(f"[{time.time() - t0:5.0f}s] {r['S']:2d}x{r['C']} {'learning' if r['learning'] else 'fixed   '} "
                  f"seed {r['seed']}: test MSE {r['mse']:.3f}  exact {r['exact']:.1%}  rules used {r['used']}/{r['S'] * r['C']}",
                  flush=True)
    np.save('sweep_results.npy', np.array(results, dtype=object), allow_pickle=True)
    print('\nSummary (mean of seeds; guessing 8 every time = MSE 4.0)')
    print(f"{'states x colours':>17} {'rules':>6} {'genes fixed/learn':>18} {'fixed MSE':>10} {'learning MSE':>13} {'rules used':>11}")
    for S, C in SIZES:
        f = [r for r in results if (r['S'], r['C']) == (S, C) and not r['learning']]
        l = [r for r in results if (r['S'], r['C']) == (S, C) and r['learning']]
        used = statistics.mean(r['used'] for r in f + l)
        print(f"{S:>11} x {C:<3} {S * C:>6} {n_genes(S, C, False):>8}/{n_genes(S, C, True):<9} "
              f"{statistics.mean(r['mse'] for r in f):>10.3f} {statistics.mean(r['mse'] for r in l):>13.3f} {used:>8.0f}/{S * C}")
