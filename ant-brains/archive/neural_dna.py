"""
Rule-table DNA vs neural-network DNA for growing a brain.

Neural DNA: every ant carries the same tiny network (~700 weights = the genome).
  Senses : colour under / ahead / left / right (one-hot), standing on a neuron?,
           growth progress, and its own 4-number memory.
  Decides: turn, colour to write, action (none / neuron / split / stop),
           strength of the wire it lays, bias of a neuron it drops, new memory.
Rule-table DNA: the 4-state x 4-colour turmite table from genome_sweep.py.

Both: same grid, same 96-neuron cap, same task (count 1s in 16 bits), same
population and generations, and every genome in a generation takes the same test.
No backprop anywhere.
"""
import random, statistics, time
from multiprocessing import Pool
import numpy as np
import genome_sweep as gs        # rule-table version + brain/test code

W_GRID, H_GRID = gs.W_GRID, gs.H_GRID
DX, DY = np.array(gs.DX), np.array(gs.DY)
C = 4                                   # colours
N_IN, N_HID, N_MEM = 4 * C + 2 + 4 + 1, 16, 4
N_OUT = 4 + C + 4 + 1 + 1 + N_MEM       # turn, write, action, weight, bias, memory
SHAPES = [(N_IN, N_HID), (N_HID,), (N_HID, N_OUT), (N_OUT,)]
N_GENES = sum(int(np.prod(s)) for s in SHAPES)
POP, ELITE, GENERATIONS = 24, 5, 80
SEEDS = [0, 1, 2]


def unpack(v):
    out, i = [], 0
    for s in SHAPES:
        n = int(np.prod(s)); out.append(v[i:i + n].reshape(s)); i += n
    return out


def random_neural(rng):
    v = rng.normal(0, 0.5, N_GENES).astype(np.float32)
    b2 = unpack(v)[3]                     # a view into v
    b2[8] += 1.5                          # start biased toward "no action"
    return v


def mutate_neural(v, rng):
    return (v + rng.normal(0, 0.05, v.shape)).astype(np.float32)   # small nudge to every weight


def grow_neural(v):
    W1, b1, W2, b2 = unpack(v)
    grid = np.zeros((H_GRID, W_GRID), np.uint8)
    nat = -np.ones((H_GRID, W_GRID), np.int32)
    pos, nbias, weights = [], [], {}
    eye = np.eye(C, dtype=np.float32)

    def add(x, y, b):
        nat[y, x] = len(pos); pos.append((x, y)); nbias.append(b); return len(pos) - 1

    cx, cy = W_GRID // 2, H_GRID // 2
    ants = [dict(x=cx, y=cy, d=0, h=np.zeros(N_MEM, np.float32), last=add(cx, cy, 0.0))]
    for step in range(gs.GROW_STEPS):
        if not ants: break
        xs = np.array([a['x'] for a in ants]); ys = np.array([a['y'] for a in ants])
        ds = np.array([a['d'] for a in ants])

        def look(dd):
            return grid[(ys + DY[dd]) % H_GRID, (xs + DX[dd]) % W_GRID]
        X = np.concatenate([
            eye[grid[ys, xs]], eye[look(ds)], eye[look((ds + 3) % 4)], eye[look((ds + 1) % 4)],
            (nat[ys, xs] >= 0)[:, None].astype(np.float32),
            np.full((len(ants), 1), step / gs.GROW_STEPS, np.float32),
            np.stack([a['h'] for a in ants]),
            np.ones((len(ants), 1), np.float32)], 1)
        out = np.tanh(X @ W1 + b1) @ W2 + b2
        turns = out[:, 0:4].argmax(1); writes = out[:, 4:4 + C].argmax(1)
        acts = out[:, 4 + C:8 + C].argmax(1)
        wgt = np.tanh(out[:, 8 + C]); nb = np.tanh(out[:, 9 + C]); hs = np.tanh(out[:, 10 + C:])

        keep, new = [], []
        for k, a in enumerate(ants):
            x, y, last = a['x'], a['y'], a['last']
            grid[y, x] = writes[k]
            here = nat[y, x]
            if here >= 0 and here != last:
                weights[(last, here)] = weights.get((last, here), 0.0) + wgt[k]; last = here
            act = acts[k]                                   # 0 none, 1 neuron, 2 split, 3 stop
            t = 'LRNU'[turns[k]]
            if act == 1 and here < 0 and len(pos) < gs.MAX_NEURONS:
                nid = add(x, y, nb[k]); weights[(last, nid)] = weights.get((last, nid), 0.0) + wgt[k]; last = nid
            elif act == 2 and len(ants) + len(new) < gs.MAX_ANTS:
                d2 = gs.turn(a['d'], {'L': 'R', 'R': 'L', 'N': 'U', 'U': 'N'}[t])
                new.append(dict(x=(x + DX[d2]) % W_GRID, y=(y + DY[d2]) % H_GRID, d=d2, h=hs[k].copy(), last=last))
            elif act == 3:
                continue
            d = gs.turn(a['d'], t)
            keep.append(dict(x=(x + DX[d]) % W_GRID, y=(y + DY[d]) % H_GRID, d=d, h=hs[k], last=last))
        ants = keep + new
    n = len(pos)
    Wm = np.zeros((n, n), np.float32)
    for (a, b), w in weights.items():
        if a != b: Wm[a, b] = w
    return Wm, np.array(nbias, np.float32)


# ------------------------------------------------------------------ shared scoring
def score(kind, g, bits):
    if kind == 'neural':
        Wm, bias = grow_neural(g)
    else:
        Wm, bias, _ = gs.grow(g)
    if Wm.shape[0] < 2 * gs.N_IO:
        return 99.0
    return gs.mse(Wm, bias, bits)


def run(args):
    kind, seed = args
    rng = random.Random(seed); nrng = np.random.default_rng(seed)
    if kind == 'neural':
        pop = [random_neural(nrng) for _ in range(POP)]
        mut = lambda g: mutate_neural(g, nrng)
    else:
        pop = [gs.random_genome(rng, 4, 4) for _ in range(POP)]
        mut = lambda g: gs.mutate(g, rng, False)
    curve, median_curve = [], []
    for gen in range(GENERATIONS):
        bits = nrng.integers(0, 2, (256, 16)).astype(np.float32)
        scored = sorted(((score(kind, g, bits), i, g) for i, g in enumerate(pop)), key=lambda t: t[:2])
        curve.append(scored[0][0]); median_curve.append(statistics.median(s[0] for s in scored))
        parents = [s[2] for s in scored[:ELITE]]
        pop = parents + [mut(rng.choice(parents)) for _ in range(POP - ELITE)]
    best = parents[0]
    test = np.random.default_rng(999).integers(0, 2, (3000, 16)).astype(np.float32)
    best_mse = score(kind, best, test)
    # Smoothness: how much does ONE mutation change performance?
    kids = [score(kind, mut(best), test) for _ in range(30)]
    return dict(kind=kind, seed=seed, mse=best_mse, kids=kids, curve=curve, median=median_curve)


if __name__ == '__main__':
    print(f'neural DNA has {N_GENES} genes; rule-table DNA (4x4) has {gs.n_genes(4, 4, False)} genes', flush=True)
    t0 = time.time()
    jobs = [(k, s) for s in SEEDS for k in ('neural', 'table')]
    results = []
    with Pool(2) as pool:
        for r in pool.imap_unordered(run, jobs):
            results.append(r)
            k = np.array(r['kids'])
            print(f"[{time.time() - t0:5.0f}s] {r['kind']:6s} seed {r['seed']}: test MSE {r['mse']:.3f} | "
                  f"one mutation: median child MSE {np.median(k):.2f}, "
                  f"{np.mean(np.abs(k - r['mse']) < 0.25):.0%} of children within 0.25 of parent, "
                  f"{np.mean(k >= 3.9):.0%} no better than guessing", flush=True)
    np.save('neural_vs_table.npy', np.array(results, dtype=object), allow_pickle=True)
    print('\nSummary (guessing 8 every time = MSE 4.0)')
    for kind in ('table', 'neural'):
        rs = [r for r in results if r['kind'] == kind]
        allk = np.concatenate([r['kids'] for r in rs])
        print(f"{kind:6s}: best-genome MSE per run {[round(r['mse'], 2) for r in rs]}  mean {statistics.mean(r['mse'] for r in rs):.2f} | "
              f"children within 0.25 of parent {np.mean(np.concatenate([np.abs(np.array(r['kids']) - r['mse']) < 0.25 for r in rs])):.0%} | "
              f"final-generation median {statistics.mean(r['median'][-1] for r in rs):.2f}")
