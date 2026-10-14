"""
Turmites that build EVERYTHING: neurons, wiring, and connection strengths.
No backpropagation: the only thing that changes is the genome, by evolution.

Genome
  rule[state][colour] = (write, turn, next_state, action, weight_gene)
  bias[colour]        = resting bias of a neuron born on that colour
  - weight_gene in [-1, 1]: each time an ant travels a wire a -> b while using
    this rule entry, it adds the gene to that wire's strength. Busy routes get
    strong (like pheromone trails); negative genes make inhibitory wires.

Network
  Each neuron is one number. For T steps: x = tanh(W^T x + bias + input).
  16 earliest-grown neurons receive the 16 input bits (+1 / -1).
  Prediction = 8 + 8 * mean(activity of the 16 latest-grown neurons).

Task: count the 1s among 16 bits (0..16).  Guessing 8 every time scores MSE 4.0.
"""
import copy, random, statistics, time
import numpy as np

W_GRID, H_GRID = 120, 90
DX, DY = [0, 1, 0, -1], [-1, 0, 1, 0]
TURNS = 'LRNU'
ACTIONS = [None, 'neuron', 'split', 'stop']
N_IO, T_STEPS = 16, 6
MAX_NEURONS, MAX_ANTS, GROW_STEPS = 96, 48, 1500


def turn(d, t):
    return {'L': (d + 3) % 4, 'R': (d + 1) % 4, 'U': (d + 2) % 4}.get(t, d)


# ------------------------------------------------------------------ genome
def random_genome(rng, S=3, C=3):
    rule = [[[rng.randrange(C), rng.choice(TURNS), rng.randrange(S),
              rng.choices(ACTIONS, [55, 20, 17, 8])[0], rng.uniform(-1, 1)]
             for _ in range(C)] for _ in range(S)]
    return {'rule': rule, 'bias': [rng.uniform(-0.5, 0.5) for _ in range(C)]}


def from_preset(text, rng):
    code = {'-': None, 'n': 'neuron', 's': 'split', 'x': 'stop'}
    rule = [[[int(t[0]), t[1], int(t[2]), code[t[3]], rng.uniform(-1, 1)]
             for t in row.split()] for row in text.split('|')]
    return {'rule': rule, 'bias': [rng.uniform(-0.5, 0.5) for _ in rule[0]]}


def mutate(g, rng, n_changes=2):
    g = copy.deepcopy(g)
    S, C = len(g['rule']), len(g['rule'][0])
    for _ in range(n_changes):
        if rng.random() < 0.15:
            c = rng.randrange(C); g['bias'][c] = float(np.clip(g['bias'][c] + rng.gauss(0, 0.3), -2, 2)); continue
        e = g['rule'][rng.randrange(S)][rng.randrange(C)]
        f = rng.choices(range(5), [1, 1, 1, 1, 3])[0]        # weight genes mutate most often
        if f == 0: e[0] = rng.randrange(C)
        elif f == 1: e[1] = rng.choice(TURNS)
        elif f == 2: e[2] = rng.randrange(S)
        elif f == 3: e[3] = rng.choices(ACTIONS, [55, 20, 17, 8])[0]
        else: e[4] = float(np.clip(e[4] + rng.gauss(0, 0.3), -1, 1))
    return g


# ------------------------------------------------------------------ growth
def grow(g):
    rule = g['rule']
    grid = bytearray(W_GRID * H_GRID)
    at, pos, born_colour = {}, [], []
    weights = {}

    def add(x, y):
        at[(x, y)] = len(pos); pos.append((x, y)); born_colour.append(grid[y * W_GRID + x]); return len(pos) - 1

    def wire(a, b, w):
        if a != b: weights[(a, b)] = weights.get((a, b), 0.0) + w

    n0 = add(W_GRID // 2, H_GRID // 2)
    ants = [[W_GRID // 2, H_GRID // 2, 0, 0, n0]]
    for _ in range(GROW_STEPS):
        if not ants: break
        new, keep = [], []
        for a in ants:
            x, y, d, s, last = a
            i = y * W_GRID + x
            write, t, nxt, act, wg = rule[s][grid[i]]
            grid[i] = write
            here = at.get((x, y), -1)
            if here >= 0 and here != last:
                wire(last, here, wg); last = here
            if act == 'neuron' and here < 0 and len(pos) < MAX_NEURONS:
                nid = add(x, y); wire(last, nid, wg); last = nid
            elif act == 'split' and len(ants) + len(new) < MAX_ANTS:
                m = {'L': 'R', 'R': 'L', 'N': 'U', 'U': 'N'}[t]
                d2 = turn(d, m)
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
    bias = np.array([g['bias'][c] for c in born_colour], dtype=np.float32)
    return Wm, bias


# ------------------------------------------------------------------ running the brain
def run(Wm, bias, bits):
    """bits: [B, 16] of 0/1  ->  predicted counts [B]"""
    n = Wm.shape[0]
    if n < 2 * N_IO:
        return None
    B = bits.shape[0]
    inp = np.zeros((B, n), dtype=np.float32)
    inp[:, :N_IO] = bits * 2 - 1
    x = np.zeros((B, n), dtype=np.float32)
    for _ in range(T_STEPS):
        x = np.tanh(x @ Wm + bias + inp)
    return 8 + 8 * x[:, n - N_IO:].mean(1)


def fitness(g, bits):
    Wm, bias = grow(g)
    pred = run(Wm, bias, bits)
    if pred is None:
        return -99.0, Wm.shape[0]
    return -float(np.mean((pred - bits.sum(1)) ** 2)), Wm.shape[0]


# ------------------------------------------------------------------ evolution
PRESETS = {
    'Branching web': '1R0- 2U1- 0U2- | 1L2s 2L1- 2R2- | 2U2- 0U2n 2L1-',
    'Cross':         '2U2- 0U0- 1R0n | 0R2s 0R2- 2N0n | 0L1n 1R2- 2U0-',
    'Spindle':       '0N2n 2L1s 1N2- 3L0s | 3L0- 0N2- 2N2- 2L0n | 1N2n 2U2- 2N1s 1L1-',
}


def evolve(generations=120, pop=32, elite=6, seed=0):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    population = [from_preset(p, rng) for p in PRESETS.values() for _ in range(3)]
    population += [random_genome(rng) for _ in range(pop - len(population))]
    history = []
    best = None
    for gen in range(generations):
        bits = nrng.integers(0, 2, (256, 16)).astype(np.float32)   # fresh exam every generation
        scored = sorted(((fitness(g, bits)[0], i, g) for i, g in enumerate(population)), reverse=True, key=lambda t: t[:2])
        best = scored[0][2]
        history.append(-scored[0][0])
        if gen % 10 == 0 or gen == generations - 1:
            print(f'gen {gen:3d}  best MSE {-scored[0][0]:.3f}  median {-statistics.median(s[0] for s in scored):.3f}', flush=True)
        parents = [s[2] for s in scored[:elite]]
        population = parents + [mutate(rng.choice(parents), rng, rng.choice([1, 2, 3])) for _ in range(pop - elite)]
    return best, history


# ------------------------------------------------------------------ comparison: same wiring, weights by backprop
def backprop_same_wiring(g, iters=400, seed=0):
    import torch
    torch.manual_seed(seed)
    Wm, bias = grow(g)
    mask = torch.tensor(Wm != 0, dtype=torch.float32)
    Wt = torch.nn.Parameter(torch.tensor(Wm))
    bt = torch.nn.Parameter(torch.tensor(bias))
    opt = torch.optim.Adam([Wt, bt], lr=0.02)
    n = Wm.shape[0]
    for _ in range(iters):
        bits = torch.randint(0, 2, (256, 16)).float()
        inp = torch.zeros(256, n); inp[:, :N_IO] = bits * 2 - 1
        x = torch.zeros(256, n)
        for _ in range(T_STEPS):
            x = torch.tanh(x @ (Wt * mask) + bt + inp)
        loss = ((8 + 8 * x[:, n - N_IO:].mean(1) - bits.sum(1)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return (Wt * mask).detach().numpy(), bt.detach().numpy()


def report(name, Wm, bias):
    rng = np.random.default_rng(123)
    bits = rng.integers(0, 2, (5000, 16)).astype(np.float32)
    pred = run(Wm, bias, bits)
    y = bits.sum(1)
    mse = float(np.mean((pred - y) ** 2))
    exact = float(np.mean(np.round(pred) == y))
    within1 = float(np.mean(np.abs(np.round(pred) - y) <= 1))
    print(f'{name:44s} test MSE {mse:6.3f}   exact {exact:5.1%}   within ±1 {within1:5.1%}')


if __name__ == '__main__':
    t0 = time.time()
    best, hist = evolve()
    np.save('evolve_history.npy', np.array(hist))
    Wm, bias = grow(best)
    print(f'\nevolution took {time.time() - t0:.0f}s; best genome grew {Wm.shape[0]} neurons, '
          f'{int((Wm != 0).sum())} wires, {int((Wm < 0).sum())} of them inhibitory\n')
    print('Results on 5,000 new test inputs:')
    report('Always guess 8 (no brain)', np.zeros((40, 40), np.float32), np.zeros(40, np.float32))
    report('Random genome (no evolution)', *grow(random_genome(random.Random(99))))
    report('Evolved genome: ants set everything', Wm, bias)
    report('Same wiring, weights trained by backprop', *backprop_same_wiring(best))
    print('\nbest genome:')
    for s, row in enumerate(best['rule']):
        print(f'  state {s}: ' + '  '.join(f"{e[0]}{e[1]}{e[2]}{({None:'-','neuron':'n','split':'s','stop':'x'})[e[3]]}({e[4]:+.2f})" for e in row))
    print('  colour biases: ' + ', '.join(f'{b:+.2f}' for b in best['bias']))
