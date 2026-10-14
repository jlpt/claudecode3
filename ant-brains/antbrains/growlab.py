# ==== Neural-DNA ants, batched for the GPU ====
# Every world (one grown brain) lives in the same tensors, so all ants in all
# brains of a generation move together in one operation per growth step.
#
# This is the code from ant_brains_gpu.ipynb (kept in archive/), unchanged
# except that evolve() can report each generation through a callback.
import math, time
import torch
import torch.nn.functional as F

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- world and brain settings ----
GW, GH = 120, 90            # grid width, height (wraps around)
C = 4                       # colours
A = 48                      # max ants per world
N = 96                      # max neurons per brain
STEPS = 1500                # growth steps
N_IO = 16                   # input neurons / output neurons
T_BRAIN = 6                 # thinking steps when the brain runs

# ---- the DNA: one small network shared by every ant ----
MEM, HID = 4, 16
IN = 4 * C + 2 + MEM + 1                       # colours under/ahead/left/right, on-neuron, progress, memory, bias
OUT = 4 + C + 4 + 1 + 1 + MEM                  # turn, write, action, wire weight, neuron bias, new memory
SHAPES = [(IN, HID), (HID,), (HID, OUT), (OUT,)]
G = sum(math.prod(s) for s in SHAPES)
ACT_NONE_IDX = IN * HID + HID + HID * OUT + (4 + C)    # position of the "no action" bias in the genome

DX = torch.tensor([0, 1, 0, -1], device=DEV)
DY = torch.tensor([-1, 0, 1, 0], device=DEV)
TURN_DELTA = torch.tensor([3, 1, 0, 2], device=DEV)    # L, R, straight, U-turn
MIRROR_DELTA = torch.tensor([1, 3, 2, 0], device=DEV)  # a split child turns the mirror way


def unpack(gen):
    """gen: [B, G] -> W1 [B,IN,HID], b1 [B,HID], W2 [B,HID,OUT], b2 [B,OUT]"""
    out, i = [], 0
    for s in SHAPES:
        n = math.prod(s)
        out.append(gen[:, i:i + n].reshape(gen.shape[0], *s)); i += n
    return out


def random_genomes(P, g=None):
    x = torch.randn(P, G, device=DEV, generator=g) * 0.5
    x[:, ACT_NONE_IDX] += 1.5            # start out mostly doing nothing special
    return x


@torch.no_grad()
def grow(genomes, noise=0.0, g=None, steps=None):
    """Grow one brain per genome. Returns W [B,N,N], bias [B,N], n_neurons [B]."""
    steps = steps or STEPS
    B = genomes.shape[0]
    W1, b1, W2, b2 = unpack(genomes)
    grid = torch.zeros(B, GH, GW, dtype=torch.long, device=DEV)
    nat = torch.full((B, GH, GW), -1, dtype=torch.long, device=DEV)     # neuron id at each square
    Wm = torch.zeros(B, N, N, device=DEV)
    nbias = torch.zeros(B, N, device=DEV)
    ncount = torch.ones(B, dtype=torch.long, device=DEV)

    ax = torch.full((B, A), GW // 2, dtype=torch.long, device=DEV)
    ay = torch.full((B, A), GH // 2, dtype=torch.long, device=DEV)
    ad = torch.zeros(B, A, dtype=torch.long, device=DEV)
    ah = torch.zeros(B, A, MEM, device=DEV)
    alast = torch.zeros(B, A, dtype=torch.long, device=DEV)
    alive = torch.zeros(B, A, dtype=torch.bool, device=DEV); alive[:, 0] = True
    nat[:, GH // 2, GW // 2] = 0                                        # seed neuron

    bidx = torch.arange(B, device=DEV)[:, None].expand(B, A)
    slots = torch.arange(A, device=DEV)[None, :].expand(B, A)
    ones = torch.ones(B, A, 1, device=DEV)

    def look(d):
        return grid[bidx, (ay + DY[d]) % GH, (ax + DX[d]) % GW]

    for step in range(steps):
        # ---- sense ----
        here = nat[bidx, ay, ax]
        X = torch.cat([F.one_hot(grid[bidx, ay, ax], C), F.one_hot(look(ad), C),
                       F.one_hot(look((ad + 3) % 4), C), F.one_hot(look((ad + 1) % 4), C)], -1).float()
        X = torch.cat([X, (here >= 0).float()[..., None], ones * (step / steps), ah, ones], -1)

        # ---- decide (the DNA network) ----
        hid = torch.tanh(torch.bmm(X, W1) + b1[:, None, :])
        out = torch.bmm(hid, W2) + b2[:, None, :]
        if noise > 0:
            out[..., :8 + C] += noise * torch.randn(out[..., :8 + C].shape, device=DEV, generator=g)
        turn = out[..., 0:4].argmax(-1)
        write = out[..., 4:4 + C].argmax(-1)
        act = out[..., 4 + C:8 + C].argmax(-1)              # 0 none, 1 neuron, 2 split, 3 stop
        wgt = torch.tanh(out[..., 8 + C])
        nb = torch.tanh(out[..., 9 + C])
        hnew = torch.tanh(out[..., 10 + C:])

        # ---- act ----
        m = alive
        grid[bidx[m], ay[m], ax[m]] = write[m]

        arr = alive & (here >= 0) & (here != alast)         # walked onto another neuron: lay a wire
        Wm.index_put_((bidx[arr], alast[arr], here[arr]), wgt[arr], accumulate=True)
        alast = torch.where(arr, here, alast)

        req = alive & (act == 1) & (here < 0)               # drop a new neuron here
        newid = ncount[:, None] + req.long().cumsum(1) - 1
        ok = req & (newid < N)
        nat[bidx[ok], ay[ok], ax[ok]] = newid[ok]
        nbias[bidx[ok], newid[ok]] = nb[ok]
        Wm.index_put_((bidx[ok], alast[ok], newid[ok]), wgt[ok], accumulate=True)
        alast = torch.where(ok, newid, alast)
        ncount = ncount + ok.sum(1)

        alive = alive & ~(act == 3)                          # stop

        # split: children fill free ant slots
        sreq = alive & (act == 2)
        free = ~alive
        slot_of_rank = torch.full((B, A), -1, dtype=torch.long, device=DEV)
        frank = free.long().cumsum(1) - 1
        slot_of_rank[bidx[free], frank[free]] = slots[free]
        srank = (sreq.long().cumsum(1) - 1).clamp(min=0)
        target = slot_of_rank.gather(1, srank)
        sok = sreq & (target >= 0)
        cd = (ad + MIRROR_DELTA[turn]) % 4
        cx, cy = (ax + DX[cd]) % GW, (ay + DY[cd]) % GH

        # parents move
        nd = (ad + TURN_DELTA[turn]) % 4
        ad = torch.where(alive, nd, ad)
        ax = torch.where(alive, (ax + DX[ad]) % GW, ax)
        ay = torch.where(alive, (ay + DY[ad]) % GH, ay)
        ah = torch.where(alive[..., None], hnew, ah)

        # children appear
        tb, ts = bidx[sok], target[sok]
        ad[tb, ts] = cd[sok]; ax[tb, ts] = cx[sok]; ay[tb, ts] = cy[sok]
        ah[tb, ts] = hnew[sok]; alast[tb, ts] = alast[sok]; alive[tb, ts] = True

    Wm.diagonal(dim1=1, dim2=2).zero_()
    return Wm, nbias, ncount.clamp(max=N)


@torch.no_grad()
def brain_mse(Wm, bias, n, bits):
    """Run every brain on the same questions. bits [M,16] (0/1). Returns MSE per brain [B]."""
    B, M = Wm.shape[0], bits.shape[0]
    inp = torch.zeros(B, M, N, device=DEV)
    inp[:, :, :N_IO] = bits * 2 - 1
    x = torch.zeros(B, M, N, device=DEV)
    for _ in range(T_BRAIN):
        x = torch.tanh(torch.bmm(x, Wm) + bias[:, None, :] + inp)
    out_idx = (n[:, None] - N_IO + torch.arange(N_IO, device=DEV)).clamp(min=0)   # last-grown neurons
    outs = x.gather(2, out_idx[:, None, :].expand(B, M, N_IO))
    pred = 8 + 8 * outs.mean(-1)
    mse = ((pred - bits.sum(1)) ** 2).mean(1)
    return torch.where(n >= 2 * N_IO, mse, torch.full_like(mse, 99.0))


def score(genomes, bits, repeats=1, noise=0.0, g=None, chunk=512):
    """Average MSE over `repeats` noisy growths of each genome."""
    P = genomes.shape[0]
    rep = genomes.repeat_interleave(repeats, 0)
    res = []
    for i in range(0, rep.shape[0], chunk):
        W, b, n = grow(rep[i:i + chunk], noise=noise, g=g)
        res.append(brain_mse(W, b, n, bits))
    return torch.cat(res).view(P, repeats).mean(1)


def evolve(generations, pop, elite, repeats, noise, sigma=0.02, seed=0, log_every=25,
           on_generation=None, log=None):
    """Evolve a population. on_generation(gen, best, median, seconds) is called every generation."""
    log = log or (lambda msg: print(msg, flush=True))
    g = torch.Generator(device=DEV).manual_seed(seed)
    popn = random_genomes(pop, g)
    best_curve, med_curve = [], []
    t0 = time.time()
    for gen in range(generations):
        bits = torch.randint(0, 2, (256, 16), device=DEV, generator=g).float()
        s = score(popn, bits, repeats, noise, g)
        order = s.argsort()
        best_curve.append(s[order[0]].item()); med_curve.append(s.median().item())
        if on_generation is not None:
            on_generation(gen, best_curve[-1], med_curve[-1], time.time() - t0)
        if gen % log_every == 0 or gen == generations - 1:
            log(f'  gen {gen:4d}  best {best_curve[-1]:.3f}  median {med_curve[-1]:.3f}  ({time.time() - t0:.0f}s)')
        parents = popn[order[:elite]]
        pick = torch.randint(0, elite, (pop - elite,), device=DEV, generator=g)
        kids = parents[pick] + sigma * torch.randn(pop - elite, G, device=DEV, generator=g)
        popn = torch.cat([parents, kids])
    return popn[0], best_curve, med_curve


def analyse(best, sigma=0.02, noise=0.3, seed=123):
    """Test the best genome: accuracy, how reliably it grows, and how smooth its mutations are."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    test = torch.randint(0, 2, (2000, 16), device=DEV, generator=g).float()
    clean = score(best[None], test).item()
    noisy = score(best[None].repeat(16, 1), test, noise=noise, g=g)          # 16 noisy growths
    kids = best[None] + sigma * torch.randn(64, G, device=DEV, generator=g)
    kid_scores = score(kids, test)
    W, b, n = grow(best[None])
    return dict(
        test_mse=clean,
        noisy_mean=noisy.mean().item(), noisy_std=noisy.std().item(),
        kids_close=(kid_scores - clean).abs().lt(0.25).float().mean().item(),
        kids_median=kid_scores.median().item(),
        neurons=int(n[0]), wires=int((W[0] != 0).sum()))
