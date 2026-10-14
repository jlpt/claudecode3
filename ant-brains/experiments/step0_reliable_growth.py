#!/usr/bin/env python
"""Step 0: does selecting for reliable growth make evolution smoother?

Two conditions, run with the same seeds (so the same starting populations):
  A. single growth   - each genome is grown once, with no noise
  B. reliable growth - each genome is grown --repeats-b times with --noise-b noise
                       in the ants' decisions, and the scores are averaged
The grown brain is scored on counting the 1s in 16 random bits (MSE; always
guessing 8 scores 4.0). After evolution, the best genome of each run is tested:
test error, how much it varies when regrown with noise, and how many of 64
mutated children score within 0.25 of it.

Usage (from the ant-brains/ folder):
  python experiments/step0_reliable_growth.py --quick      # CPU smoke test, a few minutes
  python experiments/step0_reliable_growth.py --estimate   # time one generation, print the run time, exit
  python experiments/step0_reliable_growth.py              # the full experiment

Output goes to results/step0/ (results/step0_quick/ with --quick):
  config.json      every setting, plus device, torch version and git commit
  run_log.txt      everything printed to the console
  speed_check.csv  seconds per generation for each condition, measured before the run
  curves.csv       best and median score of every generation, per condition and seed
  runs.csv         one row per condition and seed: the tests of the best genome
  curves.png       best and median score per generation
  children.png     the per-seed test results, A vs B
  summary.md       tables and the A-vs-B comparison
  checkpoints/     best genome, curves and tests of each finished run
A full run also writes the plain-text summary to results/step0_reliable_growth.txt.
Finished runs are skipped when the script is started again with the same
settings, so an interrupted run can simply be restarted. --fresh redoes them.
"""
import argparse, csv, dataclasses, json, platform, statistics, subprocess, sys, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from antbrains import growlab as gl

STEP = 'step0'
GUESS_MSE = 4.0          # always answering 8
CLOSE = 0.25             # a child is "close" if its test MSE is within this of its parent's
COLORS = {'A': '#2a78d6', 'B': '#eb6834'}


@dataclass
class Settings:
    generations: int = 200
    pop: int = 64
    elite: int = 8
    seed: int = 0            # first seed; runs use seed, seed+1, ...
    n_seeds: int = 3
    repeats_b: int = 4       # growths per genome in condition B
    noise_b: float = 0.3     # noise added to the ants' decision scores in condition B
    sigma: float = 0.02      # size of each mutation nudge
    log_every: int = 25


QUICK = dict(generations=10, pop=16, elite=4, n_seeds=2, log_every=2)

GROWTH = dict(grid_w=gl.GW, grid_h=gl.GH, colours=gl.C, max_ants=gl.A, max_neurons=gl.N,
              growth_steps=gl.STEPS, n_io=gl.N_IO, brain_steps=gl.T_BRAIN,
              dna_memory=gl.MEM, dna_hidden=gl.HID, genome_length=gl.G)


def parse_settings(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--quick', action='store_true', help='tiny run for a CPU smoke test')
    p.add_argument('--estimate', action='store_true', help='time one generation of each condition, print the estimate and exit')
    p.add_argument('--config', type=Path, help='JSON file with any of the settings below')
    p.add_argument('--out', type=Path, help='output folder (default results/step0, or results/step0_quick)')
    p.add_argument('--fresh', action='store_true', help='redo runs that already have a checkpoint')
    for f in dataclasses.fields(Settings):
        quick = f', --quick: {QUICK[f.name]}' if f.name in QUICK else ''
        p.add_argument('--' + f.name.replace('_', '-'), type=f.type, default=None, metavar=f.type.__name__.upper(),
                       help=f'default: {f.default}{quick}')
    args = p.parse_args(argv)

    s = Settings()
    if args.quick:
        s = dataclasses.replace(s, **QUICK)
    if args.config:
        cfg = json.loads(args.config.read_text())
        unknown = set(cfg) - {f.name for f in dataclasses.fields(Settings)}
        if unknown:
            p.error(f'unknown settings in {args.config}: {sorted(unknown)}')
        s = dataclasses.replace(s, **cfg)
    s = dataclasses.replace(s, **{f.name: getattr(args, f.name) for f in dataclasses.fields(Settings)
                                  if getattr(args, f.name) is not None})
    if not 0 < s.elite < s.pop:
        p.error('need 0 < elite < pop')
    return args, s


def conditions(s):
    return {'A': ('A. single growth', 1, 0.0), 'B': ('B. reliable growth', s.repeats_b, s.noise_b)}


class Log:
    def __init__(self, path=None):
        self.f = open(path, 'a', encoding='utf-8') if path else None

    def __call__(self, msg=''):
        print(msg, flush=True)
        if self.f:
            self.f.write(msg + '\n'); self.f.flush()


def sync():
    if gl.DEV.type == 'cuda':
        torch.cuda.synchronize()


def git_info():
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'antbrains', 'experiments'], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + (' (with uncommitted code changes)' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def device_name():
    return torch.cuda.get_device_name(0) if gl.DEV.type == 'cuda' else platform.processor() or 'cpu'


def speed_check(s, log):
    """Seconds per generation for each condition, from one generation of random genomes."""
    gl.grow(gl.random_genomes(2), steps=5); sync()                     # warm-up (CUDA start-up)
    bits = torch.randint(0, 2, (256, 16), device=gl.DEV).float()
    out = {}
    for c, (name, rep, noise) in conditions(s).items():
        genomes = gl.random_genomes(s.pop)
        sync(); t = time.time()
        gl.score(genomes, bits, rep, noise)
        sync(); out[c] = time.time() - t
        log(f'  {name}: {s.pop * rep} brains grown and tested in {out[c]:.1f}s per generation')
    return out


def estimate_minutes(per_gen, s, n_runs):
    # + about 4 generations' worth per run for the tests of the best genome
    return sum(per_gen[c] * (s.generations + 4) * n_runs[c] for c in per_gen) / 60


def run_key(s, c, seed):
    name, rep, noise = conditions(s)[c]
    return dict(condition=c, seed=seed, generations=s.generations, pop=s.pop, elite=s.elite,
                repeats=rep, noise=noise, sigma=s.sigma, test_noise=s.noise_b,   # analyse() regrows with noise_b
                growth=GROWTH, device=gl.DEV.type)


def run_one(s, c, seed, log):
    name, rep, noise = conditions(s)[c]
    if gl.DEV.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    best, curve, med = gl.evolve(s.generations, s.pop, s.elite, rep, noise, sigma=s.sigma, seed=seed,
                                 log_every=s.log_every, log=log)
    analysis = gl.analyse(best, sigma=s.sigma, noise=s.noise_b)
    return dict(key=run_key(s, c, seed), genome=best.cpu(), curve=curve, median=med, analysis=analysis,
                seconds=time.time() - t0,
                peak_mem_mb=torch.cuda.max_memory_allocated() / 2**20 if gl.DEV.type == 'cuda' else None)


def mean_sd(xs):
    m = statistics.mean(xs)
    return (m, statistics.stdev(xs)) if len(xs) > 1 else (m, float('nan'))


def fmt_ms(xs, f='{:.3f}'):
    m, sd = mean_sd(xs)
    return f.format(m) + ('' if sd != sd else ' ± ' + f.format(sd))


def compare(runs, seeds):
    """Paired A-vs-B comparison per seed. Returns lines of text."""
    paired = [(runs[('A', sd)]['analysis'], runs[('B', sd)]['analysis']) for sd in seeds]
    n = len(paired)
    A = [a for a, _ in paired]; B = [b for _, b in paired]

    def wins(k, b_better):
        """(seeds where B is better, seeds where A is better) on measure k."""
        b = sum(b_better(pb[k], pa[k]) for pa, pb in paired)
        a = sum(b_better(pa[k], pb[k]) for pa, pb in paired)
        return b, a

    def line(label, k, b_better, word, f):
        b, a = wins(k, b_better)
        ties = f', {n - a - b} tied' if n - a - b else ''
        return (f'{label}: B {word} than A in {b} of {n} seeds{ties} '
                f'(mean A {f.format(statistics.mean(x[k] for x in A))}, B {f.format(statistics.mean(x[k] for x in B))}).')

    higher, lower = (lambda x, y: x > y), (lambda x, y: x < y)
    lines = [line(f'Children within {CLOSE} of parent', 'kids_close', higher, 'higher', '{:.0%}'),
             line('Median child MSE', 'kids_median', lower, 'lower', '{:.2f}'),
             line('Best-genome test MSE', 'test_mse', lower, 'lower', '{:.3f}')]
    (b_close, a_close), (b_med, a_med) = wins('kids_close', higher), wins('kids_median', lower)
    if b_close == n and b_med == n:
        verdict = 'B beat A on both smoothness measures in every seed.'
    elif a_close == n and a_med == n:
        verdict = 'A beat B on both smoothness measures in every seed.'
    else:
        verdict = 'Mixed: B did not beat A on both smoothness measures in every seed.'
    if n < 3:
        verdict += f' Only {n} seed(s), so this is not enough for a conclusion (PLAN.md asks for at least 3).'
    lines.append('Verdict: ' + verdict)
    return lines


def text_summary(s, runs, seeds, verdict_lines, quick):
    out = [f'settings: generations={s.generations} pop={s.pop} elite={s.elite} seeds={seeds} '
           f'repeats_B={s.repeats_b} noise_B={s.noise_b} sigma={s.sigma} device={gl.DEV} ({device_name()})',
           f'guessing 8 every time = MSE {GUESS_MSE}; 99 = the brain grew fewer than {2 * gl.N_IO} neurons']
    if quick:
        out.append('QUICK MODE: a smoke test of the code, not a result.')
    for c, (name, _, _) in conditions(s).items():
        for sd in seeds:
            r = runs[(c, sd)]['analysis']
            out.append(f"{name:20s} run {sd}: test MSE {r['test_mse']:.3f} | regrown with noise "
                       f"{r['noisy_mean']:.2f} ± {r['noisy_std']:.2f} | children within {CLOSE}: {r['kids_close']:.0%} "
                       f"(median child {r['kids_median']:.2f}) | {r['neurons']} neurons, {r['wires']} wires")
    return '\n'.join(out + [''] + verdict_lines) + '\n'


def write_csvs(out, s, runs, seeds, per_gen):
    with open(out / 'speed_check.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['condition', 'brains_per_generation', 'seconds_per_generation'])
        for c, (_, rep, _) in conditions(s).items():
            w.writerow([c, s.pop * rep, f'{per_gen[c]:.3f}'])
    with open(out / 'curves.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['condition', 'seed', 'generation', 'best', 'median'])
        for c in conditions(s):
            for sd in seeds:
                r = runs[(c, sd)]
                for gen, (b, m) in enumerate(zip(r['curve'], r['median'])):
                    w.writerow([c, sd, gen, f'{b:.5f}', f'{m:.5f}'])
    cols = ['test_mse', 'noisy_mean', 'noisy_std', 'kids_close', 'kids_median', 'neurons', 'wires']
    with open(out / 'runs.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['condition', 'seed'] + cols + ['seconds', 'peak_gpu_mem_mb'])
        for c in conditions(s):
            for sd in seeds:
                r = runs[(c, sd)]
                pm = r['peak_mem_mb']
                w.writerow([c, sd] + [r['analysis'][k] for k in cols]
                           + [f"{r['seconds']:.1f}", '' if pm is None else f'{pm:.0f}'])


def style_axes(ax):
    ax.set_facecolor('#fcfcfb')
    ax.grid(axis='y', color='#e1e0d9', lw=0.8); ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#c3c2b7')
    ax.tick_params(colors='#52514e', labelsize=9)
    ax.xaxis.label.set_color('#52514e'); ax.yaxis.label.set_color('#52514e')
    ax.title.set_color('#0b0b0b')


def mse_axis(ax, values):
    """Log-scale MSE axis that shows everything from the best scores up to 99 (too few neurons)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter
    ax.set_yscale('log')
    lo = min([v for v in values if v > 0] + [1.0]) / 1.5
    ax.set_ylim(lo, 130)
    ax.yaxis.set_major_locator(FixedLocator([t for t in (0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 99) if t >= lo]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.axhline(GUESS_MSE, color='#898781', ls='--', lw=1)
    ax.text(ax.get_xlim()[0], GUESS_MSE * 1.06, ' always guess 8', color='#898781', fontsize=8, va='bottom')
    ax.set_ylabel('error (MSE, log scale, lower is better)')


def plot(out, s, runs, seeds, quick):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    conds = conditions(s)
    tag = ' (quick mode: smoke test only)' if quick else ''
    note = '99 = the brain grew too few neurons to be tested.'

    # ---- learning curves ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor='#fcfcfb')
    for ax, key, title in zip(axes, ['curve', 'median'], ['Best genome each generation', 'Typical (median) genome']):
        for c in conds:
            for sd in seeds:
                ax.plot(runs[(c, sd)][key], color=COLORS[c], lw=1.6, alpha=0.85)
        ax.set_xlabel('generation'); ax.set_title(title + tag, fontsize=11, loc='left')
        mse_axis(ax, [v for r in runs.values() for v in r[key]])
        style_axes(ax)
    axes[0].legend([Line2D([], [], color=COLORS[c], lw=2) for c in conds],
                   [f'{conds[c][0]} ({len(seeds)} seed{"s" if len(seeds) > 1 else ""})' for c in conds],
                   fontsize=9, frameon=False, labelcolor='#0b0b0b')
    fig.text(0.01, 0.005, note, color='#898781', fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / 'curves.png', dpi=130, facecolor=fig.get_facecolor()); plt.close(fig)

    # ---- per-seed test results, A vs B; grey lines join the two runs with the same seed ----
    panels = [('kids_close', f'Children within {CLOSE} of parent'),
              ('kids_median', 'Median child test MSE'),
              ('test_mse', 'Best genome test MSE')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), facecolor='#fcfcfb')
    for ax, (k, title) in zip(axes, panels):
        ax.set_xlim(-0.5, 1.5)
        for sd in seeds:
            v = [runs[(c, sd)]['analysis'][k] for c in conds]
            ax.plot([0, 1], v, color='#c3c2b7', lw=1, zorder=1)
            ax.scatter([0, 1], v, s=60, color=[COLORS[c] for c in conds], edgecolors='#fcfcfb',
                       linewidths=1.5, zorder=2)
        ax.set_xticks([0, 1], [conds[c][0] for c in conds])
        if k == 'kids_close':
            ax.set_ylim(0, 1.02); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_ylabel('share of 64 mutated children')
        else:
            mse_axis(ax, [r['analysis'][k] for r in runs.values()])
        ax.set_title(title, fontsize=11, loc='left')
        style_axes(ax)
    fig.text(0.01, 0.005, f'One dot per seed{tag}; grey lines join A and B runs with the same seed. {note}',
             color='#898781', fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / 'children.png', dpi=130, facecolor=fig.get_facecolor()); plt.close(fig)


def markdown_summary(s, runs, seeds, per_gen, verdict_lines, meta, quick):
    conds = conditions(s)
    md = [f'# Step 0: reliable growth{" (QUICK MODE: smoke test, not a result)" if quick else ""}', '',
          f'Run {meta["finished"]} on {meta["device"]}, code at commit {meta["git"]}.', '',
          '## Settings', '',
          f'generations {s.generations}, population {s.pop}, elite {s.elite}, seeds {seeds}, '
          f'mutation size {s.sigma}; condition B grows each genome {s.repeats_b}× with noise {s.noise_b}.',
          f'Growth: {gl.GW}×{gl.GH} grid, {gl.C} colours, ≤{gl.A} ants, ≤{gl.N} neurons, {gl.STEPS} steps; '
          f'DNA {gl.G} genes. Full list in `config.json`.', '',
          '## Result', ''] + [f'- {line}' for line in verdict_lines] + [
          '', f'Always guessing 8 scores MSE {GUESS_MSE}. 99 means the brain grew fewer than {2 * gl.N_IO} neurons.', '',
          '| condition | seed | test MSE | regrown with noise | children within 0.25 | median child | neurons | wires | minutes |',
          '|---|---|---|---|---|---|---|---|---|']
    for c in conds:
        for sd in seeds:
            r = runs[(c, sd)]; a = r['analysis']
            md.append(f"| {conds[c][0]} | {sd} | {a['test_mse']:.3f} | {a['noisy_mean']:.2f} ± {a['noisy_std']:.2f} | "
                      f"{a['kids_close']:.0%} | {a['kids_median']:.2f} | {a['neurons']} | {a['wires']} | {r['seconds'] / 60:.1f} |")
    md += ['', '| condition | test MSE | children within 0.25 | median child |', '|---|---|---|---|']
    for c in conds:
        A = [runs[(c, sd)]['analysis'] for sd in seeds]
        md.append(f"| {conds[c][0]} | {fmt_ms([a['test_mse'] for a in A])} | "
                  f"{fmt_ms([a['kids_close'] * 100 for a in A], '{:.0f}')}% | {fmt_ms([a['kids_median'] for a in A], '{:.2f}')} |")
    md += ['', '(mean ± standard deviation over seeds)', '',
           '![learning curves](curves.png)', '', '![tests of the best genomes](children.png)', '',
           '## Timing', '',
           '| condition | brains per generation | seconds per generation (speed check) | total minutes (all seeds) |',
           '|---|---|---|---|']
    for c, (name, rep, _) in conds.items():
        md.append(f"| {name} | {s.pop * rep} | {per_gen[c]:.2f} | {sum(runs[(c, sd)]['seconds'] for sd in seeds) / 60:.1f} |")
    pm = [runs[k]['peak_mem_mb'] for k in runs if runs[k]['peak_mem_mb'] is not None]
    if pm:
        md += ['', f'Peak GPU memory: {max(pm):.0f} MB.']
    return '\n'.join(md) + '\n'


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')    # a Windows console without '±' must not crash a long run
    args, s = parse_settings(argv)
    seeds = list(range(s.seed, s.seed + s.n_seeds))
    conds = conditions(s)

    if args.estimate:
        log = Log()
        log(f'device: {gl.DEV} ({device_name()})')
        per_gen = speed_check(s, log)
        log(f'Estimated time for {s.n_seeds} seed(s) x {s.generations} generations x 2 conditions: '
            f'about {estimate_minutes(per_gen, s, {c: s.n_seeds for c in conds}):.0f} min')
        return

    out = args.out or ROOT / 'results' / (STEP + ('_quick' if args.quick else ''))
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    log = Log(out / 'run_log.txt')
    started = datetime.now(timezone.utc).isoformat(timespec='seconds')
    log(f'==== {STEP} started {started} ====')
    log(f'device: {gl.DEV} ({device_name()}), torch {torch.__version__}, python {platform.python_version()}')
    log('settings: ' + json.dumps(dataclasses.asdict(s)))

    # ---- which runs are already done? ----
    runs, todo = {}, []
    for c in conds:
        for sd in seeds:
            ck = out / 'checkpoints' / f'{c}_seed{sd}.pt'
            if ck.exists() and not args.fresh:
                r = torch.load(ck, map_location='cpu', weights_only=True)
                if r['key'] == run_key(s, c, sd):
                    runs[(c, sd)] = r; continue
                log(f'  {ck.name} was made with different settings; redoing it')
            todo.append((c, sd))
    if runs:
        log(f'{len(runs)} run(s) already finished, skipping: {sorted(runs)}')

    log('speed check:')
    per_gen = speed_check(s, log)
    n_todo = {c: sum(1 for cc, _ in todo if cc == c) for c in conds}
    log(f'estimated time for the {len(todo)} remaining run(s): about {estimate_minutes(per_gen, s, n_todo):.0f} min')

    meta = dict(step=STEP, quick=args.quick, started=started, device=f'{gl.DEV} ({device_name()})',
                torch=torch.__version__, python=platform.python_version(), git=git_info())
    (out / 'config.json').write_text(json.dumps(dict(settings=dataclasses.asdict(s), seeds=seeds, growth=GROWTH,
                                                     conditions={c: dict(name=n, repeats=r, noise=z) for c, (n, r, z) in conds.items()},
                                                     **meta), indent=2) + '\n')

    # ---- run ----
    for c, sd in todo:
        log(f'{conds[c][0]}, run {sd}')
        r = run_one(s, c, sd, log)
        torch.save(r, out / 'checkpoints' / f'{c}_seed{sd}.pt')
        runs[(c, sd)] = r
        a = r['analysis']
        log(f'  -> test MSE {a["test_mse"]:.3f}; regrown with noise {a["noisy_mean"]:.2f} ± {a["noisy_std"]:.2f}; '
            f'{a["kids_close"]:.0%} of mutated children within {CLOSE} of parent; median child {a["kids_median"]:.2f} '
            f'({r["seconds"] / 60:.1f} min)')

    # ---- outputs ----
    meta['finished'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    verdict = compare(runs, seeds)
    write_csvs(out, s, runs, seeds, per_gen)
    plot(out, s, runs, seeds, args.quick)
    (out / 'summary.md').write_text(markdown_summary(s, runs, seeds, per_gen, verdict, meta, args.quick), encoding='utf-8')
    txt = text_summary(s, runs, seeds, verdict, args.quick)
    if not args.quick and args.out is None:
        (ROOT / 'results' / f'{STEP}_reliable_growth.txt').write_text(txt, encoding='utf-8')
    log('')
    log(txt)
    log(f'All output is in {out}')


if __name__ == '__main__':
    main()
