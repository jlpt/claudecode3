"""Exports a recorded run as one self-contained HTML page: the arm and ball, the brain lighting up spike by spike, a
spike raster of every neuron and the muscle activations, with a trial picker and a time scrubber."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .brain import TRACTS, BrainSubstrate

# Schematic sagittal layout of the regions on the brain map (fractions of the canvas). Posterior is left.
DRAW_POS = {
    "VIS": (0.095, 0.24), "PPC": (0.29, 0.07), "S1": (0.49, 0.02), "M1": (0.69, 0.02), "PMd": (0.89, 0.07),
    "CB": (0.24, 0.66), "SC": (0.60, 0.70), "DRG": (0.89, 0.58),
}
BOX_W = 0.175
DOTS_W = 0.68 * BOX_W  # dots use the left part of a box, row labels the right
ROW_H = 0.045
GRID_ROW_H = 0.026
HEADER_H = 0.055
RASTER_ORDER = ("VIS", "DRG", "S1", "PPC", "PMd", "M1", "CB", "SC")


def _layout(sub: BrainSubstrate):
    pos = np.zeros((sub.N, 2))
    boxes = []
    for key, (cx, top) in DRAW_POS.items():
        pops = sorted((p for p in sub.populations.values() if p.region == key), key=lambda p: -p.coords[0, 2])
        y = top + HEADER_H
        rows = []
        for p in pops:
            idx = np.arange(p.start, p.start + p.n)
            if "rf_x" in p.meta:  # retinotopic map: draw it as the grid it is
                zs, as_ = np.unique(p.coords[:, 0]), np.unique(p.coords[:, 1])
                dx = DOTS_W / max(len(zs), 1)
                for i, n in zip(range(p.n), idx):
                    c = np.searchsorted(zs, p.coords[i, 0]); r = len(as_) - 1 - np.searchsorted(as_, p.coords[i, 1])
                    pos[n] = (cx - BOX_W / 2 + 0.012 + (c + 0.5) * dx, y + (r + 0.5) * GRID_ROW_H)
                h = len(as_) * GRID_ROW_H + 0.012
            else:
                order = np.lexsort((p.coords[:, 1], p.coords[:, 0]))
                dx = min(DOTS_W / p.n, 0.018)
                x0 = cx - BOX_W / 2 + 0.012 + (DOTS_W - dx * p.n) / 2
                for k, i in enumerate(order):
                    pos[idx[i]] = (x0 + (k + 0.5) * dx, y + ROW_H / 2)
                h = ROW_H
            rows.append({"label": p.key.split(".")[1], "y": y + h / 2})
            y += h
        boxes.append({"key": key, "name": sub.regions[key].name, "x0": cx - BOX_W / 2, "x1": cx + BOX_W / 2,
                      "y0": top, "y1": y + 0.01, "rows": rows})
    return pos, boxes


def build_page(sub: BrainSubstrate, cfg, rec, results, trials, meta: dict, expressed: np.ndarray,
               standalone: bool = True) -> str:
    """standalone adds the doctype and meta tags a local file needs; leave it off for hosts that add their own."""
    env = cfg.env
    pos, boxes = _layout(sub)
    order = [n for key in RASTER_ORDER for n in range(sub.N) if sub.region_keys[sub.region_of[n]] == key]
    rank = np.empty(sub.N, int)
    rank[order] = np.arange(sub.N)
    bands, start = [], 0
    for key in RASTER_ORDER:
        n = int((np.array([sub.region_keys[r] for r in sub.region_of]) == key).sum())
        bands.append({"key": key, "start": start, "n": n})
        start += n

    # Region-to-region tracts of this brain: expressed synapse count per (source region, target region).
    links = {}
    for k in np.nonzero(expressed)[0]:
        a, b = sub.region_keys[sub.region_of[sub.pre[k]]], sub.region_keys[sub.region_of[sub.post[k]]]
        if a != b:
            links[(a, b)] = links.get((a, b), 0) + 1
    per_tract = np.bincount(sub.tract[expressed], minlength=len(TRACTS))
    total_tract = np.bincount(sub.tract, minlength=len(TRACTS))

    steps_per_frame = int(round(rec.frame_dt / env.dt))
    trial_data = []
    for t, (bx, tx) in enumerate(trials):
        sel = rec.spike_batch == t
        st = rec.spike_step[sel] // steps_per_frame
        nn = rec.spike_neuron[sel]
        trial_data.append({
            "ball_x": bx, "target_x": tx,
            "error": float(results["error"][0, t]), "success": bool(results["success"][0, t]),
            "fitness": float(results["fitness"][0, t]), "held_out": t >= meta["n_train"],
            "q": np.round(rec.q[:, t], 4).tolist(),
            "ball": np.round(rec.ball[:, t], 4).tolist(),
            "act": np.round(rec.act[:, t], 3).tolist(),
            "spikes": np.stack([st, nn], 1).ravel().astype(int).tolist(),
        })
    data = {
        "meta": meta,
        "env": {"shoulder_height": env.shoulder_height, "l1": env.link_length[0], "l2": env.link_length[1],
                "ball_radius": env.ball_radius, "link_radius": env.link_radius, "wall_x": env.wall_x,
                "tolerance": env.target_tolerance, "duration": env.duration},
        "frame_dt": rec.frame_dt,
        "neurons": {"x": np.round(pos[:, 0], 4).tolist(), "y": np.round(pos[:, 1], 4).tolist(),
                    "inh": sub.inhibitory.astype(int).tolist(), "rank": rank.tolist()},
        "boxes": boxes, "bands": bands,
        "links": [{"src": a, "dst": b, "n": n} for (a, b), n in links.items()],
        "tracts": [{"name": t, "expressed": int(e), "total": int(c)} for t, e, c in zip(TRACTS, per_tract, total_tract)],
        "trials": trial_data,
    }
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.replace("__DATA__", blob)
    return STANDALONE_HEAD + page if standalone else page


def write_page(path: Path, *args, **kwargs) -> None:
    Path(path).write_text(build_page(*args, **kwargs), encoding="utf-8")


STANDALONE_HEAD = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
"""

TEMPLATE = r"""<title>Spiking Limb Replay</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a bench console. The arena and its transport sit on the left; the brain map, spike raster and
   muscle traces are stacked on the right like channels on a recording rig. One column on phones. */
:root {
  --bg: #eef2f0; --panel: #f9fbfa; --ink: #15201c; --muted: #5a6a64; --rule: #d2dcd8; --grid: #e3eae7;
  --exc: #bf5410; --inh: #3152c4; --ball: #cf3b47; --target: #2b8a56; --arm: #24312c;
  --f-display: "Newsreader", "Iowan Old Style", Georgia, serif;
  --f-body: "Public Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f1412; --panel: #151c19; --ink: #e2eae6; --muted: #91a29b; --rule: #2a3631; --grid: #1d2622;
  --exc: #f2893c; --inh: #84a0ff; --ball: #ff6d77; --target: #4fc78b; --arm: #d6e0db; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0f1412; --panel: #151c19; --ink: #e2eae6; --muted: #91a29b; --rule: #2a3631; --grid: #1d2622;
  --exc: #f2893c; --inh: #84a0ff; --ball: #ff6d77; --target: #4fc78b; --arm: #d6e0db; color-scheme: dark; }

html, body { background: var(--bg); color: var(--ink); }
body { margin: 0; font: 15px/1.5 var(--f-body); }
.wrap { max-width: 1320px; margin: 0 auto; padding-block: 28px 40px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 20px; }
header { display: grid; gap: 6px; }
h1 { font: 600 clamp(28px, 4vw, 40px)/1.1 var(--f-display); margin: 0; text-wrap: balance; letter-spacing: -0.01em; }
.lede { margin: 0; color: var(--muted); max-width: 70ch; }
.stats { display: flex; flex-wrap: wrap; gap: 8px 22px; margin-top: 8px; font: 13px/1.4 var(--f-mono);
  font-variant-numeric: tabular-nums; color: var(--muted); }
.stats b { color: var(--ink); font-weight: 500; }
.console { display: grid; grid-template-columns: minmax(0, 5fr) minmax(0, 6fr); gap: 20px; align-items: start; }
@media (max-width: 920px) { .console { grid-template-columns: minmax(0, 1fr); } }
.panel { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 14px; display: grid; gap: 10px; min-width: 0; }
.col { display: grid; gap: 20px; min-width: 0; }
.eyebrow { font: 500 11px/1 var(--f-mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
.panel-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; font: 12px var(--f-mono); color: var(--muted); }
.legend i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }
canvas { display: block; width: 100%; max-width: 100%; }
#arena { aspect-ratio: 1.25; }
#brain { aspect-ratio: 1.55; }
#raster { aspect-ratio: 2.6; }
#muscles { aspect-ratio: 4.2; }
.transport { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
button { font: 500 13px var(--f-body); color: var(--ink); background: var(--bg); border: 1px solid var(--rule);
  border-radius: 4px; padding: 6px 12px; cursor: pointer; }
button:hover { border-color: var(--muted); }
button:focus-visible, input:focus-visible { outline: 2px solid var(--inh); outline-offset: 2px; }
button[aria-pressed="true"] { background: var(--ink); color: var(--panel); border-color: var(--ink); }
#play { min-width: 74px; }
#scrub { flex: 1 1 160px; min-width: 0; accent-color: var(--exc); }
.clock { font: 500 13px var(--f-mono); font-variant-numeric: tabular-nums; min-width: 64px; text-align: right; }
.speeds { display: flex; gap: 4px; }
.speeds button { padding: 6px 8px; font-family: var(--f-mono); font-size: 12px; }
.trials { display: grid; gap: 6px; }
.trial-group { display: flex; flex-wrap: wrap; gap: 6px; }
.trial { display: inline-flex; align-items: center; gap: 8px; font: 12px var(--f-mono); font-variant-numeric: tabular-nums; }
.pill { font: 500 10px/1 var(--f-mono); letter-spacing: 0.06em; text-transform: uppercase; padding: 3px 5px; border-radius: 3px;
  border: 1px solid currentColor; }
.pill.ok { color: var(--target); } .pill.miss { color: var(--muted); }
button[aria-pressed="true"] .pill.ok, button[aria-pressed="true"] .pill.miss { color: var(--panel); }
.readout { font: 12px/1.6 var(--f-mono); color: var(--muted); font-variant-numeric: tabular-nums; }
.readout b { color: var(--ink); font-weight: 500; }
table { border-collapse: collapse; font: 12px var(--f-mono); font-variant-numeric: tabular-nums; width: 100%; }
td, th { padding: 4px 8px 4px 0; text-align: left; border-bottom: 1px solid var(--grid); }
th { color: var(--muted); font-weight: 500; }
td.num { text-align: right; }
.bar { height: 6px; background: var(--grid); border-radius: 3px; overflow: hidden; min-width: 60px; }
.bar span { display: block; height: 100%; background: var(--muted); }
.tablewrap { overflow-x: auto; }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">HyperNEAT · Izhikevich spiking network · two-link arm</div>
    <h1>Spiking limb replay</h1>
    <p class="lede">A brain wired like the primate reaching circuit, painted by an evolved CPPN, drives a double-pendulum arm to push a ball onto the green target. Pick a trial, then watch each region fire as the arm moves.</p>
    <div class="stats" id="stats"></div>
  </header>

  <div class="console">
    <div class="col">
      <section class="panel" aria-label="Arena">
        <div class="panel-head"><span class="eyebrow">Arena · side view</span>
          <span class="legend"><span><i style="background:var(--ball)"></i>ball</span><span><i style="background:var(--target)"></i>target ± tolerance</span></span></div>
        <canvas id="arena" role="img" aria-label="Arm, ball and target"></canvas>
        <div class="transport">
          <button id="play" type="button" aria-pressed="false">Play</button>
          <input id="scrub" type="range" min="0" value="0" aria-label="Time">
          <span class="clock" id="clock">0.00 s</span>
          <span class="speeds" role="group" aria-label="Playback speed">
            <button type="button" data-speed="0.25" id="sp025">¼×</button>
            <button type="button" data-speed="0.5" id="sp05">½×</button>
            <button type="button" data-speed="1" id="sp1">1×</button>
          </span>
        </div>
        <div class="readout" id="readout"></div>
      </section>
      <section class="panel" aria-label="Trials">
        <span class="eyebrow">Trials</span>
        <div class="trials" id="trials"></div>
      </section>
      <section class="panel" aria-label="Synapses per tract">
        <span class="eyebrow">Synapses the CPPN expressed, per tract</span>
        <div class="tablewrap"><table id="tracts"></table></div>
      </section>
    </div>

    <div class="col">
      <section class="panel" aria-label="Brain map">
        <div class="panel-head"><span class="eyebrow">Brain map · posterior left, dorsal up</span>
          <span class="legend"><span><i style="background:var(--exc)"></i>excitatory spike</span><span><i style="background:var(--inh)"></i>inhibitory spike</span></span></div>
        <canvas id="brain" role="img" aria-label="Brain regions with neurons that light up when they spike"></canvas>
      </section>
      <section class="panel" aria-label="Spike raster">
        <span class="eyebrow">Spike raster · every neuron, whole trial</span>
        <canvas id="raster" role="img" aria-label="Spike raster grouped by brain region"></canvas>
      </section>
      <section class="panel" aria-label="Muscle activation">
        <div class="panel-head"><span class="eyebrow">Muscle activation</span>
          <span class="legend"><span>— flexor</span><span>- - extensor</span></span></div>
        <canvas id="muscles" role="img" aria-label="Activation of the four muscles over time"></canvas>
      </section>
    </div>
  </div>
</div>

<script type="application/json" id="replay-data">__DATA__</script>
<script>
(() => {
const D = JSON.parse(document.getElementById("replay-data").textContent);
const F = D.trials[0].q.length;
const STEPS = Math.round(D.frame_dt * 1000);
let trial = 0, frame = 0, playing = false, speed = 1, acc = 0, last = null;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let T = {};
function readTokens() { for (const k of ["bg","panel","ink","muted","rule","grid","exc","inh","ball","target","arm"]) T[k] = css("--" + k); }
readTokens();
const fmt = (x, d = 2) => (x < 0 ? "−" : "") + Math.abs(x).toFixed(d);
const cm = x => fmt(x * 100, 0);

// Spikes of each trial bucketed by frame.
for (const tr of D.trials) {
  const s = tr.spikes, start = new Int32Array(F + 1);
  for (let i = 0; i < s.length; i += 2) start[s[i] + 1]++;
  for (let f = 0; f < F; f++) start[f + 1] += start[f];
  tr.frameStart = start;
}

// ---- header and lists
const M = D.meta;
const train = D.trials.filter(t => !t.held_out), held = D.trials.filter(t => t.held_out);
const okCount = a => a.filter(t => t.success).length;
document.getElementById("stats").innerHTML =
  `<span>generation <b>${M.generation}</b></span><span>fitness <b>${M.fitness.toFixed(3)}</b></span>` +
  `<span>training trials on target <b>${okCount(train)}/${train.length}</b></span>` +
  (held.length ? `<span>held-out on target <b>${okCount(held)}/${held.length}</b></span>` : "") +
  `<span>neurons <b>${M.neurons}</b></span><span>synapses <b>${M.synapses.toLocaleString()}</b></span>` +
  `<span>CPPN <b>${M.cppn_hidden}</b> hidden nodes, <b>${M.cppn_conns}</b> links</span>`;

const trialsEl = document.getElementById("trials");
function trialButton(tr, i) {
  const b = document.createElement("button");
  b.type = "button"; b.className = "trial"; b.id = "trial-" + i;
  const err = tr.error * 100;
  b.innerHTML = `<span>ball ${cm(tr.ball_x)} → ${cm(tr.target_x)} cm</span>` +
    `<span class="pill ${tr.success ? "ok" : "miss"}">${tr.success ? "on target" : err.toFixed(0) + " cm off"}</span>`;
  b.addEventListener("click", () => { trial = i; frame = 0; acc = 0; drawStatic(); syncUI(); render(); });
  return b;
}
for (const [label, list] of [["Training", train], ["Held out", held]]) {
  if (!list.length) continue;
  const g = document.createElement("div"); g.className = "trials";
  g.innerHTML = `<span class="readout">${label}</span>`;
  const row = document.createElement("div"); row.className = "trial-group";
  for (const tr of list) row.appendChild(trialButton(tr, D.trials.indexOf(tr)));
  g.appendChild(row); trialsEl.appendChild(g);
}
const tractsEl = document.getElementById("tracts");
tractsEl.innerHTML = "<tr><th>tract</th><th class='num'>expressed</th><th class='num'>possible</th><th></th></tr>" +
  D.tracts.map(t => `<tr><td>${t.name.replace("_", " ")}</td><td class="num">${t.expressed}</td><td class="num">${t.total}</td>` +
  `<td><div class="bar"><span style="width:${(100 * t.expressed / Math.max(t.total, 1)).toFixed(0)}%"></span></div></td></tr>`).join("");

// ---- canvases
const cv = {};
for (const id of ["arena", "brain", "raster", "muscles"]) {
  const c = document.getElementById(id);
  cv[id] = { c, ctx: c.getContext("2d"), w: 0, h: 0, layer: document.createElement("canvas") };
}
function fit() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  for (const v of Object.values(cv)) {
    const r = v.c.getBoundingClientRect();
    v.w = Math.max(10, r.width); v.h = Math.max(10, r.height);
    for (const c of [v.c, v.layer]) { c.width = Math.round(v.w * dpr); c.height = Math.round(v.h * dpr); }
    v.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    v.layer.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  drawStatic(); render();
}

function drawStatic() {
  const tr = D.trials[trial];
  // Brain map: boxes, tract links, row labels.
  {
    const v = cv.brain, g = v.layer.getContext("2d"), W = v.w, H = v.h;
    g.clearRect(0, 0, W, H);
    const box = Object.fromEntries(D.boxes.map(b => [b.key, b]));
    const maxN = Math.max(1, ...D.links.map(l => l.n));
    g.strokeStyle = T.muted;
    for (const l of D.links) {
      const a = box[l.src], b = box[l.dst];
      const ax = (a.x0 + a.x1) / 2 * W, ay = (a.y0 + a.y1) / 2 * H, bx = (b.x0 + b.x1) / 2 * W, by = (b.y0 + b.y1) / 2 * H;
      g.globalAlpha = 0.12 + 0.3 * l.n / maxN; g.lineWidth = 0.6 + 2.4 * l.n / maxN;
      const mx = (ax + bx) / 2 + (by - ay) * 0.12, my = (ay + by) / 2 - (bx - ax) * 0.12;
      g.beginPath(); g.moveTo(ax, ay); g.quadraticCurveTo(mx, my, bx, by); g.stroke();
    }
    g.globalAlpha = 1;
    for (const b of D.boxes) {
      const x = b.x0 * W, y = b.y0 * H, w = (b.x1 - b.x0) * W, h = (b.y1 - b.y0) * H;
      g.fillStyle = T.panel; g.strokeStyle = T.rule; g.lineWidth = 1;
      g.beginPath(); g.roundRect(x, y, w, h, 4); g.fill(); g.stroke();
      g.fillStyle = T.ink; g.font = `600 ${Math.max(10, W * 0.019)}px ${css("--f-body")}`; g.textBaseline = "top";
      g.fillText(b.key, x + 6, y + 5);
      g.fillStyle = T.muted; g.font = `${Math.max(8, W * 0.012)}px ${css("--f-mono")}`;
      for (const r of b.rows) { g.textBaseline = "middle"; g.textAlign = "right"; g.fillText(r.label, x + w - 4, r.y * H); g.textAlign = "left"; }
    }
    g.fillStyle = T.rule;
    for (let n = 0; n < D.neurons.x.length; n++) {
      g.beginPath(); g.arc(D.neurons.x[n] * W, D.neurons.y[n] * H, Math.max(1.4, W * 0.0032), 0, 7); g.fill();
    }
  }
  // Raster.
  {
    const v = cv.raster, g = v.layer.getContext("2d"), W = v.w, H = v.h, lab = 34;
    g.clearRect(0, 0, W, H);
    const N = D.neurons.x.length, rowH = H / N, xs = (W - lab) / F;
    g.font = `${Math.max(8, Math.min(11, W * 0.016))}px ${css("--f-mono")}`; g.textBaseline = "middle";
    D.bands.forEach((b, i) => {
      if (i % 2) { g.fillStyle = T.grid; g.fillRect(lab, b.start * rowH, W - lab, b.n * rowH); }
      g.fillStyle = T.muted; g.fillText(b.key, 0, (b.start + b.n / 2) * rowH);
    });
    const s = tr.spikes;
    for (let i = 0; i < s.length; i += 2) {
      const n = s[i + 1];
      g.fillStyle = D.neurons.inh[n] ? T.inh : T.exc;
      g.fillRect(lab + s[i] * xs, D.neurons.rank[n] * rowH, Math.max(1, xs * 0.8), Math.max(1, rowH * 0.9));
    }
  }
  // Muscles.
  {
    const v = cv.muscles, g = v.layer.getContext("2d"), W = v.w, H = v.h, lab = 64;
    g.clearRect(0, 0, W, H);
    const lanes = [["shoulder", 0, 1], ["elbow", 2, 3]], lh = H / 2;
    g.font = `${Math.max(9, Math.min(12, W * 0.017))}px ${css("--f-mono")}`; g.textBaseline = "middle";
    lanes.forEach(([name, fl, ex], k) => {
      const y0 = k * lh + 4, y1 = (k + 1) * lh - 4;
      g.strokeStyle = T.grid; g.lineWidth = 1; g.beginPath(); g.moveTo(lab, y1); g.lineTo(W, y1); g.stroke();
      g.fillStyle = T.muted; g.fillText(name, 0, (y0 + y1) / 2);
      for (const [m, dash] of [[fl, []], [ex, [4, 3]]]) {
        g.setLineDash(dash); g.strokeStyle = T.ink; g.lineWidth = 1.4; g.beginPath();
        for (let f = 0; f < F; f++) {
          const x = lab + (W - lab) * f / (F - 1), y = y1 - (y1 - y0) * tr.act[f][m];
          f ? g.lineTo(x, y) : g.moveTo(x, y);
        }
        g.stroke();
      }
      g.setLineDash([]);
    });
  }
}

function drawArena(f) {
  const v = cv.arena, g = v.ctx, W = v.w, H = v.h, E = D.env, tr = D.trials[trial];
  g.clearRect(0, 0, W, H);
  const xmin = -E.wall_x - 0.04, xmax = E.wall_x + 0.04, ymin = -0.07, ymax = E.shoulder_height + E.l1 + E.l2 + 0.04;
  const s = Math.min(W / (xmax - xmin), H / (ymax - ymin));
  const ox = (W - (xmax - xmin) * s) / 2, oy = (H - (ymax - ymin) * s) / 2;
  const X = x => ox + (x - xmin) * s, Y = y => H - oy - (y - ymin) * s;
  // reach envelope of the hand
  g.strokeStyle = T.grid; g.lineWidth = 1; g.setLineDash([3, 4]);
  g.beginPath(); g.arc(X(0), Y(E.shoulder_height), (E.l1 + E.l2) * s, 0, 2 * Math.PI); g.stroke(); g.setLineDash([]);
  // floor, walls
  g.fillStyle = T.grid; g.fillRect(X(-E.wall_x), Y(0), (2 * E.wall_x) * s, 0.06 * s);
  g.strokeStyle = T.muted; g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(X(-E.wall_x), Y(0.35)); g.lineTo(X(-E.wall_x), Y(0)); g.lineTo(X(E.wall_x), Y(0)); g.lineTo(X(E.wall_x), Y(0.35)); g.stroke();
  g.fillStyle = T.muted; g.font = `${Math.max(9, W * 0.018)}px ${css("--f-mono")}`; g.textAlign = "center"; g.textBaseline = "top";
  for (let x = -0.8; x <= 0.81; x += 0.2) {
    g.fillRect(X(x) - 0.5, Y(0), 1, 4);
    g.fillText(Math.round(x * 100), X(x), Y(0) + 6);
  }
  g.textAlign = "left";
  // target
  g.fillStyle = T.target; g.globalAlpha = 0.22;
  g.fillRect(X(tr.target_x - E.tolerance), Y(0.42), 2 * E.tolerance * s, 0.42 * s); g.globalAlpha = 1;
  g.fillRect(X(tr.target_x) - 1, Y(0.42), 2, 0.42 * s);
  // start position of the ball
  g.strokeStyle = T.muted; g.setLineDash([3, 3]); g.lineWidth = 1;
  g.beginPath(); g.arc(X(tr.ball_x), Y(E.ball_radius), E.ball_radius * s, 0, 7); g.stroke(); g.setLineDash([]);
  // arm kinematics
  const arm = k => {
    const [q1, q2] = tr.q[k], p1 = q1 - Math.PI / 2, p2 = p1 + q2;
    const ex = E.l1 * Math.cos(p1), ey = E.shoulder_height + E.l1 * Math.sin(p1);
    return [ex, ey, ex + E.l2 * Math.cos(p2), ey + E.l2 * Math.sin(p2)];
  };
  // hand trail, last 0.4 s
  g.strokeStyle = T.inh; g.lineWidth = 1.5;
  for (let k = Math.max(1, f - 40); k <= f; k++) {
    const a = arm(k - 1), b = arm(k); g.globalAlpha = 0.6 * (1 - (f - k) / 40);
    g.beginPath(); g.moveTo(X(a[2]), Y(a[3])); g.lineTo(X(b[2]), Y(b[3])); g.stroke();
  }
  g.globalAlpha = 1;
  // ball
  const [bx, by] = tr.ball[f];
  g.fillStyle = T.ball; g.beginPath(); g.arc(X(bx), Y(by), E.ball_radius * s, 0, 7); g.fill();
  // arm
  const [ex, ey, hx, hy] = arm(f);
  g.strokeStyle = T.arm; g.lineCap = "round"; g.lineWidth = 2 * E.link_radius * s;
  g.beginPath(); g.moveTo(X(0), Y(E.shoulder_height)); g.lineTo(X(ex), Y(ey)); g.lineTo(X(hx), Y(hy)); g.stroke();
  g.fillStyle = T.panel;
  for (const [x, y] of [[0, E.shoulder_height], [ex, ey]]) { g.beginPath(); g.arc(X(x), Y(y), E.link_radius * s * 0.45, 0, 7); g.fill(); }
  // shoulder mount
  g.strokeStyle = T.muted; g.lineWidth = 2; g.lineCap = "butt";
  g.beginPath(); g.moveTo(X(-0.08), Y(E.shoulder_height + 0.05)); g.lineTo(X(0.08), Y(E.shoulder_height + 0.05)); g.stroke();
  g.lineCap = "butt";
}

function drawBrain(f) {
  const v = cv.brain, g = v.ctx, W = v.w, H = v.h, tr = D.trials[trial];
  g.clearRect(0, 0, W, H); g.drawImage(v.layer, 0, 0, W, H);
  const r = Math.max(1.6, W * 0.0036), s = tr.spikes, glow = new Float32Array(D.neurons.x.length);
  for (let k = 0; k < 4 && f - k >= 0; k++) {
    const a = Math.exp(-k * STEPS / 15);
    for (let i = tr.frameStart[f - k]; i < tr.frameStart[f - k + 1]; i++) { const n = s[2 * i + 1]; if (a > glow[n]) glow[n] = a; }
  }
  for (let n = 0; n < glow.length; n++) {
    if (!glow[n]) continue;
    g.globalAlpha = glow[n]; g.fillStyle = D.neurons.inh[n] ? T.inh : T.exc;
    g.beginPath(); g.arc(D.neurons.x[n] * W, D.neurons.y[n] * H, r * (1 + 0.6 * glow[n]), 0, 7); g.fill();
  }
  g.globalAlpha = 1;
}

function drawPlayhead(id, lab, f) {
  const v = cv[id], g = v.ctx;
  g.clearRect(0, 0, v.w, v.h); g.drawImage(v.layer, 0, 0, v.w, v.h);
  const x = lab + (v.w - lab) * f / (F - 1);
  g.strokeStyle = T.ink; g.lineWidth = 1; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, v.h); g.stroke();
}

const scrub = document.getElementById("scrub"), clock = document.getElementById("clock"), readout = document.getElementById("readout");
scrub.max = F - 1;
function render() {
  drawArena(frame); drawBrain(frame);
  drawPlayhead("raster", 34, frame); drawPlayhead("muscles", 64, frame);
  const tr = D.trials[trial], [bx] = tr.ball[frame];
  scrub.value = frame;
  clock.textContent = (frame * D.frame_dt).toFixed(2) + " s";
  readout.innerHTML = `ball <b>${cm(bx)} cm</b> · target <b>${cm(tr.target_x)} cm</b> · distance <b>${(Math.abs(bx - tr.target_x) * 100).toFixed(1)} cm</b>` +
    ` · final error <b>${(tr.error * 100).toFixed(1)} cm</b> (tolerance ${(D.env.tolerance * 100).toFixed(0)} cm)`;
}
function syncUI() {
  document.getElementById("play").textContent = playing ? "Pause" : "Play";
  document.getElementById("play").setAttribute("aria-pressed", playing);
  for (const b of document.querySelectorAll("[data-speed]")) b.setAttribute("aria-pressed", +b.dataset.speed === speed);
  D.trials.forEach((_, i) => document.getElementById("trial-" + i).setAttribute("aria-pressed", i === trial));
}
function tick(ts) {
  if (last === null) last = ts;
  const dt = (ts - last) / 1000; last = ts;
  const sig = css("--bg") + css("--exc");
  if (sig !== T.sig) { readTokens(); T.sig = sig; drawStatic(); }
  if (playing) {
    acc += dt * speed;
    const adv = Math.floor(acc / D.frame_dt);
    if (adv > 0) {
      acc -= adv * D.frame_dt;
      frame += adv;
      if (frame >= F) { frame = F - 1; playing = false; syncUI(); }
      render();
    }
  }
  requestAnimationFrame(tick);
}
document.getElementById("play").addEventListener("click", () => {
  if (!playing && frame >= F - 1) frame = 0;
  playing = !playing; acc = 0; syncUI(); render();
});
scrub.addEventListener("input", () => { frame = +scrub.value; playing = false; syncUI(); render(); });
for (const b of document.querySelectorAll("[data-speed]")) b.addEventListener("click", () => { speed = +b.dataset.speed; syncUI(); });
document.addEventListener("keydown", e => {
  if (e.code === "Space" && e.target === document.body) { e.preventDefault(); document.getElementById("play").click(); }
});
T.sig = css("--bg") + css("--exc");
new ResizeObserver(fit).observe(document.querySelector(".console"));
// Open on the most informative still: the middle of the first trial, then let reduced-motion viewers press Play.
frame = Math.floor(F / 2);
if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) { frame = 0; playing = true; }
syncUI(); fit();
if (document.fonts) document.fonts.ready.then(() => { drawStatic(); render(); });
requestAnimationFrame(tick);
})();
</script>
"""
