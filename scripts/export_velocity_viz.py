"""export_velocity_viz.py — reshape the velocity-ledger results bundle into the
web viz assets in public/velocity-ledger/. RESHAPE ONLY. No training, no model.

Source of truth (the ONLY number source):
  scripts/results/kgl_blog-velocity-ledger-v2/velocity_ledger.npz
  scripts/results/kgl_blog-velocity-ledger-v2/velocity_ledger_summary.json

What it writes (all numbers straight from the bundle, nothing invented):
  public/velocity-ledger/summary.json
     the 4-way best-val tie (mean/std/min/max, 3 seeds), the depth-dynamics
     table (final-checkpoint path length + mean turn angle per variant), the
     blow-up ratios, param counts, config, design notes, and the loss curves
     (train + val, seed 0) for the honest-tie panel.
  public/velocity-ledger/depth.json
     the depth telemetry per variant at the FINAL checkpoint (seed 0): the
     per-sub-update path lengths (seglen), turning angles (turn), residual-stream
     norm profile (xnorm), and, where they exist, the velocity-norm profile
     (vnorm, ledger variants). PLUS a faithful 2D polyline reconstruction of the
     residual-stream depth trajectory: a planar path whose segment lengths are
     the REAL measured seglen and whose turning angles are the REAL measured
     turn array. It is a 2D shadow (the true stream is 192-D), but every segment
     length and every bend is the real number, so the "straighter and shorter"
     verdict the eye reads off it is exactly the model's verdict. Also the path
     lengths at each of the four checkpoints, so the trajectory can straighten
     over training.

Run locally (no GPU needed): python3 scripts/export_velocity_viz.py
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'scripts' / 'results' / 'kgl_blog-velocity-ledger-v2'
OUT = ROOT / 'public' / 'velocity-ledger'
OUT.mkdir(parents=True, exist_ok=True)

npz = np.load(SRC / 'velocity_ledger.npz')
summary = json.load(open(SRC / 'velocity_ledger_summary.json'))

VARIANTS = ['plain', 'ledger', 'ngpt_lite', 'ngpt_ledger']
SEEDS = summary['config']['seeds']            # [0, 1, 2]
CKPTS = list(summary['config']['ckpt_steps'])  # [0, 1200, 6000, 12000]
NCK = len(CKPTS)
LABEL = {'plain': 'plain', 'ledger': 'ledger',
         'ngpt_lite': 'ngpt-lite', 'ngpt_ledger': 'ngpt-ledger'}


def seed_mean_std(fn):
    """fn(seed) -> scalar; return (mean, std) over the seeds."""
    vals = np.asarray([fn(s) for s in SEEDS], dtype=np.float64)
    return float(vals.mean()), float(vals.std())


def path_length(variant, seed, ck):
    return float(npz[f'{variant}_s{seed}/ck{ck}_seglen'].sum())


def mean_turn(variant, seed, ck):
    return float(npz[f'{variant}_s{seed}/ck{ck}_turn'].mean())


def polyline_2d(seglen, turn):
    """A faithful 2D reconstruction: segment i has the real length seglen[i];
    the bend between segment i and i+1 is the real turn[i]. Start heading +x.
    Returns nodes [[x,y], ...] (len = len(seglen)+1). Every length and every
    angle is the measured number; only the choice of turning left-vs-right and
    the initial heading are cosmetic (the shadow lives in a plane)."""
    seglen = np.asarray(seglen, dtype=np.float64)
    turn = np.asarray(turn, dtype=np.float64)
    heading = 0.0
    x, y = 0.0, 0.0
    nodes = [[0.0, 0.0]]
    sign = 1.0                                # alternate the bend direction so the
    for i, L in enumerate(seglen):            # path stays legible, not a spiral
        x += L * math.cos(heading)
        y += L * math.sin(heading)
        nodes.append([x, y])
        if i < len(turn):
            heading += sign * math.radians(turn[i])
            sign = -sign
    return [[round(px, 4), round(py, 4)] for px, py in nodes]


# ── summary.json : the tie + the dynamics + the curves ───────────────────────
comp = summary['comparison']
variants_out = []
for v in VARIANTS:
    c = comp[v]
    pl_mean, pl_std = seed_mean_std(lambda s: path_length(v, s, NCK - 1))
    tn_mean, tn_std = seed_mean_std(lambda s: mean_turn(v, s, NCK - 1))
    variants_out.append({
        'key': v,
        'label': LABEL[v],
        'best_val': c['best_val'],                     # mean/std/min/max/n
        'best_bpc': c['best_bpc'],
        'final_val': c['final_val'],
        'blowup_ratio': c['blowup_ratio'],
        'params': c['params'],
        'path_len': {'mean': round(pl_mean, 2), 'std': round(pl_std, 2)},
        'turn_deg': {'mean': round(tn_mean, 2), 'std': round(tn_std, 2)},
        # path length at each checkpoint (3-seed mean), for the straightening story
        'path_len_by_ckpt': [round(seed_mean_std(lambda s, ck=ck: path_length(v, s, ck))[0], 2)
                             for ck in range(NCK)],
    })

# loss curves (seed 0) for the tie panel
curves = {}
for v in VARIANTS:
    curves[v] = {
        'train_steps': [int(x) for x in npz[f'{v}_s0/train_steps']],
        'train_loss': [round(float(x), 4) for x in npz[f'{v}_s0/train_loss']],
        'val_steps': [int(x) for x in npz[f'{v}_s0/val_steps']],
        'val_loss': [round(float(x), 4) for x in npz[f'{v}_s0/val_loss']],
    }

summary_out = {
    'corpus': summary['corpus'],
    'corpus_chars': summary['corpus_chars'],
    'random_loss': summary['random_loss'],
    'headline_metric': summary['headline_metric'],
    'headline_order_best_val': summary['headline_order_best_val'],
    'config': {k: summary['config'][k] for k in
               ('D', 'LAYERS', 'HEADS', 'T', 'FF', 'BATCH', 'STEPS', 'seeds',
                'mu', 'h', 'peak_lr', 'dropout', 'vocab', 'ckpt_steps')},
    'design_notes': summary['design_notes'],
    'variants': variants_out,
    'curves': curves,
    'ckpt_steps': CKPTS,
}
(OUT / 'summary.json').write_text(json.dumps(summary_out, separators=(',', ':')))

# ── depth.json : per-variant depth telemetry at the final checkpoint (seed 0) ─
FINAL = NCK - 1
depth_out = {'ckpt_step': CKPTS[FINAL], 'variants': {}}
for v in VARIANTS:
    seglen = npz[f'{v}_s0/ck{FINAL}_seglen']
    turn = npz[f'{v}_s0/ck{FINAL}_turn']
    xnorm = npz[f'{v}_s0/ck{FINAL}_xnorm']
    entry = {
        'label': LABEL[v],
        'seglen': [round(float(x), 4) for x in seglen],
        'turn': [round(float(x), 3) for x in turn],
        'xnorm': [round(float(x), 4) for x in xnorm],
        'path_len': round(float(seglen.sum()), 3),
        'mean_turn': round(float(turn.mean()), 3),
        'nodes': polyline_2d(seglen, turn),
        # path length at every checkpoint (seed 0), so the path can straighten live
        'path_len_by_ckpt': [round(float(npz[f'{v}_s0/ck{ck}_seglen'].sum()), 3)
                             for ck in range(NCK)],
        # the 2D polyline at every checkpoint, for the training time-lapse
        'nodes_by_ckpt': [polyline_2d(npz[f'{v}_s0/ck{ck}_seglen'],
                                      npz[f'{v}_s0/ck{ck}_turn'])
                          for ck in range(NCK)],
    }
    if f'{v}_s0/ck{FINAL}_vnorm' in npz:
        entry['vnorm'] = [round(float(x), 4) for x in npz[f'{v}_s0/ck{FINAL}_vnorm']]
    depth_out['variants'][v] = entry
(OUT / 'depth.json').write_text(json.dumps(depth_out, separators=(',', ':')))

# ── report ────────────────────────────────────────────────────────────────
print('wrote', OUT / 'summary.json', (OUT / 'summary.json').stat().st_size, 'bytes')
print('wrote', OUT / 'depth.json', (OUT / 'depth.json').stat().st_size, 'bytes')
print('final-checkpoint path length (3-seed mean) + best-val (3-seed mean):')
for vo in variants_out:
    print(f"  {vo['label']:12s} pathlen {vo['path_len']['mean']:6.1f}"
          f"  turn {vo['turn_deg']['mean']:5.1f} deg"
          f"  best-val {vo['best_val']['mean']:.4f} (std {vo['best_val']['std']:.4f})")
