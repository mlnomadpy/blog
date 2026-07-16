"""Reshape the adaptive-depth run bundle into viz JSON for the explainer.

Reads scripts/results/kgl_blog-adaptive-v2/exp/results/adaptive_depth.json
(the run with the seed-0 weight export) and writes
public/depth-on-demand/{sweep,models}.json.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "kgl_blog-adaptive-v2", "exp", "results",
                   "adaptive_depth.json")
OUT = os.path.join(HERE, "..", "public", "depth-on-demand")
os.makedirs(OUT, exist_ok=True)

d = json.load(open(SRC))

sweep = {}
models = {}
for ds, per_seed in d.items():
    rows = []
    n_tol = len(per_seed[0]["sweep"])
    for ti in range(n_tol):
        rs = [s["sweep"][ti] for s in per_seed]
        rows.append(dict(
            tol=rs[0]["tol"],
            agree=float(np.mean([r["agree_ref"] for r in rs])),
            acc=float(np.mean([r["acc"] for r in rs])),
            work=float(np.mean([r["work_mean"] for r in rs])),
            accepted=float(np.mean([r["accepted_mean"] for r in rs])),
            work_min=int(min(r["work_min"] for r in rs)),
            work_max=int(max(r["work_max"] for r in rs)),
            corr=[float(r["corr_work_difficulty"]) for r in rs],
        ))
    # the saved per-point arrays (seed 0, one tol) for the scatter
    saved = next((r for r in per_seed[0]["sweep"] if r.get("works")), None)
    sweep[ds] = dict(
        acc_fixed=float(np.mean([s["acc_fixed"] for s in per_seed])),
        rows=rows,
        scatter=dict(tol=saved["tol"], works=saved["works"],
                     accepted=saved["accepted"], margins=saved["margins"]) if saved else None,
    )
    viz = per_seed[0].get("viz")
    if viz:
        models[ds] = viz

with open(os.path.join(OUT, "sweep.json"), "w") as f:
    json.dump(sweep, f)
with open(os.path.join(OUT, "models.json"), "w") as f:
    json.dump(models, f)
for n in ("sweep", "models"):
    p = os.path.join(OUT, f"{n}.json")
    print(f"  {n}.json: {os.path.getsize(p)//1024} KB")
