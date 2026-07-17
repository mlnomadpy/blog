"""Reshape the yat-attention bundles into viz JSON for the explainer.

Reads the 3-seed quality bundle (kgl_blog-yatattn-v1) and the telemetry
bundle (kgl_blog-yatattn-telem: checkpointed attention maps on one fixed
prompt). Writes public/attention-is-a-compatibility-kernel/{runs,maps}.json.
Maps are downsampled to 8-bit and a subset of layers/heads to keep the page
light.
"""

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(HERE, "results", "kgl_blog-yatattn-v1", "exp", "results",
                  "yat_attention.json")
TELEM_DIR = os.path.join(HERE, "results", "kgl_blog-yatattn-telem")
TELEM_B_DIR = os.path.join(HERE, "results", "kgl_blog-yatattn-b-telem")
OUT = os.path.join(HERE, "..", "public", "attention-is-a-compatibility-kernel")
os.makedirs(OUT, exist_ok=True)

d = json.load(open(V1))
BV1 = os.path.join(HERE, "results", "kgl_blog-yatattn-b-v1")
db = None
for p in glob.glob(os.path.join(BV1, "**", "yat_attention.json"), recursive=True):
    db = json.load(open(p))
# the contest: softmax (v1 bundle) vs the canonical kernel yat_b (b-v1 bundle);
# the bias-free runs stay available as the ablation
soft_runs = [r for r in d["runs"] if r["variant"] == "softmax"]
kern_runs = db["runs"] if db else [r for r in d["runs"] if r["variant"] == "yat"]
allruns = soft_runs + kern_runs
runs = dict(
    summary=dict(softmax=d["summary"]["softmax"],
                 yat=(db["summary"]["yat_b"] if db else d["summary"]["yat"]),
                 yat_nob=d["summary"]["yat"]),
    cfg={k: d["cfg"][k] for k in ("D", "LAYERS", "HEADS", "T", "STEPS")},
    curves=[dict(variant=("yat" if r["variant"].startswith("yat") else r["variant"]),
                 seed=r["seed"], best_val=r["best_val"],
                 curve=r["curve"]) for r in allruns],
    score_max=[dict(variant=r["variant"], seed=r["seed"],
                    mean=float(np.mean(r["score_max"])),
                    max=float(np.max(r["score_max"]))) for r in allruns],
    mass=[{k: r[k] for k in ("seed", "mass_auroc", "mass_mean_correct",
                             "mass_mean_wrong")}
          for r in kern_runs if r.get("mass_auroc") is not None],
)
with open(os.path.join(OUT, "runs.json"), "w") as f:
    json.dump(runs, f)
print(f"  runs.json: {os.path.getsize(os.path.join(OUT, 'runs.json'))//1024} KB")

# telemetry maps: (layers, heads, T, T) f16 per checkpoint per variant
maps = {}
tj = None
for p in glob.glob(os.path.join(TELEM_DIR, "**", "yat_attention.json"), recursive=True):
    tj = json.load(open(p))
prompt = next((r.get("telemetry_prompt") for r in (tj or {}).get("runs", [])
               if r.get("telemetry_prompt")), None)
LAYERS_KEEP = (0, 3, 5)
HEADS_KEEP = (0,)
paths = (glob.glob(os.path.join(TELEM_B_DIR, "**", "attn_*_s0.npz"), recursive=True)
         + glob.glob(os.path.join(TELEM_DIR, "**", "attn_*_s0.npz"), recursive=True))
seen = set()
for npz_path in paths:
    variant = os.path.basename(npz_path).split("_")[1]
    key = "yat" if variant.startswith("yat") else variant
    if key in seen:
        continue
    seen.add(key)
    variant = key
    z = np.load(npz_path)
    ck = {}
    for key in z.files:
        if not key.startswith("step"):
            continue
        A = z[key].astype(np.float32)               # (L, h, T, T)
        sub = A[np.ix_(LAYERS_KEEP, HEADS_KEEP)]    # (3, 1, T, T)
        # downsample T -> T/2 by 2x2 mean to keep the page light
        L2, H2, T1, _ = sub.shape
        sub = sub.reshape(L2, H2, T1 // 2, 2, T1 // 2, 2).mean(axis=(3, 5))
        q = np.clip(np.sqrt(sub) * 255, 0, 255).astype(np.uint8)  # sqrt for visibility
        ck[key] = q.tolist()
    maps[variant] = ck
with open(os.path.join(OUT, "maps.json"), "w") as f:
    json.dump(dict(prompt=prompt, layers=list(LAYERS_KEEP), heads=list(HEADS_KEEP),
                   maps=maps), f)
print(f"  maps.json: {os.path.getsize(os.path.join(OUT, 'maps.json'))//1024} KB")
