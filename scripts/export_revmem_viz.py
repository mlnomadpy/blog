"""Reshape the reversible-memory run bundle into viz JSON for the explainer.

Reads scripts/results/kgl_blog-revmem-v1/exp/results/reversible_memory.json
and writes public/backprop-without-the-memory/wall.json with the measured
memory/timing points, the gradient-fidelity grid, and the training summary.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "kgl_blog-revmem-v1", "exp", "results",
                   "reversible_memory.json")
OUT = os.path.join(HERE, "..", "public", "backprop-without-the-memory")
os.makedirs(OUT, exist_ok=True)

d = json.load(open(SRC))

wall = dict(
    memory=[{k: r[k] for k in ("L", "mode", "peak_mb", "temp_mb", "step_s")}
            for r in d["memory"]],
    fidelity=[{k: (None if r[k] != r[k] else r[k])  # nan -> null for JSON
               for k in ("mu", "L", "cosine", "rel_err", "budget")}
              for r in d["fidelity"]],
    training=dict(
        standard=dict(test_acc=d["training"]["standard"]["test_acc"],
                      wall_s=d["training"]["standard"]["wall_s"],
                      losses=d["training"]["standard"]["losses"]),
        reversible=dict(test_acc=d["training"]["reversible"]["test_acc"],
                        wall_s=d["training"]["reversible"]["wall_s"],
                        losses=d["training"]["reversible"]["losses"]),
    ),
)

with open(os.path.join(OUT, "wall.json"), "w") as f:
    json.dump(wall, f)
print(f"wall.json: {os.path.getsize(os.path.join(OUT, 'wall.json'))//1024} KB")
