"""Export the solve-wall bundle for the explainer viz.

Reads scripts/results/kgl_<slug>/kernel_solve_wall.{json,npz} and writes
public/you-dont-have-to-solve-a-kernel-machine/:
  agree.json    E1 ladder, solve rmse, rff baselines, per-epoch agreement
  wall.json     E2 measured solve/SGD timings and Gram memory
  scale.json    E3 covtype accuracy vs data used, solve and SGD
  compose.json  E4 the three fits' accuracies
  panel.json    E4 12-garment panel verdicts
  panel.png     sprite: 12 garments (top row) + frozen conv1 maps + trained conv1 maps

Usage: python3 scripts/export_solve_wall_viz.py [kgl_blog-solve-wall-v1] [kgl_blog-solve-wall-v1b]
The optional second bundle contributes e3b (the capacity ladder past the wall),
merged into scale.json as "sgd_big".
"""

import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLUG = sys.argv[1] if len(sys.argv) > 1 else "kgl_blog-solve-wall-v1"
SRC = os.path.join(ROOT, "scripts", "results", SLUG)
OUT = os.path.join(ROOT, "public", "you-dont-have-to-solve-a-kernel-machine")
os.makedirs(OUT, exist_ok=True)

R = json.load(open(os.path.join(SRC, "kernel_solve_wall.json")))
Z = np.load(os.path.join(SRC, "kernel_solve_wall.npz"))


def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f)
    print(f"wrote {name}")


# E1 ------------------------------------------------------------------
e1 = R["e1"]
dump("agree.json", {
    "n_train": e1["n_train"], "lam": e1["lam"], "eps": e1["eps"],
    "solve_rmse": e1["solve_rmse"], "solve_seconds": e1["solve_seconds"],
    "ladder": e1["ladder"], "K_head": e1["K_head"],
    "rff_same_features": e1["rff_rmse_same_features"],
    "rff_same_params": e1["rff_rmse_same_params"],
    "agree_r": e1["agree_series"]["r"] if e1.get("agree_series") else [],
    "agree_rmse": e1["agree_series"]["rmse"] if e1.get("agree_series") else [],
})

# E2 ------------------------------------------------------------------
dump("wall.json", {"rows": R["e2"]["rows"], "gpu_gb": R["e2"]["gpu_gb"]})

# E3 ------------------------------------------------------------------
scale = {"solve": R["e3"]["solve"], "sgd": R["e3"]["sgd"],
         "n_pool": R["e3"]["n_pool"], "eps": R["e3"]["eps"],
         "K": R["e3"]["K"], "wall_note": R["e3"]["wall_note"]}
if len(sys.argv) > 2:
    Rb = json.load(open(os.path.join(ROOT, "scripts", "results", sys.argv[2],
                                     "kernel_solve_wall.json")))
    scale["sgd_big"] = Rb["e3b"]["rows"]
dump("scale.json", scale)

# E4 ------------------------------------------------------------------
e4 = R["e4"]
dump("compose.json", {
    "n_small": e4["n_small"], "n_full": e4["n_full"],
    "solve_raw": e4["acc_solve_raw"], "solve_frozen": e4["acc_solve_frozen"],
    "e2e_small_mean": e4["acc_e2e_small_mean"], "e2e_small_std": e4["acc_e2e_small_std"],
    "e2e_small_seeds": e4["acc_e2e_small_seeds"],
    "e2e_full_mean": e4["acc_e2e_full_mean"], "e2e_full_std": e4["acc_e2e_full_std"],
    "e2e_full_seeds": e4["acc_e2e_full_seeds"],
})

# panel sprite: rows = [garments 28px | frozen maps 14px x8 | trained maps 14px x8]
if "e4_panel_images" in Z:
    imgs = Z["e4_panel_images"]          # 12 x 28 x 28
    fro = Z["e4_panel_fmaps_frozen"]     # 12 x 14 x 14 x 8
    trn = Z["e4_panel_fmaps_trained"]    # 12 x 14 x 14 x 8
    n = imgs.shape[0]

    def norm(a):
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo + 1e-9)

    CELL = 28
    W, H = n * CELL, CELL + 2 * CELL  # garment row + frozen strip + trained strip
    sheet = np.zeros((H, W), np.float32)
    for i in range(n):
        sheet[0:CELL, i * CELL:(i + 1) * CELL] = imgs[i]
        # 8 maps tiled 4x2 into a 28x28 block (each map 14x14 -> take first 4, 2x2)
        for s, block in ((CELL, fro[i]), (2 * CELL, trn[i])):
            tile = np.zeros((28, 28), np.float32)
            for k in range(4):
                r, c = divmod(k, 2)
                tile[r * 14:(r + 1) * 14, c * 14:(c + 1) * 14] = norm(block[..., k])
            sheet[s:s + CELL, i * CELL:(i + 1) * CELL] = tile
    Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8), "L").resize(
        (W * 4, H * 4), Image.NEAREST).save(os.path.join(OUT, "panel.png"))
    print("wrote panel.png")

    dump("panel.json", {
        "n": int(n), "cell": CELL, "scale": 4,
        "true": Z["e4_panel_true"].tolist(),
        "pred_raw": Z["e4_panel_pred_raw"].tolist(),
        "pred_frozen": Z["e4_panel_pred_frozen"].tolist(),
        "pred_e2e": Z["e4_panel_pred_e2e"].tolist(),
        "classes": ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal",
                     "shirt", "sneaker", "bag", "boot"],
    })
else:
    print("no panel arrays in npz (smoke bundle), skipped panel export")

print("done ->", OUT)
