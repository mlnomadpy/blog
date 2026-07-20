"""Export the geometry-of-attention post's assets from the trained runs.

Reads the geometry bundles (kgl_blog-attngeom-qk: softmax + yat_b at their
swept LRs; kgl_blog-attngeom-goat: goat_v) produced by yat_attention.py with
TELEMETRY=1 GEOMETRY=1, and writes public/the-geometry-of-attention/:

  census.json     per model x layer x head x checkpoint: the top-1 owner of
                  every query row (from the model's own attention maps), the
                  window's characters, and occupancy summaries
  scaletest.json  the real trained q/k vectors for two heads per model (one
                  sharp, one diffuse) + per-head kernel scalars, so the query-
                  scaling panel can recompute rows live in the reader's browser
  numbers.json    every statistic quoted in the post's prose, with definitions

Local, seconds-scale: a replay of exported weights, no training anywhere.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "..", "public", "the-geometry-of-attention")
os.makedirs(OUT, exist_ok=True)

BUNDLES = {
    "softmax": ("kgl_blog-attngeom-qk", "attn_softmax_s0.npz", "geom_softmax_s0.npz", False),
    "yat_b": ("kgl_blog-attngeom-qk", "attn_yat_b_s0.npz", "geom_yat_b_s0.npz", False),
    "goat_v": ("kgl_blog-attngeom-goat", "attn_goat_v_s0.npz", "geom_goat_v_s0.npz", True),
}
STEPS = [0, 1200, 6000, 11999]


def run_json(bundle):
    with open(os.path.join(RES, bundle, "yat_attention.json")) as f:
        return json.load(f)


def owners(attn, strict):
    """Top-1 key per query row from a (L,H,T,T) map; rows with <2 available
    keys are -1 (nothing to win)."""
    L, H, T, _ = attn.shape
    win = attn.argmax(axis=-1).astype(np.int16)           # (L,H,T)
    first = 2 if strict else 1                            # rows with >=2 keys
    win[:, :, :first] = -1
    return win


def occupancy(win, T):
    """Distinct winners / T, per head; -1 rows ignored."""
    L, H, _ = win.shape
    occ = np.zeros((L, H))
    for l in range(L):
        for h in range(H):
            w = win[l, h]
            occ[l, h] = len(set(w[w >= 0].tolist())) / T
    return occ


def norm_entropy_rows(attn, strict):
    """Mean normalized row entropy over rows with >=2 available keys."""
    L, H, T, _ = attn.shape
    ent = []
    for i in range(2 if strict else 1, T):
        n_avail = i if strict else i + 1
        rows = attn[:, :, i, :].astype(np.float64)
        rows = rows / np.maximum(rows.sum(-1, keepdims=True), 1e-12)
        h = -(rows * np.log(np.maximum(rows, 1e-12))).sum(-1)
        ent.append(h / np.log(n_avail))
    return float(np.mean(ent))


def main():
    census = {"steps": STEPS, "models": {}}
    numbers = {"definitions": {
        "winner": "argmax of the model's own attention row (top-1 key)",
        "nearest": "argmin_j ||q_i - k_j|| over the causal window, from the trained q/k",
        "aligned": "argmax_j q_i . k_j over the causal window",
        "scale_changed(t)": "fraction of rows whose winner changes when every query is scaled q -> t q (scores recomputed with the head's learned scalars)",
        "occupancy": "distinct top-1 winners / sequence length, per head",
    }, "models": {}}
    scaletest = {"models": {}}

    for model, (bundle, attn_f, geom_f, strict) in BUNDLES.items():
        za = np.load(os.path.join(RES, bundle, attn_f))
        zg = np.load(os.path.join(RES, bundle, geom_f))
        rj = run_json(bundle)
        run = next(r for r in rj["runs"] if r["variant"] == model)
        chars = run["telemetry_prompt"]
        q = zg["q"].astype(np.float64)                    # (L,H,T,dh)
        k = zg["k"].astype(np.float64)
        L, H, T, dh = q.shape
        b_l = np.array(run.get("b_learned", [[0.0] * H] * L))
        e_l = np.array(run.get("eps_learned", [[1.0] * H] * L))

        # ── census: owners at every checkpoint, from the model's own maps ──
        occ_by_step, own_by_step = [], []
        for s in STEPS:
            A = za[f"step{s}"].astype(np.float32)
            win = owners(A, strict)
            own_by_step.append(win.tolist())
            occ_by_step.append(occupancy(win, T).tolist())
        census["models"][model] = {
            "chars": chars,
            "owners": own_by_step,                        # [step][L][H][T]
            "occupancy": occ_by_step,                     # [step][L][H]
        }

        # ── geometry stats on the final-step q/k ──
        A_final = zg["attn"].astype(np.float64)
        stats = {"L": L, "H": H, "T": T, "dh": dh}
        match_near, match_align, ent_final = [], [], []
        d2_all = ((q[:, :, :, None, :] - k[:, :, None, :, :]) ** 2).sum(-1)   # (L,H,T,T)
        dots_all = np.einsum("lhtd,lhsd->lhts", q, k)
        first = 2 if strict else 1
        for l in range(L):
            for h in range(H):
                for i in range(first, T):
                    ub = i if strict else i + 1           # keys j < ub
                    row = A_final[l, h, i, :ub]
                    win = int(row.argmax())
                    near = int(d2_all[l, h, i, :ub].argmin())
                    alig = int(dots_all[l, h, i, :ub].argmax())
                    match_near.append(win == near)
                    match_align.append(win == alig)
        stats["winner_is_nearest"] = float(np.mean(match_near))
        stats["winner_is_aligned"] = float(np.mean(match_align))
        stats["mean_norm_entropy_final"] = norm_entropy_rows(
            za[f"step{STEPS[-1]}"].astype(np.float64), strict)
        stats["occupancy_mean_by_step"] = [float(np.mean(o)) for o in occ_by_step]
        stats["occupancy_final_min"] = float(np.min(occ_by_step[-1]))
        stats["occupancy_final_max"] = float(np.max(occ_by_step[-1]))

        # ── the scale test: q -> t q, winner recomputed from the raw law ──
        ts = [0.5, 0.8, 1.25, 2.0, 4.0]
        changed = {t: [] for t in ts}
        for l in range(L):
            for h in range(H):
                dots = dots_all[l, h]
                if model == "softmax":
                    def score(t):
                        return t * dots                    # argmax unaffected by /sqrt(dh)
                else:
                    b, eps = b_l[l][h], e_l[l][h]
                    qq = (q[l, h] ** 2).sum(-1)
                    kk2 = (k[l, h] ** 2).sum(-1)

                    def score(t):
                        d2 = (t * t) * qq[:, None] + kk2[None, :] - 2 * t * dots
                        return (t * dots + b) ** 2 / (np.maximum(d2, 0.0) + eps)
                base = score(1.0)
                for i in range(first, T):
                    ub = i if strict else i + 1
                    w0 = int(base[i, :ub].argmax())
                    for t in ts:
                        wt = int(score(t)[i, :ub].argmax())
                        changed[t].append(wt != w0)
        stats["scale_changed"] = {str(t): float(np.mean(changed[t])) for t in ts}

        # ── hull census in the real 48-d key space: how many keys sit strictly
        # inside the convex hull of their head's other keys (LP feasibility)? ──
        try:
            from scipy.optimize import linprog
            interior = 0
            for l in range(L):
                for h in range(H):
                    K = k[l, h]                            # (T, dh)
                    for j in range(T):
                        others = np.delete(K, j, axis=0)   # (T-1, dh)
                        A_eq = np.vstack([others.T, np.ones(T - 1)])
                        b_eq = np.concatenate([K[j], [1.0]])
                        r = linprog(np.zeros(T - 1), A_eq=A_eq, b_eq=b_eq,
                                    bounds=(0, None), method="highs")
                        if r.status == 0:
                            interior += 1
            stats["keys_inside_hull"] = interior
            stats["keys_total"] = L * H * T
        except Exception as e:                             # pragma: no cover
            stats["keys_inside_hull"] = None
            stats["hull_error"] = str(e)
        numbers["models"][model] = stats

        # ── browser payload: two heads, chosen sharp + diffuse ──
        ent_head = np.zeros((L, H))
        A_last = za[f"step{STEPS[-1]}"].astype(np.float64)
        for l in range(L):
            for h in range(H):
                rows_ = A_last[l, h, first:, :]
                rows_ = rows_ / np.maximum(rows_.sum(-1, keepdims=True), 1e-12)
                hh = -(rows_ * np.log(np.maximum(rows_, 1e-12))).sum(-1)
                navail = np.arange(first, T) + (0 if strict else 1)
                ent_head[l, h] = float(np.mean(hh / np.log(np.maximum(navail, 2))))
        flat = [(ent_head[l, h], l, h) for l in range(L) for h in range(H)]
        flat.sort()
        picks = [flat[0][1:], flat[-1][1:]]                # sharpest + most diffuse
        heads = []
        for (l, h) in picks:
            heads.append({
                "layer": int(l), "head": int(h),
                "entropy": round(float(ent_head[l, h]), 3),
                "b": round(float(b_l[l][h]), 4), "eps": round(float(e_l[l][h]), 4),
                "q": np.round(q[l, h], 3).tolist(),
                "k": np.round(k[l, h], 3).tolist(),
            })
        scaletest["models"][model] = {
            "strict": strict, "chars": chars, "heads": heads,
        }

    for name, obj in [("census", census), ("numbers", numbers), ("scaletest", scaletest)]:
        p = os.path.join(OUT, f"{name}.json")
        with open(p, "w") as f:
            json.dump(obj, f, separators=(",", ":"))
        print(f"{name}.json  {os.path.getsize(p)/1e6:.2f} MB")

    print(json.dumps(numbers["models"], indent=1)[:4000])


if __name__ == "__main__":
    main()
