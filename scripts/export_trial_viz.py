#!/usr/bin/env python3
"""Reshape the multi-dataset trial bundle into per-viz JSON for the follow-up post.

Reads the ONE source of truth, scripts/results/kgl_blog-deepsurv-trial-v2/
trial_results.json (produced by scripts/deepsurv_trial.py on Kaggle), and writes
small, panel-shaped JSON files into public/survival-trial/. No training, no model
loading: this only re-slices numbers that already exist in the bundle.

Outputs (each backs one engine panel in survival-model-on-trial.mdx):
  forest.json       -- per (dataset, model) C-index mean/std + seed-0 bootstrap CI.
  ablation.json     -- K x init sweep of the Yat model on metabric + gbsg.
  calibration.json  -- predicted-vs-observed survival by risk tertile, yat + mlp.
  ood.json          -- kernel-max OOD AUROCs (three constructions) with CIs + caveats.
  plausibility.json -- Yat prototypes on METABRIC in clinical units + in-range flags.

Run: python3 scripts/export_trial_viz.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "results" / "kgl_blog-deepsurv-trial-v2" / "trial_results.json"
OUT = ROOT / "public" / "survival-trial"
OUT.mkdir(parents=True, exist_ok=True)

d = json.load(open(SRC))
DATASETS = d["meta"]["datasets"]
MODELS = d["meta"]["models"]  # coxph, coxnet, rsf, mlp, yat


def dump(name, obj):
    with open(OUT / name, "w") as f:
        json.dump(obj, f)
    print(f"  wrote {name} ({(OUT / name).stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# 1. FOREST: per (dataset, model) C-index mean/std + seed-0 bootstrap CI. The
#    "viability at a glance" panel: does the white-box model land in the pack?
# ---------------------------------------------------------------------------
def forest():
    label = {"coxph": "Cox PH", "coxnet": "penalized Cox",
             "rsf": "Random Survival Forest", "mlp": "ReLU DeepSurv", "yat": "Yat DeepSurv"}
    cards = {k: dict(n=d["cards"][k]["n"], d=d["cards"][k]["n_covariates"],
                     cens=round(d["cards"][k]["censoring_rate"], 3),
                     median_t=round(d["cards"][k]["median_time"], 1))
             for k in DATASETS}
    rows = {}
    for ds in DATASETS:
        rows[ds] = []
        for m in MODELS:
            s = d["summary"][ds][m]
            c = s["cindex"]
            b = s.get("boot_cindex_seed0", {})
            rows[ds].append(dict(
                model=m, label=label[m],
                mean=round(c["mean"], 4), std=round(c["std"], 4),
                lo=round(b.get("lo", float("nan")), 4) if b.get("lo") is not None else None,
                hi=round(b.get("hi", float("nan")), 4) if b.get("hi") is not None else None,
                auc=round(s["mean_auc"]["mean"], 4), ibs=round(s["ibs"]["mean"], 4),
                is_yat=(m == "yat"), is_net=(m in ("mlp", "yat")),
            ))
    dump("forest.json", dict(datasets=DATASETS, cards=cards, rows=rows,
                             note="Harrell C-index, mean +/- std over 6 seeds; bracket is "
                                  "the seed-0 bootstrap 95% CI (300 resamples). Per-model "
                                  "LR chosen by inner validation; test set untouched."))


# ---------------------------------------------------------------------------
# 2. ABLATION: Yat C-index vs prototype count K, kmeans vs random seeding.
# ---------------------------------------------------------------------------
def ablation():
    out = {}
    for ds in d["ablation"]:
        by_init = {"kmeans": [], "random": []}
        for r in d["ablation"][ds]:
            by_init[r["init"]].append(dict(
                K=r["K"], mean=round(r["cindex_mean"], 4), std=round(r["cindex_std"], 4),
                ibs=round(r["ibs_mean"], 4)))
        for k in by_init:
            by_init[k].sort(key=lambda r: r["K"])
        out[ds] = by_init
    dump("ablation.json", dict(datasets=list(out.keys()), Ks=[6, 12, 24, 48], series=out,
                               note="Yat DeepSurv C-index vs number of prototypes K, kmeans "
                                    "vs random seeding, mean +/- std over 3 seeds."))


# ---------------------------------------------------------------------------
# 3. CALIBRATION: predicted vs KM-observed survival by risk tertile, yat + mlp.
# ---------------------------------------------------------------------------
def calibration():
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for m in ("yat", "mlp"):
            rel = d["calibration"][ds][m]["reliability"]
            out[ds][m] = dict(
                horizons=[round(h, 1) for h in rel["horizons"]],
                mae=round(rel["mean_abs_error"], 4),
                groups=[dict(group=g["group"], n=g["n"],
                             pred=[round(v, 4) for v in g["pred"]],
                             obs=[round(v, 4) for v in g["obs"]])
                        for g in rel["groups"]])
    dump("calibration.json", dict(datasets=DATASETS, models=["yat", "mlp"], data=out,
                                  note="Predicted survival (model) vs Kaplan-Meier observed, "
                                       "patients split into risk tertiles. MAE is the mean "
                                       "absolute reliability gap across horizons and groups."))


# ---------------------------------------------------------------------------
# 4. OOD: kernel-max AUROC separating in- from out-of-distribution, per dataset,
#    for three honest constructions. Carries the caveats verbatim.
# ---------------------------------------------------------------------------
def ood():
    keys = ["permutation", "extreme_quantile", "real_subgroup"]
    short = {"permutation": "shuffled covariates",
             "extreme_quantile": "far-tail synthetic",
             "real_subgroup": "held-out subgroup"}
    out = {}
    for ds in DATASETS:
        o = d["ood"][ds]
        out[ds] = dict(kmax_in_median=round(o.get("kmax_in_median", float("nan")), 3), bars=[])
        for k in keys:
            if k in o:
                v = o[k]
                out[ds]["bars"].append(dict(
                    key=k, label=short[k],
                    auroc=round(v.get("auroc", float("nan")), 4),
                    lo=round(v.get("lo", float("nan")), 4) if v.get("lo") is not None else None,
                    hi=round(v.get("hi", float("nan")), 4) if v.get("hi") is not None else None,
                    caveat=v.get("caveat", "")))
    # the one true cross-dataset shift that ran (support <-> whas500)
    cds = []
    for ds in DATASETS:
        r = d["cross_dataset_ood"].get(ds, {})
        if not r.get("skipped"):
            cds.append(dict(train=ds, other=r["other"], auroc=round(r["auroc"], 4),
                            lo=round(r["lo"], 4), hi=round(r["hi"], 4)))
    dump("ood.json", dict(datasets=DATASETS, data=out, cross_dataset=cds,
                          note="AUROC of the kernel-max score separating real held-out "
                               "patients (high) from constructed out-of-distribution ones "
                               "(low). 0.5 = the model finds them no stranger than real "
                               "patients. Bracket is a 300-resample bootstrap CI."))


# ---------------------------------------------------------------------------
# 5. PLAUSIBILITY: Yat prototypes on METABRIC in clinical units + in-range flags.
# ---------------------------------------------------------------------------
def plausibility():
    p = d["plausibility"]["metabric"]
    protos = []
    for r in p["prototypes"]:
        protos.append(dict(
            id=r["id"], readout=round(r["readout"], 4),
            clin=[round(v, 3) for v in r["covariates_clin"]],
            in_range=r["per_covar_in_range"],
            n_out=r["n_covars_out_of_range"],
            max_excursion=round(r["max_excursion_sd"], 3),
            nn_dist=round(r["nearest_patient_dist"], 3),
            nn_clin=[round(v, 3) for v in r["nearest_patient_clin"]],
            plausible=(r["n_covars_out_of_range"] == 0)))
    protos.sort(key=lambda r: r["readout"])
    dump("plausibility.json", dict(
        feature_names=p["feature_names"], prototypes=protos,
        frac_plausible=round(p["frac_all_covars_in_range"], 3),
        n_plausible=sum(1 for r in protos if r["plausible"]),
        n_total=len(protos),
        median_nn_dist=round(p["median_nearest_dist"], 3),
        note="Each Yat prototype de-normalized to clinical units, with a per-covariate "
             "flag for whether it lands inside the training min/max and its distance to "
             "the nearest real training patient."))


if __name__ == "__main__":
    print(f"reading {SRC.relative_to(ROOT)}")
    forest(); ablation(); calibration(); ood(); plausibility()
    print(f"done -> {OUT.relative_to(ROOT)}")
