"""Extract compact viz assets from the completed DeepSurv trial battery.

Reads the ONE verified artifact
  scripts/results/kgl_blog-deepsurv-trial-v2/trial_results.json
(produced by scripts/deepsurv_trial.py on Kaggle; no training happens here) and
writes small JSON files into
  public/you-dont-have-to-solve-a-kernel-machine/
that the explainer's interactive panels load and re-slice. Every number is copied
straight from the artifact; nothing is invented or recomputed except trivial
re-slicing. Run locally:

  python3 scripts/deepsurv_trial_assets.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'scripts' / 'results' / 'kgl_blog-deepsurv-trial-v2' / 'trial_results.json'
OUT = ROOT / 'public' / 'you-dont-have-to-solve-a-kernel-machine'
OUT.mkdir(parents=True, exist_ok=True)

d = json.load(open(SRC))
DATASETS = d['meta']['datasets']            # metabric, support, gbsg, whas500, flchain
MODELS = d['meta']['models']                # coxph, coxnet, rsf, mlp, yat


import math


def r(x, n=4):
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):   # NaN/inf are not valid JSON for the browser's parser
        return None
    return round(x, n)


# ---- board.json: the 5x5 C-index board + cards + per-seed + headline boot CI ----
cards = {}
for k, c in d['cards'].items():
    cards[k] = dict(n=c['n'], d=c['n_covariates'],
                    cens=r(c['censoring_rate'], 3), median_t=r(c['median_time'], 1),
                    desc=c['desc'])

board = {}
for ds in DATASETS:
    board[ds] = {}
    for m in MODELS:
        s = d['summary'][ds][m]
        boot = s.get('boot_cindex_seed0', {})
        seeds = [r(x['cindex']) for x in d['per_seed'][ds][m]]
        board[ds][m] = dict(
            mean=r(s['cindex']['mean']), std=r(s['cindex']['std']),
            ibs_mean=r(s['ibs']['mean']), auc_mean=r(s['mean_auc']['mean']),
            boot_lo=r(boot.get('lo')), boot_hi=r(boot.get('hi')),
            boot_mean=r(boot.get('mean')),
            seeds=seeds,
            lr_mode=r(s.get('selected_lr_mode'), 4) if m in ('mlp', 'yat') else None,
        )
json.dump(dict(meta=dict(seeds=len(d['meta']['seeds']), epochs=d['meta']['epochs'],
                         K=d['meta']['K_hidden'], n_boot=d['meta']['n_boot'],
                         lr_grid=d['meta']['lr_grid']),
               cards=cards, models=MODELS, datasets=DATASETS, board=board),
          open(OUT / 'board.json', 'w'))

# ---- ablation.json: K x init sweep on metabric + gbsg ----
abl = {}
for ds, rows in d['ablation'].items():
    abl[ds] = [dict(K=x['K'], init=x['init'], c=r(x['cindex_mean']),
                    c_std=r(x['cindex_std']), ibs=r(x['ibs_mean']))
               for x in rows]
# the matched-capacity headline (K=24) yat C-index per dataset, for reference lines
head24 = {ds: r(d['summary'][ds]['yat']['cindex']['mean']) for ds in d['ablation']}
json.dump(dict(ablation=abl, headline_k24=head24,
               K_grid=sorted({x['K'] for x in d['ablation']['metabric']})),
          open(OUT / 'ablation.json', 'w'))

# ---- inherit.json: one real trial number behind each inherited kernel property ----
protos = d['plausibility']['metabric']['prototypes']
readouts = sorted([r(p['readout'], 3) for p in protos])
n_raise = sum(1 for p in protos if p['readout'] > 0)
n_protect = sum(1 for p in protos if p['readout'] < 0)
ood_m = d['ood']['metabric']
inherit = dict(
    attribution=dict(
        readouts=readouts, n_raise=n_raise, n_protect=n_protect,
        most_protective=r(min(p['readout'] for p in protos), 3),
        most_raising=r(max(p['readout'] for p in protos), 3)),
    ood=dict(
        kmax_median={ds: r(d['ood'][ds]['kmax_in_median'], 2) for ds in DATASETS},
        permutation_auroc=r(ood_m['permutation']['auroc'], 3),
        extreme_auroc=r(ood_m['extreme_quantile']['auroc'], 3),
        cross_dataset={ds: (None if d['cross_dataset_ood'][ds].get('skipped')
                            else dict(other=d['cross_dataset_ood'][ds]['other'],
                                      auroc=r(d['cross_dataset_ood'][ds]['auroc'], 3)))
                       for ds in DATASETS}),
    capacity=dict(
        # C-index vs K on metabric (kmeans init), showing capacity is a dial not a cliff
        curve=[dict(K=x['K'], c=r(x['cindex_mean']))
               for x in d['ablation']['metabric'] if x['init'] == 'kmeans']),
    plausibility=dict(
        median_nearest_dist=r(d['plausibility']['metabric']['median_nearest_dist'], 3),
        frac_in_range=r(d['plausibility']['metabric']['frac_all_covars_in_range'], 3),
        n_prototypes=len(protos)),
)
json.dump(inherit, open(OUT / 'inherit.json', 'w'))

# ---- plausibility.json: the 24 prototypes' validity checks (metabric) ----
pl = d['plausibility']['metabric']
plaus = dict(
    feature_names=pl['feature_names'],
    median_nearest_dist=r(pl['median_nearest_dist'], 3),
    frac_in_range=r(pl['frac_in_range'] if 'frac_in_range' in pl
                    else pl['frac_all_covars_in_range'], 3),
    prototypes=[dict(id=p['id'], readout=r(p['readout'], 3),
                     nn_dist=r(p['nearest_patient_dist'], 3),
                     out_of_range=p['n_covars_out_of_range'],
                     max_excursion=r(p['max_excursion_sd'], 3),
                     in_range=p['per_covar_in_range'],
                     covars=[r(x, 2) for x in p['covariates_clin']])
                for p in protos])
json.dump(plaus, open(OUT / 'plausibility.json', 'w'))

for f in ['board.json', 'ablation.json', 'inherit.json', 'plausibility.json']:
    print(f"wrote {f}  ({(OUT / f).stat().st_size} bytes)")
print(f"-> {OUT}")
