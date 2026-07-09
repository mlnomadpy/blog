"""The White-Box Survival Model on Trial: a rigorous multi-dataset battery.

Backs the follow-up post to "A Risk Model That Names Its Reasons". An external
reviewer called the original "a promising research prototype, not a validated
method". This script turns each caution into a real, honest measurement. If the
Yat model loses ground under harder scrutiny, THAT is the finding -- nothing here
is rigged to produce a win.

The reviewer's critique IS the experiment plan:

  1. MULTIPLE DATASETS (not just METABRIC): metabric, support, gbsg (pycox) plus
     whas500, flchain, veterans (sksurv). Real clinical/survival data, all load
     on Kaggle with no external download. n / covars / censoring reported per set.

  2. BROADER BASELINES (not just a matched ReLU MLP), per dataset, same split:
       (a) classical CoxPH               (sksurv CoxPHSurvivalAnalysis)
       (b) penalized Cox (elastic net)   (sksurv CoxnetSurvivalAnalysis)
       (c) Random Survival Forest        (sksurv RandomSurvivalForest)
       (d) standard ReLU DeepSurv MLP    (from yat_deepsurv.py, the black box)
       (e) YAT DeepSurv                  (the method under test)
     LR-FAIR: the two neural nets do NOT share a learning rate (a shared LR is an
     optimization artifact -- the probe found Yat prefers ~1e-2, the MLP ~3e-3).
     Each net sweeps lr in {3e-3, 1e-2, 3e-2}, picks its OWN best by inner-validation
     C-index (each candidate early-stopped at its best epoch), and is reported at
     that LR. The LR-free classical baselines anchor the headline comparison.

  3. METRICS WITH UNCERTAINTY: Harrell C, integrated time-dependent AUC,
     integrated Brier, Cox partial NLL. Mean +/- std over >=5 seeds (reviewer
     flagged 3 as too few), AND bootstrap CIs on the test C-index.

  4. CALIBRATION: predicted-vs-observed survival at fixed horizons (reliability),
     a one-sample (D-calibration-style) statistic, and a subgroup calibration
     split by a covariate median.

  5. PROTOTYPE-COUNT + INIT ABLATION (Yat): sweep K in {6,12,24,48} x {kmeans,
     random} on 1-2 datasets; C-index / IBS vs K.

  6. OOD RIGOR: kernel-max separating in- vs out-of-distribution, AUROC with CIs,
     across THREE honest OOD constructions (feature permutation, extreme-quantile
     patients, a real held-out covariate subgroup) PLUS a true cross-dataset
     covariate-shift test where feature dimensions align.

  7. PROTOTYPE PLAUSIBILITY (Yat on METABRIC): each learned prototype de-normalized
     to clinical units, distance to nearest real patient, per-covariate in/out of
     the training min/max -- plausible clinical region vs math-useful-but-odd point.

Output: ONE results bundle (trial_results.json + trial_results.npz) with, per
(dataset, model, seed): all metrics + CIs; per dataset: reliability curves, the
ablation sweep, OOD AUROCs, the prototype plausibility table; plus a compact
machine-readable `summary` the post's tables read from. Deterministic seeds.

Compute policy: TRAINS ON KAGGLE, never locally. Writes flat files into ./results
(cwd) so the kgl.py launcher surfaces them to /kaggle/working.

SMOKE=1 : 1 dataset (metabric), all models, 2 seeds, tiny epochs, small ablation
          and OOD, exercises the FULL telemetry schema in ~3-5 min.

Run on Kaggle via kgl.py:
  cd ~/.claude/skills/kaggle-cli-experiments && python3 kgl.py \
    --entry deepsurv_trial.py --slug blog-deepsurv-trial \
    --expdir /Users/tahabsn/conductor/workspaces/blog/hartford/scripts \
    --pip "flax optax scikit-survival lifelines pycox scikit-learn" --no-wait

This is a research illustration, NOT a clinical tool. No number here should touch
a real medical decision.
"""
import warnings; warnings.filterwarnings('ignore')
import json, os, time
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp
import optax
from flax import nnx

t_start = time.time()
SMOKE = os.environ.get('SMOKE', '0') == '1'
print(f"=== DEEPSURV TRIAL (SMOKE={SMOKE}) ===", flush=True)

# ---------------------------------------------------------------------------
# Output: flat files in ./results (cwd) on Kaggle; the launcher copies them up.
# ---------------------------------------------------------------------------
ON_KAGGLE = Path('/kaggle/working').exists()
if ON_KAGGLE:
    OUTDIR = Path('results')
else:
    OUTDIR = Path(__file__).resolve().parents[1] / 'public' / 'deepsurv-trial'
OUTDIR.mkdir(parents=True, exist_ok=True)


def jsonable(x):
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, dict): return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    return x


# ===========================================================================
# 1. DATASETS. Each returns X (float32 [n,d]), t (float32), e (float32 0/1),
#    feature names, and a short card. Everything loads on Kaggle with no
#    external download (pycox bundles its CSVs; sksurv ships its datasets).
# ===========================================================================
def _clean(X, t, e):
    X = np.asarray(X, dtype='float32')
    t = np.asarray(t, dtype='float32')
    e = np.asarray(e, dtype='float32')
    good = np.isfinite(X).all(1) & np.isfinite(t) & np.isfinite(e) & (t > 0)
    return X[good], t[good], e[good]


def load_metabric():
    from pycox.datasets import metabric
    df = metabric.read_df()
    feats = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
    names = ['MKI67', 'EGFR', 'PGR', 'ERBB2', 'Hormone therapy',
             'Radiotherapy', 'Chemotherapy', 'ER-positive', 'Age at diagnosis']
    X, t, e = _clean(df[feats].values, df['duration'].values, df['event'].values)
    return X, t, e, names, 'METABRIC breast cancer overall survival (pycox)'


def load_support():
    from pycox.datasets import support
    df = support.read_df()
    feats = [c for c in df.columns if c not in ('duration', 'event')]
    X, t, e = _clean(df[feats].values, df['duration'].values, df['event'].values)
    return X, t, e, list(feats), 'SUPPORT critically-ill hospitalized adults (pycox)'


def load_gbsg():
    from pycox.datasets import gbsg
    df = gbsg.read_df()
    feats = [c for c in df.columns if c not in ('duration', 'event')]
    X, t, e = _clean(df[feats].values, df['duration'].values, df['event'].values)
    return X, t, e, list(feats), 'GBSG/Rotterdam breast cancer recurrence-free survival (pycox)'


def _sksurv_xy(load_fn, name, desc):
    from sksurv.datasets import get_x_y
    import pandas as pd
    data = load_fn()
    # sksurv loaders return either (X_df, y) or a bunch; normalize to (X_df, y)
    if isinstance(data, tuple):
        Xdf, y = data
    else:
        Xdf, y = data.data, data.target  # unlikely path
    # y is a structured array with (event_bool, time). Field names vary.
    ev_field, tm_field = y.dtype.names
    e = y[ev_field].astype('float32')
    t = y[tm_field].astype('float32')
    # one-hot encode categoricals, keep numeric
    Xdf = pd.get_dummies(Xdf, drop_first=True)
    names = list(Xdf.columns)
    X = Xdf.values.astype('float32')
    X, t, e = _clean(X, t, e)
    return X, t, e, [str(n) for n in names], desc


def load_whas500():
    from sksurv.datasets import load_whas500
    return _sksurv_xy(load_whas500, 'whas500',
                      'WHAS500 Worcester heart-attack survival (sksurv)')


def load_flchain():
    from sksurv.datasets import load_flchain
    return _sksurv_xy(load_flchain, 'flchain',
                      'FLCHAIN free light chain / mortality (sksurv)')


def load_veterans():
    from sksurv.datasets import load_veterans_lung_cancer
    return _sksurv_xy(load_veterans_lung_cancer, 'veterans',
                      'Veterans lung cancer trial (sksurv)')


DATASET_LOADERS = {
    'metabric': load_metabric,
    'support': load_support,
    'gbsg': load_gbsg,
    'whas500': load_whas500,
    'flchain': load_flchain,
    'veterans': load_veterans,
}

if SMOKE:
    DATASETS = ['metabric']
    SEEDS = [0, 1]
    EPOCHS = 600
    ABLATE_DATASETS = ['metabric']
    ABLATE_K = [6, 12]
    N_BOOT = 50
    LR_GRID = [3e-3, 1e-2]        # smoke: 2-point sweep to exercise selection schema
else:
    DATASETS = ['metabric', 'support', 'gbsg', 'whas500', 'flchain']
    SEEDS = [0, 1, 2, 3, 4, 5]
    EPOCHS = 800                  # generous cap; best-epoch selection stops over-training
    ABLATE_DATASETS = ['metabric', 'gbsg']
    ABLATE_K = [6, 12, 24, 48]
    N_BOOT = 300
    LR_GRID = [3e-3, 1e-2, 3e-2]  # each neural net picks its OWN best LR (LR-fair)

VAL_EVERY = 25   # evaluate validation C-index every VAL_EVERY epochs (best-epoch pick)
K_HIDDEN = 24    # matched capacity for the main comparison


def load_dataset(key):
    X, t, e, names, desc = DATASET_LOADERS[key]()
    return dict(X=X, t=t, e=e, names=names, desc=desc, n=len(X), d=X.shape[1],
                cens=float(1.0 - e.mean()), event_rate=float(e.mean()),
                median_t=float(np.median(t)))


def split_standardize(ds, seed, test_frac=0.30, val_frac=0.20):
    """Deterministic train/test split + train-stat standardization, with an
    inner train/val split carved from train for LR + best-epoch selection.
    The test set is NEVER touched during model/LR selection."""
    X, t, e = ds['X'], ds['t'], ds['e']
    n = len(X)
    rng = np.random.RandomState(1000 + seed)
    perm = rng.permutation(n)
    n_te = int(test_frac * n)
    te_idx, tr_all = perm[:n_te], perm[n_te:]
    mu = X[tr_all].mean(0); sd = X[tr_all].std(0) + 1e-6  # stats on full train only
    Xn = (X - mu) / sd
    # inner train/val (fixed by seed) for hyperparameter selection
    n_val = max(int(val_frac * len(tr_all)), 1)
    val_idx, trn_idx = tr_all[:n_val], tr_all[n_val:]
    return dict(
        # full train (used to refit at the chosen LR, and as the Breslow reference)
        Xtr=Xn[tr_all], ttr=t[tr_all], etr=e[tr_all],
        # inner train / validation for selection
        Xtrn=Xn[trn_idx], ttrn=t[trn_idx], etrn=e[trn_idx],
        Xval=Xn[val_idx], tval=t[val_idx], eval=e[val_idx],
        # held-out test
        Xte=Xn[te_idx], tte=t[te_idx], ete=e[te_idx],
        mu=mu, sd=sd, tr_idx=tr_all, te_idx=te_idx, D=X.shape[1])


# ===========================================================================
# 2. MODELS (reused from scripts/yat_deepsurv.py).
# ===========================================================================
class YatLayer(nnx.Module):
    def __init__(self, d_in, k, *, rngs, Winit, b0=1.0, eps0=1.0):
        self.W = nnx.Param(jnp.asarray(Winit))
        self.log_b = nnx.Param(jnp.full((), jnp.log(jnp.expm1(b0))))
        self.log_eps = nnx.Param(jnp.full((), jnp.log(jnp.expm1(eps0))))

    def __call__(self, x):
        b = jax.nn.softplus(self.log_b.value)
        eps = jax.nn.softplus(self.log_eps.value)
        dot = x @ self.W.value.T
        d2 = jnp.sum(x ** 2, -1, keepdims=True) + jnp.sum(self.W.value ** 2, -1) - 2 * dot
        return (dot + b) ** 2 / (d2 + eps)


class YatSurv(nnx.Module):
    """Yat DeepSurv: prototype-kernel trunk -> scalar log-risk."""
    def __init__(self, d_in, k, *, rngs, Winit):
        self.yat = YatLayer(d_in, k, rngs=rngs, Winit=Winit)
        self.read = nnx.Linear(k, 1, use_bias=False, rngs=rngs)

    def features(self, x):
        return self.yat(x)

    def __call__(self, x):
        return self.read(self.yat(x))[:, 0]


class MLPSurv(nnx.Module):
    """Standard DeepSurv: ReLU MLP -> scalar log-risk (Katzman 2018)."""
    def __init__(self, d_in, k, *, rngs):
        self.l1 = nnx.Linear(d_in, k, rngs=rngs)
        self.l2 = nnx.Linear(k, k, rngs=rngs)
        self.out = nnx.Linear(k, 1, use_bias=False, rngs=rngs)

    def features(self, x):
        return jax.nn.relu(self.l2(jax.nn.relu(self.l1(x))))

    def __call__(self, x):
        h = jax.nn.relu(self.l1(x))
        h = jax.nn.relu(self.l2(h))
        return self.out(h)[:, 0]


def cox_ph_loss(logrisk, times, events):
    """Negative Cox partial log-likelihood (Breslow ties)."""
    order = jnp.argsort(-times)
    h = logrisk[order]; ev = events[order]
    m = jnp.max(h)
    log_cum = m + jnp.log(jnp.cumsum(jnp.exp(h - m)) + 1e-12)
    nll = -jnp.sum((h - log_cum) * ev) / (jnp.sum(ev) + 1e-8)
    return nll


@nnx.jit
def train_step(model, opt, xb, tb, eb):
    def loss_fn(m):
        return cox_ph_loss(m(xb), tb, eb)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)
    return loss


# ---------------------------------------------------------------------------
# LR-FAIR neural training. A Kaggle convergence probe showed the Yat trunk and the
# ReLU MLP have DIFFERENT optimal learning rates (Yat converges slowly, ~1e-2; the
# MLP overfits at 1e-2 and prefers ~3e-3). Comparing them at a single shared LR is
# an optimization artifact, not a result. So EACH neural net picks its OWN LR by
# validation, and we report it at its own best LR and best epoch (early-stopping-
# style, so neither is judged on an over-trained endpoint). The classical baselines
# (coxph/coxnet/rsf) are LR-free and anchor the comparison.
# ---------------------------------------------------------------------------
def _c_index(model, X, tt, ee):
    from sksurv.metrics import concordance_index_censored
    if ee.sum() < 2:
        return float('nan')
    r = np.asarray(model(jnp.asarray(X)))
    if not np.isfinite(r).all():
        return float('nan')
    return float(concordance_index_censored(ee.astype(bool), tt, r)[0])


def _snapshot(model):
    return jax.tree.map(lambda a: np.asarray(a).copy(), nnx.state(model))


def _restore(model, snap):
    nnx.update(model, jax.tree.map(jnp.asarray, snap))


def train_net(model, Xtrn, ttrn, etrn, Xval, tval, eval_, *, lr, epochs=EPOCHS,
              wd=1e-4, val_every=VAL_EVERY):
    """Train on the inner-train split; track validation C-index every val_every
    epochs; return (model_restored_to_best_epoch, best_val_c, best_epoch)."""
    opt = nnx.Optimizer(model, optax.adamw(lr, weight_decay=wd), wrt=nnx.Param)
    Xj, tj, ej = jnp.asarray(Xtrn), jnp.asarray(ttrn), jnp.asarray(etrn)
    best_c, best_ep, best_snap = -np.inf, 0, _snapshot(model)
    for ep in range(1, epochs + 1):
        train_step(model, opt, Xj, tj, ej)
        if ep % val_every == 0 or ep == epochs:
            c = _c_index(model, Xval, tval, eval_)
            if np.isfinite(c) and c > best_c:
                best_c, best_ep, best_snap = c, ep, _snapshot(model)
    _restore(model, best_snap)
    return model, float(best_c), int(best_ep)


def yat_init(Xtr, K, seed, init):
    """Prototype seeding: k-means centroids or random training patients."""
    if init == 'kmeans':
        from sklearn.cluster import KMeans
        km = KMeans(K, n_init=3 if SMOKE else 5, random_state=seed).fit(Xtr)
        return km.cluster_centers_.astype('float32')
    else:  # random real patients as prototypes
        rng = np.random.RandomState(2000 + seed)
        idx = rng.choice(len(Xtr), size=K, replace=len(Xtr) < K)
        return Xtr[idx].astype('float32') + 0.01 * rng.randn(K, Xtr.shape[1]).astype('float32')


def build_net(kind, sp, seed, K=K_HIDDEN):
    """Fresh neural model of the requested kind, deterministic in (kind, seed)."""
    if kind == 'mlp':
        return MLPSurv(sp['D'], K, rngs=nnx.Rngs(seed))
    Winit = yat_init(sp['Xtrn'], K, seed, 'kmeans')  # seed on inner-train patients
    return YatSurv(sp['D'], K, rngs=nnx.Rngs(seed), Winit=Winit)


def select_and_fit_net(kind, sp, seed, lr_grid=None):
    """LR-fair training: sweep lr_grid, select this model's OWN best LR by inner
    validation C-index (each candidate early-stopped at its best epoch), then
    return the winning model + a record of the selection. Never compares one net
    at another net's LR. The test set is untouched throughout."""
    lr_grid = lr_grid or LR_GRID
    sweep, best = [], None
    for lr in lr_grid:
        model = build_net(kind, sp, seed)
        model, val_c, best_ep = train_net(
            model, sp['Xtrn'], sp['ttrn'], sp['etrn'],
            sp['Xval'], sp['tval'], sp['eval'], lr=lr)
        sweep.append(dict(lr=lr, val_cindex=val_c, best_epoch=best_ep))
        if best is None or val_c > best['val_cindex']:
            best = dict(model=model, lr=lr, val_cindex=val_c, best_epoch=best_ep)
    return best, sweep


# ===========================================================================
# 3. METRICS. Harrell C, integrated td-AUC, integrated Brier, Cox NLL.
#    Baselines expose a risk score AND a survival function.
# ===========================================================================
from sksurv.metrics import (concordance_index_censored, cumulative_dynamic_auc,
                            integrated_brier_score)
from sksurv.util import Surv


def surv_struct(tt, ee):
    return Surv.from_arrays(event=ee.astype(bool), time=tt)


def breslow_H0(risk_tr, tt, ee):
    """Breslow baseline cumulative hazard H0(t) from training risks."""
    ehr = np.exp(np.clip(risk_tr, -20, 20))
    order = np.argsort(tt)
    tt_s, ee_s, ehr_s = tt[order], ee[order], ehr[order]
    rev_cum = np.cumsum(ehr_s[::-1])[::-1]
    times, H0 = [], []; cum = 0.0
    uniq = np.unique(tt_s[ee_s == 1])
    for tk in uniq:
        d = np.sum((tt_s == tk) & (ee_s == 1))
        denom = rev_cum[np.searchsorted(tt_s, tk, side='left')]
        cum += d / (denom + 1e-12)
        times.append(float(tk)); H0.append(cum)
    return np.array(times), np.array(H0)


def surv_at(times_grid, H0, risk, horizons):
    """S(h|x) = exp(-H0(h) * exp(risk)) at each horizon. -> [n, H]."""
    if len(H0) == 0:
        return np.ones((len(risk), len(horizons)))
    idx = np.searchsorted(times_grid, horizons, side='right') - 1
    idx = np.clip(idx, 0, len(H0) - 1)
    H0_h = H0[idx]
    return np.exp(-np.outer(np.exp(np.clip(risk, -20, 20)), H0_h))


def safe_horizons(ttr, etr, tte, ete):
    """Horizons strictly inside BOTH train and test event-time support."""
    ev_t = tte[ete == 1]
    if len(ev_t) < 5:
        ev_t = tte
    lo = max(np.percentile(ev_t, 10), tte.min() + 1e-3, ttr.min() + 1e-3)
    hi = min(np.percentile(ev_t, 90), tte.max() - 1e-3, ttr.max() - 1e-3)
    if not (hi > lo):
        lo, hi = np.percentile(tte, [25, 75])
    return np.linspace(lo, hi, 12)


def score_predictions(risk_tr, risk_te, sp):
    """Given risk scores + a split, compute the metric quartet.
    risk_tr / risk_te: log-risk (higher = higher risk). Survival curves via Breslow.
    Returns dict + the per-patient test risk (for bootstrap)."""
    ttr, etr = sp['ttr'], sp['etr']
    tte, ete = sp['tte'], sp['ete']
    c = concordance_index_censored(ete.astype(bool), tte, risk_te)[0]
    nll = float(cox_ph_loss(jnp.asarray(risk_te), jnp.asarray(tte), jnp.asarray(ete)))
    times_grid, H0 = breslow_H0(risk_tr, ttr, etr)
    horizons = safe_horizons(ttr, etr, tte, ete)
    Ste = surv_at(times_grid, H0, risk_te, horizons)
    try:
        _, mean_auc = cumulative_dynamic_auc(
            surv_struct(ttr, etr), surv_struct(tte, ete), risk_te, horizons)
    except Exception as ex:
        print(f"    (AUC fail: {ex})", flush=True); mean_auc = np.nan
    try:
        ibs = integrated_brier_score(
            surv_struct(ttr, etr), surv_struct(tte, ete), Ste, horizons)
    except Exception as ex:
        print(f"    (IBS fail: {ex})", flush=True); ibs = np.nan
    return dict(cindex=float(c), mean_auc=float(mean_auc), ibs=float(ibs),
                cox_nll=nll), dict(times_grid=times_grid, H0=H0, horizons=horizons)


def bootstrap_cindex(tte, ete, risk_te, n_boot=N_BOOT, seed=0):
    """Bootstrap CI on the test-set Harrell C-index (resample patients)."""
    rng = np.random.RandomState(5000 + seed)
    n = len(tte); vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if ete[idx].sum() < 2:
            continue
        try:
            vals.append(concordance_index_censored(
                ete[idx].astype(bool), tte[idx], risk_te[idx])[0])
        except Exception:
            continue
    vals = np.array(vals)
    if len(vals) < 5:
        return dict(mean=float('nan'), lo=float('nan'), hi=float('nan'), n=len(vals))
    return dict(mean=float(vals.mean()),
                lo=float(np.percentile(vals, 2.5)),
                hi=float(np.percentile(vals, 97.5)), n=len(vals))


# ---- classical / forest baselines ----------------------------------------
def fit_classical(kind, sp, seed):
    """Return (risk_tr, risk_te) log-risk-like scores. Higher = higher risk."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest
    Xtr, Xte = sp['Xtr'], sp['Xte']
    ytr = surv_struct(sp['ttr'], sp['etr'])
    if kind == 'coxph':
        est = CoxPHSurvivalAnalysis(alpha=1e-2)  # tiny ridge for stability
    elif kind == 'coxnet':
        est = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01,
                                     fit_baseline_model=False, max_iter=100000)
    elif kind == 'rsf':
        est = RandomSurvivalForest(
            n_estimators=50 if SMOKE else 200, min_samples_leaf=15,
            max_features='sqrt', n_jobs=-1, random_state=seed)
    else:
        raise ValueError(kind)
    est.fit(Xtr, ytr)
    # predict() returns a risk score (higher = higher risk) for all three
    return np.asarray(est.predict(Xtr)), np.asarray(est.predict(Xte))


# ===========================================================================
# 4. CALIBRATION.
# ===========================================================================
def km_survival(tt, ee, horizon):
    order = np.argsort(tt); tt_s, ee_s = tt[order], ee[order]
    S = 1.0; n = len(tt_s)
    for i, tk in enumerate(tt_s):
        if tk > horizon: break
        at_risk = n - i
        if ee_s[i] == 1 and at_risk > 0:
            S *= (1 - 1.0 / at_risk)
    return float(S)


def reliability(sp, risk_tr, risk_te, aux, n_groups=3):
    """Predicted vs KM-observed survival, patients grouped by risk tercile."""
    tte, ete = sp['tte'], sp['ete']
    horizons = aux['horizons']
    Sp = surv_at(aux['times_grid'], aux['H0'], risk_te, horizons)  # [n,H]
    cuts = np.percentile(risk_te, np.linspace(0, 100, n_groups + 1)[1:-1])
    grp = np.digitize(risk_te, cuts)
    curve = []
    for g in range(n_groups):
        m = grp == g
        if m.sum() < 2:
            continue
        pred = Sp[m].mean(0)
        obs = np.array([km_survival(tte[m], ete[m], h) for h in horizons])
        curve.append(dict(group=int(g), n=int(m.sum()),
                          pred=pred.tolist(), obs=obs.tolist()))
    # mean absolute reliability error across groups/horizons
    errs = [abs(np.array(c['pred']) - np.array(c['obs'])).mean() for c in curve]
    return dict(horizons=horizons.tolist(), groups=curve,
                mean_abs_error=float(np.mean(errs)) if errs else float('nan'))


def d_calibration(sp, risk_tr, risk_te, aux, n_bins=10):
    """One-sample D-calibration: the predicted survival prob at a patient's own
    event/censor time should be Uniform(0,1) across the cohort. We bin those
    predicted probabilities and report a chi-square-style uniformity statistic
    (lower = better calibrated). Censored patients contribute their conditional
    mass spread over the bins above their probability (Haider et al. 2020)."""
    tte, ete = sp['tte'], sp['ete']
    tg, H0 = aux['times_grid'], aux['H0']
    if len(H0) == 0:
        return dict(stat=float('nan'), bins=[float('nan')] * n_bins)
    # predicted survival prob at each patient's own observed time
    idx = np.clip(np.searchsorted(tg, tte, side='right') - 1, 0, len(H0) - 1)
    H0_i = H0[idx]
    S_at = np.exp(-H0_i * np.exp(np.clip(risk_te, -20, 20)))  # [n] in (0,1]
    S_at = np.clip(S_at, 1e-6, 1.0)
    bins = np.zeros(n_bins)
    edges = np.linspace(0, 1, n_bins + 1)
    for i in range(len(tte)):
        p = S_at[i]
        if ete[i] == 1:
            b = min(int(p * n_bins), n_bins - 1)
            bins[b] += 1.0
        else:
            # censored: uniformly spread 1 unit of mass over [0, p] (bins below p),
            # plus the partial bin containing p.
            b_p = min(int(p * n_bins), n_bins - 1)
            if p <= 0:
                bins[0] += 1.0; continue
            for b in range(b_p):
                bins[b] += (edges[b + 1] - edges[b]) / p
            bins[b_p] += (p - edges[b_p]) / p
    expected = len(tte) / n_bins
    stat = float(((bins - expected) ** 2 / (expected + 1e-9)).sum())
    return dict(stat=stat, bins=bins.tolist(), expected=float(expected))


def subgroup_calibration(sp, risk_tr, risk_te, aux, split_col):
    """Reliability error computed separately on the two halves of a covariate
    median split -- does calibration hold within subgroups?"""
    Xte = sp['Xte']
    med = np.median(Xte[:, split_col])
    out = {}
    for side, mask in (('low', Xte[:, split_col] <= med), ('high', Xte[:, split_col] > med)):
        if mask.sum() < 8:
            out[side] = dict(n=int(mask.sum()), mean_abs_error=float('nan'))
            continue
        sub = dict(sp); sub['Xte'] = Xte[mask]; sub['tte'] = sp['tte'][mask]
        sub['ete'] = sp['ete'][mask]
        rel = reliability(sub, risk_tr, risk_te[mask], aux)
        out[side] = dict(n=int(mask.sum()), mean_abs_error=rel['mean_abs_error'])
    return dict(split_col=int(split_col), out=out)


# ===========================================================================
# 5. OOD RIGOR. kernel-max separating in- vs out-of-distribution.
# ===========================================================================
def yat_kmax(model, X):
    return np.asarray(model.features(jnp.asarray(X))).max(1)


def auroc(pos, neg):
    """AUROC that `pos` scores exceed `neg`. Here IN-dist should score HIGH,
    so pos = in-dist kernel-max, neg = OOD kernel-max. Higher AUROC = better
    separation (model recognizes strangers)."""
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, s))


def auroc_ci(pos, neg, n_boot=N_BOOT, seed=0):
    rng = np.random.RandomState(6000 + seed)
    base = auroc(pos, neg)
    vals = []
    for _ in range(n_boot):
        p = pos[rng.randint(0, len(pos), len(pos))]
        q = neg[rng.randint(0, len(neg), len(neg))]
        vals.append(auroc(p, q))
    vals = np.array([v for v in vals if np.isfinite(v)])
    if len(vals) < 5:
        return dict(auroc=base, lo=float('nan'), hi=float('nan'))
    return dict(auroc=base, lo=float(np.percentile(vals, 2.5)),
                hi=float(np.percentile(vals, 97.5)))


def build_ood(sp, model, seed):
    """Three honest OOD constructions + AUROC of kernel-max separating each from
    real held-out patients. Honesty caveats are recorded per construction."""
    rng = np.random.RandomState(7000 + seed)
    Xte = sp['Xte']
    kmax_in = yat_kmax(model, Xte)
    results = {}

    # (a) FEATURE PERMUTATION: shuffle each covariate independently across
    # patients -> destroys the joint distribution while preserving each marginal.
    # Honesty: marginals are IN-distribution; only the covariance is broken.
    Xperm = Xte.copy()
    for j in range(Xperm.shape[1]):
        Xperm[:, j] = Xperm[rng.permutation(len(Xperm)), j]
    kmax_perm = yat_kmax(model, Xperm)
    results['permutation'] = dict(
        **auroc_ci(kmax_in, kmax_perm, seed=seed),
        caveat='marginals preserved; only the joint dependence is broken (mild OOD)')

    # (b) EXTREME-QUANTILE PATIENTS: push each covariate to +-3..5 sigma (normalized
    # space) -> patients on the far tail of the manifold.
    # Honesty: synthetic, a deliberately strong shift; an easy case by design.
    Xext = Xte[:min(len(Xte), 400)].copy()
    Xext = Xext + rng.choice([-1, 1], Xext.shape) * (3.0 + 2.0 * rng.rand(*Xext.shape))
    kmax_ext = yat_kmax(model, Xext)
    results['extreme_quantile'] = dict(
        **auroc_ci(kmax_in, kmax_ext, seed=seed),
        caveat='synthetic far-tail patients (+/-3..5 sigma); strong shift by design')

    # (c) REAL HELD-OUT SUBGROUP: withhold the top-decile of a CONTINUOUS covariate
    # as "unseen population", the rest as in-dist. A genuine covariate-defined
    # subpopulation the model can still be scored on.
    # Honesty: a real subpopulation, but same dataset -> a covariate shift, not a
    # new data source.
    # Rank covariates by number of distinct values (continuous first); a binary
    # covariate's 90th percentile equals its max, giving an empty top decile, so we
    # walk candidates until the split is non-degenerate.
    n_distinct = np.array([len(np.unique(Xte[:, j])) for j in range(Xte.shape[1])])
    col, out_mask = None, None
    for cand in np.argsort(-n_distinct):
        thr = np.percentile(Xte[:, cand], 90)
        m = Xte[:, cand] > thr
        if m.sum() >= 5 and (~m).sum() >= 5:
            col, out_mask = int(cand), m; break
    if col is not None:
        results['real_subgroup'] = dict(
            **auroc_ci(kmax_in[~out_mask], kmax_in[out_mask], seed=seed),
            split_col=int(col),
            caveat='real covariate-defined subpopulation (top decile of most-variable '
                   'covariate); a covariate shift within one dataset, not a new source')
    else:
        results['real_subgroup'] = dict(auroc=float('nan'), caveat='subgroup too small')

    results['kmax_in_median'] = float(np.median(kmax_in))
    return results


def cross_dataset_ood(model, sp_train_ds, other_key, seed):
    """TRUE cross-dataset covariate shift: score another dataset's patients whose
    feature space matches by dimension. Standardize the other dataset by the TRAIN
    dataset's own mu/sd, so any low kernel-max is genuine distribution shift, not
    a scaling artifact. Only valid when covariate counts match."""
    other = load_dataset(other_key)
    if other['d'] != sp_train_ds['D']:
        return dict(skipped=True, reason=f"dim mismatch {other['d']} vs {sp_train_ds['D']}")
    Xo = (other['X'] - sp_train_ds['mu']) / sp_train_ds['sd']
    kmax_in = yat_kmax(model, sp_train_ds['Xte'])
    kmax_out = yat_kmax(model, Xo.astype('float32'))
    return dict(skipped=False, other=other_key,
                **auroc_ci(kmax_in, kmax_out, seed=seed),
                caveat='TRUE cross-dataset shift: a different clinical population '
                       'standardized by the training dataset stats; features align '
                       'by dimension only, so semantics may differ (honest upper bound '
                       'on how strange a foreign cohort looks)')


# ===========================================================================
# 6. PROTOTYPE PLAUSIBILITY (Yat, METABRIC).
# ===========================================================================
def prototype_plausibility(model, sp, names):
    """Each prototype de-normalized to clinical units + validity checks:
    distance to nearest real training patient, and per-covariate in/out of the
    training min/max range."""
    W = np.asarray(model.yat.W.value)          # [K, D] normalized
    a = np.asarray(model.read.kernel.value)[:, 0]
    Xtr = sp['Xtr']; mu, sd = sp['mu'], sp['sd']
    W_clin = W * sd + mu
    Xtr_clin = Xtr * sd + mu
    tr_min = Xtr.min(0); tr_max = Xtr.max(0)
    table = []
    for u in range(len(W)):
        # nearest real patient (normalized Euclidean) + its distance
        d = np.linalg.norm(Xtr - W[u], axis=1)
        j = int(np.argmin(d))
        in_range = (W[u] >= tr_min) & (W[u] <= tr_max)
        # z-distance beyond range per covariate (0 if inside)
        below = np.maximum(tr_min - W[u], 0.0)
        above = np.maximum(W[u] - tr_max, 0.0)
        excursion = (below + above)  # in sd units
        table.append(dict(
            id=u, readout=float(a[u]),
            covariates_clin=W_clin[u].tolist(),
            nearest_patient_dist=float(d[j]),
            nearest_patient_idx=j,
            nearest_patient_clin=Xtr_clin[j].tolist(),
            n_covars_out_of_range=int((~in_range).sum()),
            max_excursion_sd=float(excursion.max()),
            per_covar_in_range=in_range.astype(int).tolist(),
        ))
    dists = np.array([r['nearest_patient_dist'] for r in table])
    frac_plausible = float(np.mean([r['n_covars_out_of_range'] == 0 for r in table]))
    return dict(feature_names=names, prototypes=table,
                median_nearest_dist=float(np.median(dists)),
                frac_all_covars_in_range=frac_plausible,
                note='A prototype is "clinically plausible" when every covariate '
                     'lands within the observed training min/max and it sits close to '
                     'a real patient; large excursions flag math-useful-but-odd points.')


# ===========================================================================
# 7. MAIN BATTERY.
# ===========================================================================
NET_MODELS = ['mlp', 'yat']
CLASSICAL_MODELS = ['coxph', 'coxnet', 'rsf']
ALL_MODELS = CLASSICAL_MODELS + NET_MODELS

bundle = dict(
    meta=dict(smoke=SMOKE, datasets=DATASETS, seeds=SEEDS, epochs=EPOCHS,
              K_hidden=K_HIDDEN, n_boot=N_BOOT, models=ALL_MODELS,
              lr_grid=LR_GRID, val_frac=0.20, selection='per-model LR by inner '
              'validation C-index + best-epoch (early-stopping); test set untouched',
              note='Research illustration of interpretable survival modeling, not a '
                   'clinical tool. Every number a follow-up post quotes is in here. '
                   'LR-FAIR: each neural net (mlp, yat) is trained and reported AT ITS '
                   'OWN best learning rate; the classical baselines (coxph/coxnet/rsf) '
                   'are LR-free and anchor the comparison.'),
    cards={}, per_seed={}, summary={}, calibration={}, ablation={}, ood={},
    cross_dataset_ood={}, plausibility={})

# preload dataset cards
loaded = {}
for key in DATASETS:
    ds = load_dataset(key)
    loaded[key] = ds
    bundle['cards'][key] = dict(
        desc=ds['desc'], n=ds['n'], n_covariates=ds['d'],
        censoring_rate=ds['cens'], event_rate=ds['event_rate'],
        median_time=ds['median_t'], feature_names=ds['names'])
    print(f"[data] {key}: n={ds['n']} d={ds['d']} cens={ds['cens']:.3f} "
          f"events={ds['event_rate']:.3f}", flush=True)

# ---- main comparison: dataset x model x seed --------------------------------
for key in DATASETS:
    ds = loaded[key]
    bundle['per_seed'][key] = {m: [] for m in ALL_MODELS}
    # keep one seed-0 Yat model + split for calibration / OOD / plausibility
    keep = {}
    for seed in SEEDS:
        sp = split_standardize(ds, seed)
        # classical baselines
        for kind in CLASSICAL_MODELS:
            try:
                rtr, rte = fit_classical(kind, sp, seed)
                met, aux = score_predictions(rtr, rte, sp)
                # RSF risk scores are not Cox log-hazards, so its partial NLL is not
                # comparable to the Cox-family models; drop it rather than emit a
                # meaningless (overflowing) number.
                if kind == 'rsf':
                    met['cox_nll'] = float('nan')
                    met['cox_nll_note'] = 'not comparable: RSF risk is not a Cox log-hazard'
                boot = bootstrap_cindex(sp['tte'], sp['ete'], rte, seed=seed)
                met['boot_cindex'] = boot
                bundle['per_seed'][key][kind].append(met)
            except Exception as ex:
                print(f"  [{key} {kind} seed{seed}] FAIL: {ex}", flush=True)
                bundle['per_seed'][key][kind].append(dict(
                    cindex=float('nan'), mean_auc=float('nan'), ibs=float('nan'),
                    cox_nll=float('nan'), boot_cindex=dict(mean=float('nan')),
                    error=str(ex)))
        # neural baselines -- LR-FAIR: each net picks its OWN best LR by validation
        for kind in NET_MODELS:
            best, sweep = select_and_fit_net(kind, sp, seed)
            model = best['model']
            rtr = np.asarray(model(jnp.asarray(sp['Xtr'])))
            rte = np.asarray(model(jnp.asarray(sp['Xte'])))
            met, aux = score_predictions(rtr, rte, sp)
            met['boot_cindex'] = bootstrap_cindex(sp['tte'], sp['ete'], rte, seed=seed)
            met['selected_lr'] = float(best['lr'])
            met['selected_epoch'] = int(best['best_epoch'])
            met['val_cindex'] = float(best['val_cindex'])
            met['lr_sweep'] = sweep
            bundle['per_seed'][key][kind].append(met)
            if seed == 0:
                keep[kind] = dict(model=model, sp=sp, rtr=rtr, rte=rte, aux=aux,
                                  selected_lr=float(best['lr']))
        print(f"[{key} seed{seed}] " + "  ".join(
            f"{m}:{bundle['per_seed'][key][m][-1]['cindex']:.3f}"
            + (f"@lr{bundle['per_seed'][key][m][-1]['selected_lr']:g}"
               if m in NET_MODELS else '')
            for m in ALL_MODELS), flush=True)

    # summary table: mean +/- std across seeds, per model per metric
    summ = {}
    for m in ALL_MODELS:
        rows = bundle['per_seed'][key][m]
        summ[m] = {}
        for metric in ('cindex', 'mean_auc', 'ibs', 'cox_nll'):
            vals = np.array([r.get(metric, np.nan) for r in rows], dtype=float)
            vals = vals[np.isfinite(vals)]
            summ[m][metric] = dict(
                mean=float(vals.mean()) if len(vals) else float('nan'),
                std=float(vals.std()) if len(vals) else float('nan'),
                n=int(len(vals)))
        # headline C-index bootstrap CI: take seed-0's bootstrap
        b0 = rows[0].get('boot_cindex', {})
        summ[m]['boot_cindex_seed0'] = b0
        # neural nets: record the per-seed selected LR + best epoch (LR-fairness audit)
        if m in NET_MODELS:
            lrs = [r.get('selected_lr') for r in rows if r.get('selected_lr') is not None]
            eps = [r.get('selected_epoch') for r in rows if r.get('selected_epoch') is not None]
            if lrs:
                uniq, cnt = np.unique(lrs, return_counts=True)
                summ[m]['selected_lr_per_seed'] = [float(x) for x in lrs]
                summ[m]['selected_lr_mode'] = float(uniq[int(np.argmax(cnt))])
                summ[m]['selected_epoch_per_seed'] = [int(x) for x in eps]
    bundle['summary'][key] = summ

    # ---- calibration (seed-0 models) ----
    bundle['calibration'][key] = {}
    for m, kk in keep.items():
        rel = reliability(kk['sp'], kk['rtr'], kk['rte'], kk['aux'])
        dcal = d_calibration(kk['sp'], kk['rtr'], kk['rte'], kk['aux'])
        # subgroup split on the most-variable covariate
        split_col = int(np.argmax(kk['sp']['Xte'].var(0)))
        subg = subgroup_calibration(kk['sp'], kk['rtr'], kk['rte'], kk['aux'], split_col)
        bundle['calibration'][key][m] = dict(
            reliability=rel, d_calibration=dcal, subgroup=subg)

    # ---- OOD (seed-0 Yat) ----
    if 'yat' in keep:
        y0 = keep['yat']
        bundle['ood'][key] = build_ood(y0['sp'], y0['model'], seed=0)
        # true cross-dataset shift to the first other dataset with matching dims
        cds = None
        for other in DATASETS:
            if other == key:
                continue
            r = cross_dataset_ood(y0['model'],
                                  dict(Xte=y0['sp']['Xte'], mu=y0['sp']['mu'],
                                       sd=y0['sp']['sd'], D=y0['sp']['D']),
                                  other, seed=0)
            if not r.get('skipped'):
                cds = r; break
        bundle['cross_dataset_ood'][key] = cds or dict(
            skipped=True, reason='no other dataset with matching covariate count')

    # ---- prototype plausibility (Yat only; METABRIC is the clinical showcase) ----
    if 'yat' in keep and key == 'metabric':
        bundle['plausibility'][key] = prototype_plausibility(
            keep['yat']['model'], keep['yat']['sp'], ds['names'])
    print(f"[{key}] summary+calibration+ood+plausibility done "
          f"(elapsed {time.time()-t_start:.0f}s)", flush=True)

# ---- ablation: K x init on the Yat model (1-2 datasets) --------------------
print("=== ABLATION: prototype count K x init ===", flush=True)
for key in ABLATE_DATASETS:
    ds = loaded[key]
    bundle['ablation'][key] = []
    for K in ABLATE_K:
        for init in ('kmeans', 'random'):
            cinds, ibss = [], []
            abl_seeds = SEEDS[:2] if SMOKE else SEEDS[:3]
            for seed in abl_seeds:
                sp = split_standardize(ds, seed)
                Winit = yat_init(sp['Xtrn'], K, seed, init)
                model = YatSurv(sp['D'], K, rngs=nnx.Rngs(seed), Winit=Winit)
                # ablation is Yat-only (no cross-model LR fairness concern); fix lr to
                # the Yat-optimal 1e-2 from the probe, still best-epoch selected on val.
                model, _, _ = train_net(
                    model, sp['Xtrn'], sp['ttrn'], sp['etrn'],
                    sp['Xval'], sp['tval'], sp['eval'], lr=1e-2)
                rtr = np.asarray(model(jnp.asarray(sp['Xtr'])))
                rte = np.asarray(model(jnp.asarray(sp['Xte'])))
                met, _ = score_predictions(rtr, rte, sp)
                cinds.append(met['cindex']); ibss.append(met['ibs'])
            cinds = np.array(cinds); ibss = np.array(ibss)
            bundle['ablation'][key].append(dict(
                K=K, init=init, seeds=list(abl_seeds),
                cindex_mean=float(np.nanmean(cinds)), cindex_std=float(np.nanstd(cinds)),
                ibs_mean=float(np.nanmean(ibss)), ibs_std=float(np.nanstd(ibss))))
            print(f"  [{key}] K={K:3d} {init:7s} C={np.nanmean(cinds):.3f} "
                  f"IBS={np.nanmean(ibss):.3f}", flush=True)

# ===========================================================================
# 8. DUMP the bundle.
# ===========================================================================
bundle['meta']['wall_clock_s'] = float(time.time() - t_start)

with open(OUTDIR / 'trial_results.json', 'w') as f:
    json.dump(jsonable(bundle), f)
print(f"wrote trial_results.json ({(OUTDIR / 'trial_results.json').stat().st_size} bytes)",
      flush=True)

# npz mirror of the numeric-heavy arrays (reliability curves, ablation grids)
npz = {}
for key in DATASETS:
    for m in ALL_MODELS:
        arr = np.array([r.get('cindex', np.nan) for r in bundle['per_seed'][key][m]])
        npz[f'{key}__{m}__cindex'] = arr
np.savez(OUTDIR / 'trial_results.npz', **npz)
print(f"wrote trial_results.npz ({len(npz)} arrays)", flush=True)

# ---- schema self-check: fail loudly if a section is empty / all-NaN --------
def _finite_somewhere(obj):
    if isinstance(obj, float):
        return np.isfinite(obj)
    if isinstance(obj, (int, bool, str)):
        return True
    if isinstance(obj, dict):
        return any(_finite_somewhere(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_finite_somewhere(v) for v in obj)
    return False

print("=== SCHEMA CHECK ===", flush=True)
sections = ['cards', 'per_seed', 'summary', 'calibration', 'ablation', 'ood',
            'cross_dataset_ood', 'plausibility']
ok = True
for s in sections:
    sec = bundle[s]
    populated = bool(sec) and _finite_somewhere(sec)
    print(f"  {s:20s}: {'OK' if populated else 'EMPTY/NaN'} "
          f"({len(sec)} keys)", flush=True)
    if not populated:
        ok = False
# LR-fairness audit: every neural (dataset, seed) row must carry a selected_lr.
lr_ok = True
for key in DATASETS:
    for m in NET_MODELS:
        for r in bundle['per_seed'][key][m]:
            if not np.isfinite(r.get('selected_lr', np.nan)):
                lr_ok = False; print(f"  MISSING selected_lr: {key}/{m}", flush=True)
print(f"  selected_lr present for all neural rows: {'YES' if lr_ok else 'NO'}", flush=True)
for key in DATASETS:
    for m in NET_MODELS:
        lrs = bundle['summary'][key][m].get('selected_lr_per_seed', [])
        print(f"    {key}/{m}: selected LRs {lrs}", flush=True)
ok = ok and lr_ok
print(f"SCHEMA {'ALL POPULATED' if ok else 'HAS EMPTY SECTIONS'}", flush=True)
print(f"=== DONE in {time.time()-t_start:.0f}s -> {OUTDIR} ===", flush=True)
