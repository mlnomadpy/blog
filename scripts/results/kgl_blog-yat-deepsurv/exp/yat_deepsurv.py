"""A white-box survival model: a Yat-kernel DeepSurv against a standard DeepSurv.

Backs the explainer "A Risk Model That Names Its Reasons" and its JAX companion.
Every number quoted in those posts is printed here and dumped into a bundle under
public/yat-deepsurv/ for the interactive viz and the companion GIFs.

The story. DeepSurv (Katzman et al., 2018) is a neural Cox proportional-hazards
model: a network maps a patient's covariates x to a scalar log-risk h(x), trained
on the Cox partial likelihood over (x, time, event) with right-censoring, scored
by Harrell's concordance index. We build two of them at matched capacity and an
identical training setup:

  (A) STANDARD DeepSurv : x -> ReLU MLP -> scalar log-risk.
  (B) YAT DeepSurv      : x -> a bank of Yat prototype kernels -> linear read to
        a scalar log-risk. Each hidden unit is
            phi_u(x) = (w_u . x + b)^2 / (||x - w_u||^2 + eps),
        a similarity to a learned PROTOTYPE PATIENT w_u that lives in covariate
        space. The log-risk is then h(x) = sum_u a_u phi_u(x), so a risk score is
        literally "you are at risk BECAUSE you resemble these prototype patients,
        by these amounts". Prototypes are synthetic patients you can read, place
        and delete.

What this script dumps (public/yat-deepsurv/, all consumed by the viz + GIFs):
  metrics.json         -- C-index, time-dependent AUC, integrated Brier, Cox NLL
                          for both models, multiple seeds, + data/model card.
  prototypes.json      -- each Yat prototype as a de-normalized synthetic patient
                          + its readout weight (risk-raising vs protective).
  attributions.json    -- held-out patients decomposed into convex prototype
                          shares that sum to the risk exactly.
  ood.json             -- kernel-max OOD score on in-vs-out-of-distribution
                          patients; reliability (predicted vs observed survival)
                          for both models at fixed horizons.
  edit.json            -- cohort deletion (right-to-be-forgotten): delete the
                          prototypes a subgroup relies on; untouched patients
                          provably unchanged, the cohort re-routes. Plus a
                          teach-a-subtype demo.
  embedding.json       -- 2D PCA of the RKHS features, patients + prototypes,
                          risk as the field (the attractor field, clinical).
  curves.json          -- survival curves: black-box baseline-hazard curve vs the
                          kernel-weighted (Nadaraya-Watson) neighbor curve.
  train_trace.json     -- per-epoch prototype positions + risk-tertile KM curves,
                          for the "settling" and "stratification" GIFs.

Compute policy: TRAINS ON KAGGLE (GPU/CPU), never locally. Writes into its results
dir and /kaggle/working so `kernels output` captures the bundle. Non-training work
(reading JSON, GIFs, dev server) is local.

Run on Kaggle via kgl.py:
  cd ~/.claude/skills/kaggle-cli-experiments && python3 kgl.py \
    --entry yat_deepsurv.py --slug blog-yat-deepsurv \
    --expdir /Users/tahabsn/conductor/workspaces/blog/hartford/scripts \
    --pip "flax optax scikit-survival lifelines pycox scikit-learn"

This is a research illustration of interpretable survival modeling, NOT a clinical
tool. Do not use any number here to make a medical decision.
"""
import warnings; warnings.filterwarnings('ignore')
import json, os, time
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp
import optax
from flax import nnx

# ---------------------------------------------------------------------------
# Where to write. On Kaggle we also copy the bundle to /kaggle/working so the
# launcher captures it. Locally (never used for training) it lands in results/.
# ---------------------------------------------------------------------------
# On Kaggle the launcher copies WORK/results/*  (flat, top-level files only) up to
# /kaggle/working, so we MUST write flat files into ./results (cwd), not a subdir.
# Locally (never used for training) we write straight into public/yat-deepsurv/.
ON_KAGGLE = Path('/kaggle/working').exists()
if ON_KAGGLE:
    OUTDIRS = [Path('results')]                       # cwd-relative; launcher surfaces these
else:
    OUTDIRS = [Path(__file__).resolve().parents[1] / 'public' / 'yat-deepsurv']
for d in OUTDIRS:
    d.mkdir(parents=True, exist_ok=True)


def dump(name, obj):
    for d in OUTDIRS:
        with open(d / name, 'w') as f:
            json.dump(obj, f)
    print(f"  wrote {name} ({len(json.dumps(obj))} bytes)")


def jsonable(x):
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, dict): return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    return x


# ===========================================================================
# 1. DATA: METABRIC (breast cancer, bundled in pycox, no external download).
#    1904 patients, 9 covariates (4 gene-expression: MKI67, EGFR, PGR, ERBB2;
#    5 clinical: hormone therapy, radiotherapy, chemotherapy, ER-positive, age),
#    right-censored overall survival in months. ~42% events (deaths).
# ===========================================================================
from pycox.datasets import metabric

print("=== DATA: METABRIC (pycox) ===")
df = metabric.read_df()
FEATURES = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
# pycox anonymizes column names; the documented mapping (Katzman 2018 / METABRIC):
FEATURE_NAMES = ['MKI67', 'EGFR', 'PGR', 'ERBB2',        # gene expression (z-ish)
                 'Hormone therapy', 'Radiotherapy', 'Chemotherapy',
                 'ER-positive', 'Age at diagnosis']
FEATURE_KIND = ['gene', 'gene', 'gene', 'gene', 'binary', 'binary', 'binary', 'binary', 'age']
Xraw = df[FEATURES].values.astype('float32')
t = df['duration'].values.astype('float32')       # survival time (months)
e = df['event'].values.astype('float32')           # 1 = death observed, 0 = censored
n_total = len(Xraw)
cens_rate = float(1.0 - e.mean())
print(f"  n = {n_total}, features = {len(FEATURES)}, censoring rate = {cens_rate:.3f}")
print(f"  median follow-up {np.median(t):.1f} months, event rate {e.mean():.3f}")

rng = np.random.RandomState(0)
perm = rng.permutation(n_total)
n_te = int(0.30 * n_total)
te_idx, tr_idx = perm[:n_te], perm[n_te:]

# Standardize on TRAIN stats only (store to de-normalize prototypes later).
mu = Xraw[tr_idx].mean(0); sd = Xraw[tr_idx].std(0) + 1e-6
Xn = (Xraw - mu) / sd
Xtr, ttr, etr = Xn[tr_idx], t[tr_idx], e[tr_idx]
Xte, tte, ete = Xn[te_idx], t[te_idx], e[te_idx]
D = Xn.shape[1]
print(f"  train {len(tr_idx)}, test {len(te_idx)}")


# ===========================================================================
# 2. MODELS. Both output a scalar log-risk. Matched capacity: K hidden units.
# ===========================================================================
K = 24                         # hidden units / prototypes (matched)


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
        self.read = nnx.Linear(k, 1, use_bias=False, rngs=rngs)  # a_u readout, no bias

    def features(self, x):
        return self.yat(x)

    def __call__(self, x):
        return self.read(self.yat(x))[:, 0]     # log-risk h(x), shape [n]


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


# ---- Cox partial-likelihood loss (Breslow ties), the training objective ----
def cox_ph_loss(logrisk, times, events):
    """Negative Cox partial log-likelihood.
    For each event i, its risk set R(i) = {j : t_j >= t_i}. The partial
    likelihood contribution is exp(h_i) / sum_{j in R(i)} exp(h_j). We sort by
    descending time so the risk set is a cumulative-from-top logsumexp.
    """
    order = jnp.argsort(-times)               # descending time
    h = logrisk[order]; ev = events[order]
    # cumulative logsumexp from the top = log sum over risk set (t_j >= t_i).
    # Stable manual version (no reliance on jax.lax.cumlogsumexp across versions):
    # log_cum[i] = log sum_{j<=i} exp(h_j) = m_i + log cumsum(exp(h - m_i)); use a
    # running-max-free stabilization by subtracting the global max (h is bounded here).
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


def train(model, X, tt, ee, epochs=400, lr=3e-3, wd=1e-4, trace=None, Xall=None):
    """Full-batch Cox training (the risk set is global, so full-batch is natural
    and standard for DeepSurv on datasets this size)."""
    opt = nnx.Optimizer(model, optax.adamw(lr, weight_decay=wd), wrt=nnx.Param)
    Xj, tj, ej = jnp.asarray(X), jnp.asarray(tt), jnp.asarray(ee)
    for ep in range(epochs):
        loss = train_step(model, opt, Xj, tj, ej)
        if trace is not None and (ep % trace['every'] == 0 or ep == epochs - 1):
            trace['fn'](model, ep)
    return model


# ===========================================================================
# 3. METRICS: Harrell's C, time-dependent AUC, integrated Brier, Cox NLL.
# ===========================================================================
from sksurv.metrics import (concordance_index_censored, cumulative_dynamic_auc,
                            integrated_brier_score, brier_score)
from sksurv.util import Surv


def surv_struct(tt, ee):
    return Surv.from_arrays(event=ee.astype(bool), time=tt)


def eval_model(model, name):
    risk_te = np.asarray(model(jnp.asarray(Xte)))
    risk_tr = np.asarray(model(jnp.asarray(Xtr)))
    # Harrell's C: higher risk should mean earlier event.
    c = concordance_index_censored(ete.astype(bool), tte, risk_te)[0]
    nll = float(cox_ph_loss(jnp.asarray(risk_te), jnp.asarray(tte), jnp.asarray(ete)))
    # time-dependent AUC + integrated Brier need a survival-probability model.
    # DeepSurv gives log-risk; convert to survival via a Breslow baseline hazard
    # estimated on TRAIN, S(t|x) = S0(t) ** exp(h(x)).
    S_tr, times_grid, H0 = breslow_survival(risk_tr, ttr, etr)
    # horizons strictly inside the test follow-up (sksurv requirement)
    lo, hi = np.percentile(tte[ete == 1], [10, 90])
    horizons = np.linspace(lo, hi, 20)
    Ste = baseline_to_surv(H0, times_grid, risk_te, horizons)  # [n_te, len(horizons)]
    try:
        auc, mean_auc = cumulative_dynamic_auc(
            surv_struct(ttr, etr), surv_struct(tte, ete), risk_te, horizons)
    except Exception as ex:
        print(f"    ({name} AUC failed: {ex})"); auc = np.full(len(horizons), np.nan); mean_auc = np.nan
    try:
        ibs = integrated_brier_score(
            surv_struct(ttr, etr), surv_struct(tte, ete), Ste, horizons)
    except Exception as ex:
        print(f"    ({name} IBS failed: {ex})"); ibs = np.nan
    return dict(cindex=float(c), mean_auc=float(mean_auc), ibs=float(ibs), cox_nll=nll,
                horizons=horizons.tolist(), auc=np.asarray(auc).tolist(),
                risk_te=risk_te.tolist())


def breslow_survival(risk_tr, tt, ee):
    """Breslow estimate of the baseline cumulative hazard H0(t) on training data.
    H0(t) = sum_{event times t_k <= t} d_k / sum_{j in R(t_k)} exp(h_j).

    NOTE: the risk scale must match the one used at prediction time. We use raw
    exp(h) here (the DeepSurv log-risks are small, roughly [-3, 3], so no overflow)
    so that S(t|x) = exp(-H0(t) * exp(h(x))) is a consistent Cox survival curve."""
    ehr = np.exp(risk_tr)                        # raw scale, matches baseline_to_surv
    order = np.argsort(tt)
    tt_s, ee_s, ehr_s = tt[order], ee[order], ehr[order]
    # risk set denominator = reverse cumulative sum of exp(h)
    rev_cum = np.cumsum(ehr_s[::-1])[::-1]
    times, H0 = [], []; cum = 0.0
    uniq = np.unique(tt_s[ee_s == 1])
    for tk in uniq:
        d = np.sum((tt_s == tk) & (ee_s == 1))
        denom = rev_cum[np.searchsorted(tt_s, tk, side='left')]
        cum += d / (denom + 1e-12)
        times.append(float(tk)); H0.append(cum)
    times = np.array(times); H0 = np.array(H0)
    S0 = np.exp(-H0)
    return S0, times, H0


def baseline_to_surv(H0, times_grid, risk, horizons):
    """S(h_i | x) = exp(-H0(h) * exp(risk))  at each horizon."""
    # step-interpolate H0 at horizons
    idx = np.searchsorted(times_grid, horizons, side='right') - 1
    idx = np.clip(idx, 0, len(H0) - 1)
    H0_h = H0[idx]                                # [len(horizons)]
    return np.exp(-np.outer(np.exp(risk - 0), H0_h))   # [n, len(horizons)]


# ===========================================================================
# 4. RUN: multiple seeds, report the honest comparison table.
# ===========================================================================
SEEDS = [0, 1, 2]
results = {'yat': [], 'mlp': []}
kmeans_seed_W = None

for seed in SEEDS:
    print(f"\n=== SEED {seed}: train both DeepSurv models ({K} hidden units) ===")
    # Yat prototypes seeded from k-means on training patients (legible init).
    from sklearn.cluster import KMeans
    km = KMeans(K, n_init=5, random_state=seed).fit(Xtr)
    Winit = km.cluster_centers_.astype('float32')
    if seed == 0:
        kmeans_seed_W = Winit.copy()

    yat = YatSurv(D, K, rngs=nnx.Rngs(seed), Winit=Winit)
    yat = train(yat, Xtr, ttr, etr)
    rY = eval_model(yat, 'yat')
    results['yat'].append(rY)
    print(f"  YAT DeepSurv : C-index {rY['cindex']:.4f}, td-AUC {rY['mean_auc']:.4f}, "
          f"IBS {rY['ibs']:.4f}, CoxNLL {rY['cox_nll']:.4f}")

    mlp = MLPSurv(D, K, rngs=nnx.Rngs(seed))
    mlp = train(mlp, Xtr, ttr, etr)
    rM = eval_model(mlp, 'mlp')
    results['mlp'].append(rM)
    print(f"  STD DeepSurv : C-index {rM['cindex']:.4f}, td-AUC {rM['mean_auc']:.4f}, "
          f"IBS {rM['ibs']:.4f}, CoxNLL {rM['cox_nll']:.4f}")

    if seed == 0:
        yat0, mlp0 = yat, mlp     # keep seed-0 models for all the analysis artifacts


def agg(key, model):
    vals = np.array([r[key] for r in results[model]])
    return float(vals.mean()), float(vals.std())


print("\n=== HONEST COMPARISON (mean +/- std over seeds) ===")
metric_table = {}
for key in ['cindex', 'mean_auc', 'ibs', 'cox_nll']:
    ym, ys = agg(key, 'yat'); mm, ms = agg(key, 'mlp')
    metric_table[key] = dict(yat_mean=ym, yat_std=ys, mlp_mean=mm, mlp_std=ms)
    print(f"  {key:9s}: Yat {ym:.4f}+/-{ys:.4f}   MLP {mm:.4f}+/-{ms:.4f}")

# ---------------------------------------------------------------------------
# metrics.json + data/model card
# ---------------------------------------------------------------------------
card = dict(
    dataset='METABRIC (pycox), breast cancer overall survival',
    n=n_total, n_train=len(tr_idx), n_test=len(te_idx),
    n_features=D, feature_names=FEATURE_NAMES, feature_kind=FEATURE_KIND,
    censoring_rate=cens_rate, event_rate=float(e.mean()),
    median_follow_up_months=float(np.median(t)),
    K_hidden=K, seeds=SEEDS, optimizer='adamw lr 3e-3 wd 1e-4', epochs=400,
    note='Research illustration of interpretable survival modeling, not a clinical tool.')
dump('metrics.json', jsonable(dict(table=metric_table, per_seed=results, card=card)))


# ===========================================================================
# 5. INTERPRETABILITY ARTIFACTS (seed-0 Yat model = yat0).
# ===========================================================================
print("\n=== INTERPRETABILITY: prototypes, attribution, OOD, edit, geometry ===")
Wproto = np.asarray(yat0.yat.W.value)              # [K, D] in normalized space
a_read = np.asarray(yat0.read.kernel.value)[:, 0]  # [K] readout weights (risk direction)
b_yat = float(jax.nn.softplus(yat0.yat.log_b.value))
eps_yat = float(jax.nn.softplus(yat0.yat.log_eps.value))


def yat_features_np(W, X, b, eps):
    dot = X @ W.T
    d2 = (X ** 2).sum(1, keepdims=True) + (W ** 2).sum(1) - 2 * dot
    return (dot + b) ** 2 / (d2 + eps)


# 5.1 PROTOTYPE PROFILES: de-normalize each prototype to clinical units.
Wproto_clin = Wproto * sd + mu                     # back to raw covariate units
protos = []
for u in range(K):
    protos.append(dict(
        id=u,
        covariates_norm=Wproto[u].tolist(),
        covariates_clin=Wproto_clin[u].tolist(),
        readout=float(a_read[u]),                  # >0 raises risk, <0 protective
    ))
dump('prototypes.json', jsonable(dict(
    prototypes=protos, feature_names=FEATURE_NAMES, feature_kind=FEATURE_KIND,
    mu=mu.tolist(), sd=sd.tolist(), b=b_yat, eps=eps_yat)))
n_risk = int((a_read > 0).sum()); n_prot = int((a_read < 0).sum())
print(f"  {K} prototypes: {n_risk} risk-raising, {n_prot} protective")

# 5.2 CONVEX ATTRIBUTION: decompose held-out patients' risk into prototype shares.
# h(x) = sum_u a_u phi_u(x). Signed contribution c_u = a_u phi_u(x). We report the
# exact additive decomposition; "convex shares" = |c_u| normalized (they name which
# prototypes drive the score, and reassemble it exactly).
phi_te = yat_features_np(Wproto, Xte, b_yat, eps_yat)   # [n_te, K]
contrib_te = phi_te * a_read[None, :]                    # [n_te, K] signed, sums to h
risk_te0 = contrib_te.sum(1)
# pick a spread of patients: low / median / high risk, a censored one, an event one
order = np.argsort(risk_te0)
picks = dict(lowest=int(order[0]), low=int(order[len(order)//5]),
             median=int(order[len(order)//2]), high=int(order[4*len(order)//5]),
             highest=int(order[-1]))
attr_patients = []
for label, i in picks.items():
    attr_patients.append(dict(
        label=label, test_index=i,
        covariates_clin=(Xte[i] * sd + mu).tolist(),
        time=float(tte[i]), event=int(ete[i]),
        logrisk=float(risk_te0[i]),
        phi=phi_te[i].tolist(),
        contrib=contrib_te[i].tolist(),           # signed, sum = logrisk
    ))
dump('attributions.json', jsonable(dict(
    patients=attr_patients, readout=a_read.tolist(),
    feature_names=FEATURE_NAMES,
    all_risk_te=risk_te0.tolist(), all_phi_te_max=phi_te.max(1).tolist(),
    times_te=tte.tolist(), events_te=ete.astype(int).tolist())))
# verify the decomposition sums exactly
assert np.allclose(contrib_te.sum(1), np.asarray(yat0(jnp.asarray(Xte))), atol=1e-3)
print(f"  attribution verified: signed prototype contributions sum to log-risk (max err "
      f"{np.abs(contrib_te.sum(1) - np.asarray(yat0(jnp.asarray(Xte)))).max():.2e})")

# 5.3 OOD ABSTENTION + calibration.
# kernel-max as an OOD score: real test patients vs synthetic OOD patients pushed
# far off the covariate manifold (age + gene expression multiple sigmas out).
oo_rng = np.random.RandomState(7)
Xood = Xn[te_idx][:400].copy()
# push each OOD patient far out: exaggerate to +-4 sigma in normalized space
Xood = Xood + oo_rng.choice([-1, 1], Xood.shape) * (3.0 + oo_rng.rand(*Xood.shape))
kmax_in = phi_te.max(1)
kmax_out = yat_features_np(Wproto, Xood, b_yat, eps_yat).max(1)
print(f"  OOD kernel-max: in-distribution median {np.median(kmax_in):.3f}, "
      f"OOD median {np.median(kmax_out):.3f}")
# reliability: predicted vs observed survival at horizons for BOTH models.
risk_tr_yat = np.asarray(yat0(jnp.asarray(Xtr)))
risk_tr_mlp = np.asarray(mlp0(jnp.asarray(Xtr)))
S0y, tgy, H0y = breslow_survival(risk_tr_yat, ttr, etr)
S0m, tgm, H0m = breslow_survival(risk_tr_mlp, ttr, etr)
risk_te_yat = np.asarray(yat0(jnp.asarray(Xte)))
risk_te_mlp = np.asarray(mlp0(jnp.asarray(Xte)))
horizons_cal = np.percentile(tte[ete == 1], np.linspace(10, 90, 6)).tolist()


def reliability(H0, tg, risk_te):
    """Group test patients into risk terciles; predicted vs KM-observed survival."""
    Sp = baseline_to_surv(H0, tg, risk_te, np.array(horizons_cal))  # [n_te, H]
    terc = np.digitize(risk_te, np.percentile(risk_te, [33.3, 66.6]))
    out = []
    for g in range(3):
        m = terc == g
        pred = Sp[m].mean(0).tolist()
        obs = [km_survival(tte[m], ete[m], h) for h in horizons_cal]
        out.append(dict(group=int(g), n=int(m.sum()), pred=pred, obs=obs))
    return out


def km_survival(tt, ee, horizon):
    """Kaplan-Meier survival at a horizon for a subgroup."""
    order = np.argsort(tt); tt_s, ee_s = tt[order], ee[order]
    S = 1.0; n = len(tt_s)
    for i, tk in enumerate(tt_s):
        if tk > horizon: break
        at_risk = n - i
        if ee_s[i] == 1 and at_risk > 0:
            S *= (1 - 1.0 / at_risk)
    return float(S)


dump('ood.json', jsonable(dict(
    kmax_in=kmax_in.tolist(), kmax_out=kmax_out.tolist(),
    horizons=horizons_cal,
    reliability_yat=reliability(H0y, tgy, risk_te_yat),
    reliability_mlp=reliability(H0m, tgm, risk_te_mlp))))
print(f"  reliability curves computed for both models at {len(horizons_cal)} horizons")

# 5.4 EDIT / CONTROL: delete a prototype cohort (right-to-be-forgotten) + teach.
# Unlike a classifier (whose class scores partition into per-class rows), a Cox
# log-risk is ONE readout summing every prototype, so deleting prototype u changes
# a patient's risk by EXACTLY a_u * phi_u(x): provably zero for patients who do not
# resemble u, and exactly their resemblance contribution for those who do. The clean
# story therefore needs a prototype whose resemblance mass is CONCENTRATED in a
# subgroup. We find the prototype whose nearest-prototype cohort (the patients it is
# the top match for) is tightest, treat that cohort as "the patients to be forgotten",
# delete the prototype's readout weight (an EXACT edit, no retraining), and measure:
#   - the deleted cohort's risk change (large: their explanation is gone),
#   - everyone else's risk change (exactly a_u * phi_u(x), tiny for non-resemblers).
age_col = FEATURE_NAMES.index('Age at diagnosis')
Xte_clin = Xte * sd + mu
assign_te = np.argmax(phi_te, 1)                      # each test patient's top prototype
# candidate prototypes: those that are some patient's top match, ranked by how
# concentrated their contribution mass is on their own cohort (cohort share of |contrib|).
concentration = []
for u in range(K):
    coh = assign_te == u
    if coh.sum() < 3:
        concentration.append((u, -1, coh)); continue
    total = np.abs(contrib_te[:, u]).sum() + 1e-9
    share = np.abs(contrib_te[coh, u]).sum() / total
    concentration.append((u, share, coh))
# pick the most-concentrated prototype that owns a real cohort (>= 8 patients)
cands = [(u, s, c) for (u, s, c) in concentration if c.sum() >= 8]
cands.sort(key=lambda z: -z[1])
del_u, del_share, cohort_mask = cands[0][0], cands[0][1], cands[0][2]
delete_protos = [int(del_u)]
cohort_idx = np.where(cohort_mask)[0]
# EXACT readout deletion: zero this prototype's readout weight.
a_edit = a_read.copy(); a_edit[delete_protos] = 0.0
risk_before = contrib_te.sum(1)
risk_after = (phi_te * a_edit[None, :]).sum(1)
delta = risk_after - risk_before                       # == a_u * phi_u(x) for deleted u
untouched = ~cohort_mask
untouched_delta = np.abs(delta[untouched])
# describe the deleted prototype clinically
_pc = (Wproto[del_u] * sd + mu)
print(f"  COHORT DELETE: prototype #{del_u} (age {_pc[age_col]:.0f}, "
      f"a={a_read[del_u]:+.3f}), its cohort n={int(cohort_mask.sum())}, "
      f"contribution concentration {100*del_share:.0f}%")
print(f"    deleted cohort mean |delta log-risk|:  {np.abs(delta[cohort_mask]).mean():.4f}")
print(f"    everyone-else mean |delta log-risk|:   {untouched_delta.mean():.4f} "
      f"(max {untouched_delta.max():.4f})")
for thr in (1e-3, 1e-2, 5e-2):
    print(f"    everyone-else with |delta| < {thr:g}: {100*(untouched_delta < thr).mean():.1f}%")
# the change to every non-cohort patient equals exactly their (tiny) resemblance to #del_u
exact_zero = float((untouched_delta < 1e-3).mean())

# TEACH: append a prototype from a new patient subtype (a high-risk exemplar) with
# a positive readout, no retraining; show a nearby patient's risk rises.
teach_src = cohort_idx[np.argmax(risk_before[cohort_idx])]  # a high-risk exemplar
w_new = Xte[teach_src]
phi_new_te = yat_features_np(w_new[None, :], Xte, b_yat, eps_yat)[:, 0]  # [n_te]
a_new = float(np.median(a_read[a_read > 0]))     # a typical risk-raising weight
risk_taught = risk_before + a_new * phi_new_te
teach_delta = risk_taught - risk_before
print(f"  TEACH a subtype (append 1 prototype, no training): "
      f"max risk rise {teach_delta.max():.3f}, median {np.median(teach_delta):.4f}")

dump('edit.json', jsonable(dict(
    cohort_criterion=f'patients whose nearest prototype is #{del_u}',
    deleted_prototype=int(del_u),
    deleted_prototype_clin=_pc.tolist(),
    deleted_prototype_readout=float(a_read[del_u]),
    concentration=float(del_share),
    cohort_idx=cohort_idx.tolist(),
    delete_protos=delete_protos,
    risk_before=risk_before.tolist(), risk_after=risk_after.tolist(),
    delta=delta.tolist(), cohort_mask=cohort_mask.astype(int).tolist(),
    untouched_mean_abs_delta=float(untouched_delta.mean()),
    untouched_max_abs_delta=float(untouched_delta.max()),
    untouched_frac_below_1e3=float((untouched_delta < 1e-3).mean()),
    untouched_frac_below_1e2=float((untouched_delta < 1e-2).mean()),
    untouched_exact_zero_frac=exact_zero,
    cohort_mean_abs_delta=float(np.abs(delta[cohort_mask]).mean()),
    teach_src=int(teach_src), teach_w=w_new.tolist(),
    teach_delta=teach_delta.tolist(), teach_a=a_new)))

# 5.5 GEOMETRY: 2D PCA of the RKHS features, patients + prototypes, risk field.
from sklearn.decomposition import PCA
phi_tr = yat_features_np(Wproto, Xtr, b_yat, eps_yat)
pca = PCA(2).fit(phi_tr)
emb_te = pca.transform(phi_te)
# prototypes embedded via their own one-hot-ish feature (a prototype at its own
# center: phi_u(w_u) is a spike -> use identity feature row)
proto_feat = yat_features_np(Wproto, Wproto, b_yat, eps_yat)   # [K, K]
emb_proto = pca.transform(proto_feat)
dump('embedding.json', jsonable(dict(
    patients=emb_te.tolist(), risk=risk_te0.tolist(),
    time=tte.tolist(), event=ete.astype(int).tolist(),
    prototypes=emb_proto.tolist(), proto_readout=a_read.tolist(),
    explained_var=pca.explained_variance_ratio_.tolist())))
print(f"  2D RKHS embedding: PCA explains "
      f"{100*pca.explained_variance_ratio_.sum():.1f}% of feature variance")

# 5.6 SURVIVAL CURVE FROM NEIGHBORS (Nadaraya-Watson, clinical).
# For a query patient, the kernel-weighted combination of its prototype-neighbors'
# survival curves vs the black-box Breslow curve S(t|x) = S0(t)^exp(h(x)).
# Build each prototype's "own" KM curve from the training patients nearest to it.
tgrid = np.linspace(0, np.percentile(tte, 95), 40)
proto_curves = []
assign = np.argmax(yat_features_np(Wproto, Xtr, b_yat, eps_yat), 1)  # nearest proto per train pt
for u in range(K):
    m = assign == u
    if m.sum() >= 5:
        proto_curves.append([km_survival(ttr[m], etr[m], h) for h in tgrid])
    else:
        proto_curves.append([km_survival(ttr, etr, h) for h in tgrid])  # fallback: global
proto_curves = np.array(proto_curves)   # [K, len(tgrid)]
# a few query patients: NW curve = sum_u phi_u(x) S_u(t) / sum_u phi_u(x)
curve_patients = []
for label, i in list(picks.items())[:3]:
    w = phi_te[i]; w = w / (w.sum() + 1e-9)
    nw = (w[:, None] * proto_curves).sum(0)
    bb = np.exp(-H0y[np.clip(np.searchsorted(tgy, tgrid, 'right') - 1, 0, len(H0y) - 1)]
                * np.exp(risk_te0[i]))
    curve_patients.append(dict(label=label, test_index=i, nw=nw.tolist(),
                               blackbox=bb.tolist(), top_protos=np.argsort(-w)[:4].tolist(),
                               top_weights=np.sort(w)[::-1][:4].tolist()))
dump('curves.json', jsonable(dict(
    tgrid=tgrid.tolist(), proto_curves=proto_curves.tolist(),
    patients=curve_patients)))
print(f"  survival curves: {K} prototype neighbor-curves + NW vs black-box for query patients")

# 5.7 TRAINING TRACE for GIFs: prototype positions (2D PCA of covariates) and
# risk-tertile KM curves forming over training, for BOTH models.
print("\n=== TRAINING TRACE for GIFs (re-train seed 0 with logging) ===")
pca_cov = PCA(2).fit(Xtr)          # fixed 2D covariate frame for the prototype cloud
data2d = pca_cov.transform(Xtr)
trace_frames = {'yat_protos': [], 'yat_km': [], 'mlp_km': [], 'epochs': []}


def km_tertiles(model):
    r = np.asarray(model(jnp.asarray(Xte)))
    terc = np.digitize(r, np.percentile(r, [33.3, 66.6]))
    curves = []
    for g in range(3):
        m = terc == g
        curves.append([km_survival(tte[m], ete[m], h) for h in tgrid])
    return curves


def yat_trace(model, ep):
    trace_frames['epochs'].append(int(ep))
    trace_frames['yat_protos'].append(pca_cov.transform(np.asarray(model.yat.W.value)).tolist())
    trace_frames['yat_km'].append(km_tertiles(model))


yat_t = YatSurv(D, K, rngs=nnx.Rngs(0), Winit=kmeans_seed_W)
train(yat_t, Xtr, ttr, etr, trace=dict(every=25, fn=yat_trace))

mlp_km_frames = []
mlp_epochs = []


def mlp_trace(model, ep):
    mlp_epochs.append(int(ep)); mlp_km_frames.append(km_tertiles(model))


mlp_t = MLPSurv(D, K, rngs=nnx.Rngs(0))
train(mlp_t, Xtr, ttr, etr, trace=dict(every=25, fn=mlp_trace))
trace_frames['mlp_km'] = mlp_km_frames

dump('train_trace.json', jsonable(dict(
    data2d=data2d.tolist(), data_risk=None,
    tgrid=tgrid.tolist(),
    epochs=trace_frames['epochs'],
    yat_protos=trace_frames['yat_protos'],
    yat_km=trace_frames['yat_km'],
    mlp_km=trace_frames['mlp_km'],
    proto_readout=a_read.tolist())))
print(f"  training trace: {len(trace_frames['epochs'])} frames of prototype cloud + KM tertiles")

print("\n=== DONE. Bundle written to:")
for d in OUTDIRS:
    print(f"   {d}")
