"""Calibration of a Yat-kernel MLP vs a matched ReLU MLP on Fashion-MNIST.

Provenance for every number in `calibration-of-a-bounded-net.mdx`.

Setup mirrors the C1 pair (`scripts/yat_fmnist_export.py`): one hidden layer of
K=50 units, 784 -> 50 -> 10, adam 4e-3, batch 256, 12 epochs. The Yat net's
prototypes are seeded from training images; the ReLU net is the same shape.
Train on 55k Fashion-MNIST images, hold out 5k as a validation split (used ONLY
to fit temperature scaling), evaluate on the full 10k test set. MNIST test
images are the out-of-distribution probe.

Confidence is defined IDENTICALLY for both nets: the softmax of the final
linear readout's logits, conf(x) = max_c softmax(z(x))_c. For the Yat net the
logits are the linear readout of the K kernel scores (z = A^T phi(x) + b), for
the ReLU net the linear readout of the K ReLU features. Same head, same
softmax, same pipeline; only the feature map differs. No kernel-score softmax,
no normalized-activation confidence: that is the fairness control.

Computed per net, per seed (seeds 0, 1, 2):
  - test accuracy
  - reliability diagram: 15 equal-width confidence bins on [0,1]
    (per-bin count, mean confidence, accuracy)
  - ECE (expected calibration error, the count-weighted |acc - conf| sum)
  - NLL and Brier score on the test set
  - temperature T fit on the held-out 5k by minimizing NLL of z/T,
    then post-scaling ECE / NLL on the test set
  - OOD (MNIST): confidence histogram, mean/median confidence,
    fraction above 0.9 and 0.99
  - OOD separation (AUROC of Fashion-test vs MNIST): using max softmax
    confidence (both nets), max logit (both nets), and, for the Yat net,
    the max kernel score max_u phi_u(x), the pre-softmax field magnitude
    that C1's abstention rule reads. This is the bridge back to C1: the
    same network, three ways of asking "do you recognize this?"

Exports (public/yat-calibration/):
  summary.json   config + per-seed metrics + mean/std
  bins.json      seed-0 reliability bins, both nets, pre and post temperature
  hist.json      seed-0 max-confidence histograms, test and MNIST, both nets
  logits.json    seed-0 logits subsample (4000 test + labels, 2000 MNIST) for
                 the in-browser temperature / threshold panels, plus the Yat
                 net's max kernel score for the same examples
  channels.json  per-seed subsample (3000 test, 1500 MNIST) of the Yat net's
                 two OOD channels: max kernel score and max softmax
                 confidence, so the field-vs-softmax panel can flip seeds and
                 show which channel is stable

Trained logits are cached in /tmp/yat_calibration_cache.npz so metric/export
tweaks re-run in seconds; delete the cache to retrain.

Run: python3 scripts/yat_calibration.py
"""
import warnings; warnings.filterwarnings('ignore')
import json
from pathlib import Path

import numpy as np
import torchvision
import jax, jax.numpy as jnp
import optax
from flax import nnx
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public' / 'yat-calibration'
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]
K, EPOCHS, BATCH, LR = 50, 12, 256, 4e-3
N_BINS = 15
CLS = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

# ---------------------------------------------------------------- data
tr = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=True, download=True)
te = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=False, download=True)
X = tr.data.numpy().reshape(-1, 784).astype('float32') / 255.0
y = tr.targets.numpy().astype('int32')
Xte = te.data.numpy().reshape(-1, 784).astype('float32') / 255.0
yte = te.targets.numpy().astype('int32')
mn = torchvision.datasets.MNIST('/tmp/mnist', train=False, download=True)
Xood = mn.data.numpy().reshape(-1, 784).astype('float32') / 255.0

# fixed split: train 55k, val 5k (temperature-fitting only)
perm = np.random.RandomState(123).permutation(len(X))
Xtr, ytr = X[perm[:55000]], y[perm[:55000]]
Xval, yval = X[perm[55000:]], y[perm[55000:]]


# ---------------------------------------------------------------- models
class YatLayer(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs, Winit, b0=1.0, eps0=1.0):
        self.W = nnx.Param(jnp.asarray(Winit))
        self.log_b = nnx.Param(jnp.full((), jnp.log(jnp.expm1(b0))))
        self.log_eps = nnx.Param(jnp.full((), jnp.log(jnp.expm1(eps0))))

    def __call__(self, x):
        b = jax.nn.softplus(self.log_b.value)
        eps = jax.nn.softplus(self.log_eps.value)
        dot = x @ self.W.value.T
        d2 = jnp.sum(x ** 2, -1, keepdims=True) + jnp.sum(self.W.value ** 2, -1) - 2 * dot
        return (dot + b) ** 2 / (d2 + eps)


class YatMLP(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs, Winit):
        self.yat = YatLayer(rngs=rngs, Winit=Winit)
        self.readout = nnx.Linear(K, 10, rngs=rngs)

    def __call__(self, x):
        return self.readout(self.yat(x))


class ReluMLP(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(784, K, rngs=rngs)
        self.l2 = nnx.Linear(K, 10, rngs=rngs)

    def __call__(self, x):
        return self.l2(jax.nn.relu(self.l1(x)))


@nnx.jit
def train_step(model, opt, xb, yb):
    def loss_fn(m):
        return optax.softmax_cross_entropy_with_integer_labels(m(xb), yb).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)
    return loss


def train(model, seed):
    opt = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)
    for epoch in range(EPOCHS):
        p = np.random.RandomState(1000 * seed + epoch).permutation(len(Xtr))
        for i in range(0, len(Xtr) - BATCH, BATCH):
            j = p[i:i + BATCH]
            train_step(model, opt, jnp.asarray(Xtr[j]), jnp.asarray(ytr[j]))
    return model


def logits_of(model, Xq, chunk=5000):
    out = [np.asarray(model(jnp.asarray(Xq[i:i + chunk])))
           for i in range(0, len(Xq), chunk)]
    return np.concatenate(out).astype('float64')


# ---------------------------------------------------------------- metrics
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def reliability(z, yq, n_bins=N_BINS):
    """15 equal-width bins on confidence; returns per-bin count/conf/acc + ECE."""
    p = softmax(z)
    conf = p.max(1)
    pred = p.argmax(1)
    correct = (pred == yq).astype('float64')
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    count = np.zeros(n_bins); mconf = np.zeros(n_bins); macc = np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        count[b] = m.sum()
        if count[b] > 0:
            mconf[b] = conf[m].mean()
            macc[b] = correct[m].mean()
    ece = float((count / len(yq) * np.abs(macc - mconf)).sum())
    return dict(count=count, conf=mconf, acc=macc, ece=ece)


def nll_of(z, yq):
    p = softmax(z)
    return float(-np.log(np.clip(p[np.arange(len(yq)), yq], 1e-12, 1)).mean())


def brier_of(z, yq):
    p = softmax(z)
    onehot = np.eye(10)[yq]
    return float(((p - onehot) ** 2).sum(1).mean())


def fit_temperature(z_val, y_val):
    """Minimize held-out NLL of z/T over T > 0."""
    def obj(logT):
        return nll_of(z_val / np.exp(logT), y_val)
    r = minimize_scalar(obj, bounds=(np.log(0.05), np.log(20.0)), method='bounded')
    return float(np.exp(r.x))


def conf_hist(z, n=40):
    conf = softmax(z).max(1)
    h, _ = np.histogram(conf, bins=n, range=(0, 1))
    return (h / len(conf)).tolist(), conf


def ood_stats(z):
    conf = softmax(z).max(1)
    return dict(mean=float(conf.mean()), median=float(np.median(conf)),
                frac90=float((conf > 0.9).mean()), frac99=float((conf > 0.99).mean()))


def auroc(pos, neg):
    """AUROC of score separating pos (in-distribution) from neg (OOD),
    higher score = more in-distribution. Rank-based (Mann-Whitney)."""
    s = np.concatenate([pos, neg])
    ranks = np.empty(len(s))
    order = np.argsort(s, kind='mergesort')
    sr = s[order]
    i = 0
    while i < len(sr):                       # average ranks over ties
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------- run
CACHE = Path('/tmp/yat_calibration_cache.npz')
if not CACHE.exists():
    blobs = {}
    for seed in SEEDS:
        Winit = Xtr[np.random.RandomState(seed + 1).permutation(len(Xtr))[:K]]
        nets = {
            'yat': train(YatMLP(rngs=nnx.Rngs(seed), Winit=Winit), seed),
            'relu': train(ReluMLP(rngs=nnx.Rngs(seed)), seed),
        }
        for name, net in nets.items():
            blobs[f'{name}{seed}_te'] = logits_of(net, Xte)
            blobs[f'{name}{seed}_val'] = logits_of(net, Xval)
            blobs[f'{name}{seed}_ood'] = logits_of(net, Xood)
        # the Yat net's pre-softmax field magnitude: max kernel score
        yat = nets['yat']
        for tag, Xq in (('te', Xte), ('ood', Xood)):
            ks = [np.asarray(yat.yat(jnp.asarray(Xq[i:i + 5000]))).max(1)
                  for i in range(0, len(Xq), 5000)]
            blobs[f'kmax{seed}_{tag}'] = np.concatenate(ks).astype('float64')
        print(f'seed {seed}: trained + cached')
    np.savez_compressed(CACHE, **blobs)
Z = np.load(CACHE)

results = {'yat': [], 'relu': []}
seed0_export = {}

for seed in SEEDS:
    for name in ('yat', 'relu'):
        z_te = Z[f'{name}{seed}_te']
        z_val = Z[f'{name}{seed}_val']
        z_ood = Z[f'{name}{seed}_ood']

        acc = float((z_te.argmax(1) == yte).mean())
        rel = reliability(z_te, yte)
        T = fit_temperature(z_val, yval)
        rel_T = reliability(z_te / T, yte)
        au = dict(msp=auroc(softmax(z_te).max(1), softmax(z_ood).max(1)),
                  maxlogit=auroc(z_te.max(1), z_ood.max(1)))
        if name == 'yat':
            au['kmax'] = auroc(Z[f'kmax{seed}_te'], Z[f'kmax{seed}_ood'])
        row = dict(
            seed=seed, acc=acc,
            ece=rel['ece'], nll=nll_of(z_te, yte), brier=brier_of(z_te, yte),
            T=T, ece_T=rel_T['ece'], nll_T=nll_of(z_te / T, yte),
            ood=ood_stats(z_ood), auroc=au,
            id_conf_mean=float(softmax(z_te).max(1).mean()),
        )
        results[name].append(row)
        print(f"seed {seed} {name:4s}  acc {acc*100:5.1f}%  ECE {rel['ece']*100:5.2f}%  "
              f"NLL {row['nll']:.4f}  Brier {row['brier']:.4f}  T {T:.3f}  "
              f"ECE@T {rel_T['ece']*100:5.2f}%  OOD conf mean {row['ood']['mean']:.3f}  "
              f"OOD>0.9 {row['ood']['frac90']*100:4.1f}%  "
              f"AUROC msp {au['msp']:.3f} maxlogit {au['maxlogit']:.3f}"
              + (f" kmax {au['kmax']:.3f}" if 'kmax' in au else ''))

        if seed == SEEDS[0]:
            hist_id, _ = conf_hist(z_te)
            hist_ood, _ = conf_hist(z_ood)
            seed0_export[name] = dict(
                z_te=z_te, z_ood=z_ood, T=T,
                rel=rel, rel_T=rel_T, hist_id=hist_id, hist_ood=hist_ood,
                auroc=au,
            )

# ---------------------------------------------------------------- exports
def mean_std(rows, key):
    v = np.array([r[key] for r in rows])
    return dict(mean=float(v.mean()), std=float(v.std()))


summary = dict(
    config=dict(K=K, epochs=EPOCHS, batch=BATCH, lr=LR, seeds=SEEDS,
                train=55000, val=5000, test=len(yte), ood='MNIST test (10k)',
                bins=N_BINS,
                confidence='max softmax probability of the final linear readout logits, identical pipeline for both nets'),
    classes=CLS,
    per_seed=results,
    agg={name: {k: mean_std(rows, k) for k in ('acc', 'ece', 'nll', 'brier', 'T', 'ece_T', 'nll_T', 'id_conf_mean')}
         for name, rows in results.items()},
    ood_agg={name: {k: dict(mean=float(np.mean([r['ood'][k] for r in rows])),
                            std=float(np.std([r['ood'][k] for r in rows])))
                    for k in ('mean', 'median', 'frac90', 'frac99')}
             for name, rows in results.items()},
    auroc_agg={name: {k: dict(mean=float(np.mean([r['auroc'][k] for r in rows])),
                              std=float(np.std([r['auroc'][k] for r in rows])))
                      for k in results[name][0]['auroc']}
               for name, rows in results.items()},
)
json.dump(summary, open(OUT / 'summary.json', 'w'), indent=1)

bins = {name: dict(
    T=d['T'],
    pre=dict(count=d['rel']['count'].tolist(), conf=np.round(d['rel']['conf'], 4).tolist(),
             acc=np.round(d['rel']['acc'], 4).tolist(), ece=d['rel']['ece']),
    post=dict(count=d['rel_T']['count'].tolist(), conf=np.round(d['rel_T']['conf'], 4).tolist(),
              acc=np.round(d['rel_T']['acc'], 4).tolist(), ece=d['rel_T']['ece']),
) for name, d in seed0_export.items()}
bins['nBins'] = N_BINS
json.dump(bins, open(OUT / 'bins.json', 'w'))

hist = {name: dict(test=np.round(d['hist_id'], 5).tolist(),
                   ood=np.round(d['hist_ood'], 5).tolist())
        for name, d in seed0_export.items()}
hist['nBins'] = 40
json.dump(hist, open(OUT / 'hist.json', 'w'))

# logits subsample for the live panels (seed 0). Fixed subsample, same indices
# for both nets so the panels compare like with like.
sub = np.random.RandomState(7).permutation(len(yte))[:4000]
sub_ood = np.random.RandomState(7).permutation(len(Xood))[:2000]
logits = dict(
    labels=yte[sub].tolist(),
    yat=dict(test=np.round(seed0_export['yat']['z_te'][sub], 2).tolist(),
             ood=np.round(seed0_export['yat']['z_ood'][sub_ood], 2).tolist(),
             T=seed0_export['yat']['T'],
             kmaxTest=np.round(Z['kmax0_te'][sub], 2).tolist(),
             kmaxOod=np.round(Z['kmax0_ood'][sub_ood], 2).tolist()),
    relu=dict(test=np.round(seed0_export['relu']['z_te'][sub], 2).tolist(),
              ood=np.round(seed0_export['relu']['z_ood'][sub_ood], 2).tolist(),
              T=seed0_export['relu']['T']),
)
json.dump(logits, open(OUT / 'logits.json', 'w'))

# per-seed field-vs-softmax channels for the Yat net (subsampled)
ch_te = np.random.RandomState(11).permutation(len(yte))[:3000]
ch_ood = np.random.RandomState(11).permutation(len(Xood))[:1500]
channels = dict(seeds=SEEDS, net='yat')
for seed in SEEDS:
    conf_te = softmax(Z[f'yat{seed}_te']).max(1)
    conf_ood = softmax(Z[f'yat{seed}_ood']).max(1)
    channels[f's{seed}'] = dict(
        kmaxTest=np.round(Z[f'kmax{seed}_te'][ch_te], 1).tolist(),
        kmaxOod=np.round(Z[f'kmax{seed}_ood'][ch_ood], 1).tolist(),
        confTest=np.round(conf_te[ch_te], 3).tolist(),
        confOod=np.round(conf_ood[ch_ood], 3).tolist(),
    )
json.dump(channels, open(OUT / 'channels.json', 'w'))

print('\n=== aggregate (mean over seeds 0,1,2) ===')
for name in ('yat', 'relu'):
    a = summary['agg'][name]
    o = summary['ood_agg'][name]
    u = summary['auroc_agg'][name]
    print(f"{name:4s}  acc {a['acc']['mean']*100:.1f}±{a['acc']['std']*100:.1f}%  "
          f"ECE {a['ece']['mean']*100:.2f}±{a['ece']['std']*100:.2f}%  "
          f"NLL {a['nll']['mean']:.4f}  Brier {a['brier']['mean']:.4f}  "
          f"T {a['T']['mean']:.3f}±{a['T']['std']:.3f}  ECE@T {a['ece_T']['mean']*100:.2f}%  "
          f"OOD conf {o['mean']['mean']:.3f}  OOD>0.9 {o['frac90']['mean']*100:.1f}%  "
          f"AUROC msp {u['msp']['mean']:.3f}±{u['msp']['std']:.3f}"
          + (f"  kmax {u['kmax']['mean']:.3f}±{u['kmax']['std']:.3f}" if 'kmax' in u else ''))
print('wrote', OUT)
