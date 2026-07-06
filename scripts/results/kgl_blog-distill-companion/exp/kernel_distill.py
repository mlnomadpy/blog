"""Distillation as kernel transfer, on Fashion-MNIST (JAX + Flax NNX + Optax).

Five runs, one question: how much of a teacher's "knowledge" is the class-similarity
GEOMETRY its outputs carry, rather than the answers themselves?

  1. TEACHER        small CNN trained with labels (the reference geometry). From its
                    softened softmax we extract the class-similarity kernel
                    S(T) = E[softmax(z/T) softmax(z/T)^T] and its latent class-mean Gram.
  2. STUDENT-KD     standard distillation baseline: KL to the teacher's soft targets
                    at temperature T, no hard labels.
  3. STUDENT-KERNEL the centerpiece: NO labels, NO per-class targets, no teacher head
                    to imitate. Each batch, the teacher's softened outputs define a
                    pairwise similarity G_ij = cos(p_i, p_j); the student's normalized
                    embeddings must reproduce that Gram. Purely relational.
  4. RANDOM         control: probe an untrained student.
  5. LABEL          ceiling: same student arch trained on the true labels.

Reported: linear + nearest-centroid probe accuracy for all five, the eigenvalue
spectrum of each latent class-mean Gram (the inheritance claim made spectral), CKA
between student and teacher class-mean Grams, confusion-structure agreement, and a
temperature sweep of S(T). Everything is exported to public/kernel-distill/ for the
in-browser visualizations.

Run: python scripts/kernel_distill.py            (local CPU, ~10 min)
Seeds: teacher/KD/label/random use seed 0; the kernel student runs seeds 0, 1, 2.
"""
import warnings; warnings.filterwarnings('ignore')
import os, json, time, numpy as np
import jax, jax.numpy as jnp, optax
from flax import nnx
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

CLS = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
TEMP = 4.0                     # the distillation / kernel temperature used everywhere
N_TRAIN, BATCH = 20000, 256
EPOCHS_T, EPOCHS_S = 6, 8
OUT = 'public/kernel-distill'

# ── data ──────────────────────────────────────────────────────────────────────
tr = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=True, download=True)
te = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=False, download=True)
Xtr_all = (tr.data.numpy().astype('float32') / 255.0)[..., None]
ytr_all = tr.targets.numpy()
Xte = (te.data.numpy().astype('float32') / 255.0)[..., None]
yte = te.targets.numpy()
sub = np.random.RandomState(0).permutation(len(Xtr_all))[:N_TRAIN]
Xtr, ytr = Xtr_all[sub], ytr_all[sub]


# ── model ─────────────────────────────────────────────────────────────────────
class CNN(nnx.Module):
    def __init__(self, c1, c2, d, rngs):
        self.conv1 = nnx.Conv(1, c1, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(c1, c2, kernel_size=(3, 3), rngs=rngs)
        self.fc = nnx.Linear(c2 * 7 * 7, d, rngs=rngs)
        self.head = nnx.Linear(d, 10, rngs=rngs)

    def embed(self, x):                      # the latent the probes and Grams read
        x = nnx.relu(self.conv1(x)); x = nnx.max_pool(x, (2, 2), strides=(2, 2))
        x = nnx.relu(self.conv2(x)); x = nnx.max_pool(x, (2, 2), strides=(2, 2))
        return self.fc(x.reshape(x.shape[0], -1))

    def __call__(self, x):
        return self.head(nnx.relu(self.embed(x)))


def teacher_net(seed): return CNN(16, 32, 64, nnx.Rngs(seed))
def student_net(seed): return CNN(8, 16, 32, nnx.Rngs(seed))


# ── batched forward helpers ───────────────────────────────────────────────────
@nnx.jit
def _logits(model, x): return model(x)

@nnx.jit
def _embed(model, x): return model.embed(x)

def batched(fn, model, X, bs=2000):
    return np.concatenate([np.asarray(fn(model, jnp.asarray(X[i:i + bs]))) for i in range(0, len(X), bs)])

def logits_of(model, X): return batched(_logits, model, X)
def embed_of(model, X): return batched(_embed, model, X)


# ── training steps ────────────────────────────────────────────────────────────
@nnx.jit
def step_ce(model, opt, x, y):
    def loss_fn(m):
        return optax.softmax_cross_entropy_with_integer_labels(m(x), y).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)
    return loss

@nnx.jit
def step_kd(model, opt, x, p_t):
    # KL(teacher soft targets || student soft output) at temperature TEMP, no labels
    def loss_fn(m):
        logq = jax.nn.log_softmax(m(x) / TEMP)
        return (TEMP ** 2) * (p_t * (jnp.log(p_t + 1e-12) - logq)).sum(1).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)
    return loss

@nnx.jit
def step_kernel(model, opt, x, pn):
    # relational only: the student's embedding cosine Gram must match the Gram the
    # teacher's softened outputs assign to this batch. No labels, no class targets.
    def loss_fn(m):
        g_t = pn @ pn.T
        e = m.embed(x)
        en = e / (jnp.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
        g_s = en @ en.T
        mask = 1.0 - jnp.eye(x.shape[0])
        return (((g_s - g_t) ** 2) * mask).sum() / mask.sum()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)
    return loss


# ── measurement kit ───────────────────────────────────────────────────────────
def unit(v): return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

def acc_direct(model):
    return 100.0 * (logits_of(model, Xte).argmax(1) == yte).mean()

def probes(model):
    """Linear + nearest-centroid probe on the frozen latent. Labels are used only
    here, as the measuring stick, never in the student's training signal."""
    Etr, Ete = embed_of(model, Xtr), embed_of(model, Xte)
    lin = LogisticRegression(max_iter=1000).fit(Etr, ytr)
    linear = 100.0 * lin.score(Ete, yte)
    mu = np.stack([unit(Etr)[ytr == c].mean(0) for c in range(10)])
    pred = (unit(Ete) @ unit(mu).T).argmax(1)
    return linear, 100.0 * (pred == yte).mean(), pred

def class_mean_gram(model, X=None, y=None):
    E = unit(embed_of(model, Xte if X is None else X))
    yy = yte if y is None else y
    mu = np.stack([E[yy == c].mean(0) for c in range(10)])
    muc = mu - mu.mean(0)
    return muc @ muc.T

def spectrum(G):
    w = np.linalg.eigvalsh(G)[::-1]
    w = np.clip(w, 0, None)
    return w / (w.sum() + 1e-12)

def cka(Ga, Gb):
    """centered-kernel alignment between two 10x10 class-mean Grams"""
    H = np.eye(10) - np.ones((10, 10)) / 10
    A, B = H @ Ga @ H, H @ Gb @ H
    return float((A * B).sum() / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12))

def confusion(pred):
    cm = np.zeros((10, 10))
    for t, p in zip(yte, pred): cm[t, p] += 1
    return cm / cm.sum(1, keepdims=True)

def conf_corr(cma, cmb):
    off = ~np.eye(10, dtype=bool)
    a, b = cma[off], cmb[off]
    return float(np.corrcoef(a, b)[0, 1])

def train(model, stepfn, targets, epochs, tag, per_epoch=None):
    opt = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    losses = []
    for ep in range(epochs):
        perm = np.random.RandomState(100 + ep).permutation(N_TRAIN)
        tot = 0.0
        for i in range(0, N_TRAIN, BATCH):
            idx = perm[i:i + BATCH]
            t = targets[idx] if targets is not None else jnp.asarray(ytr[idx])
            tot += float(stepfn(model, opt, jnp.asarray(Xtr[idx]), jnp.asarray(t)))
        losses.append(tot / (N_TRAIN // BATCH))
        line = f'  {tag} epoch {ep + 1}/{epochs}  loss {losses[-1]:.4f}'
        if per_epoch is not None: line += '  ' + per_epoch(model, ep)
        print(line, flush=True)
    return losses


# ══ 1. TEACHER ════════════════════════════════════════════════════════════════
t0 = time.time()
print('=== teacher (labels, CE) ===')
teacher = teacher_net(0)
train(teacher, step_ce, None, EPOCHS_T, 'teacher')
t_acc = acc_direct(teacher)
t_lin, t_cen, t_cen_pred = probes(teacher)
print(f'teacher: direct {t_acc:.1f}%  linear probe {t_lin:.1f}%  centroid probe {t_cen:.1f}%')

# teacher soft targets on the train subset (fixed for both students)
zt_tr = logits_of(teacher, Xtr)
p_tr = np.asarray(jax.nn.softmax(jnp.asarray(zt_tr) / TEMP, axis=1))
pn_tr = unit(p_tr)                        # normalized soft outputs -> the batch Gram target
zt_te = logits_of(teacher, Xte)

# the handoff set: 30 test garments (3 per class), fixed here so the seed-0 kernel
# student's Gram over them can be checkpointed every epoch (drives the handoff GIF).
rng_pick = np.random.RandomState(7)
PICK = np.concatenate([rng_pick.choice(np.where(yte == c)[0], 24, replace=False) for c in range(10)])
rng_pick.shuffle(PICK)
H_IDX = np.concatenate([np.where(yte[PICK] == c)[0][:3] for c in range(10)])
H_TEST = PICK[H_IDX]                       # indices into the test set, class-sorted
p_h = unit(np.asarray(jax.nn.softmax(jnp.asarray(zt_te[H_TEST]) / TEMP, axis=1)))
G_h = p_h @ p_h.T                          # the teacher's target Gram on the 30 garments

def student_handoff_gram(model):
    """The seed-0 student's normalized-embedding Gram over the 30 handoff garments."""
    e = unit(embed_of(model, Xte[H_TEST]))
    return e @ e.T

# the class kernel S(T) and its temperature sweep, from real test logits
T_GRID = [0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
def class_kernel(T):
    p = np.asarray(jax.nn.softmax(jnp.asarray(zt_te) / T, axis=1))
    S = (p[:, :, None] * p[:, None, :]).mean(0)
    d = np.sqrt(np.diag(S))
    return S, S / np.outer(d, d)          # raw and diagonal-normalized

sweep = {}
for T in T_GRID:
    S, Sn = class_kernel(T)
    off = ~np.eye(10, dtype=bool)
    w = spectrum(Sn - Sn.mean())          # spectrum of the centered normalized kernel
    sweep[T] = {'S': Sn, 'offdiag': float(Sn[off].mean()),
                'effRank': float(np.exp(-(w * np.log(w + 1e-12)).sum()))}
S4, S4n = class_kernel(TEMP)
print(f'teacher class kernel S(T): mean off-diagonal '
      + ' '.join(f'T={T}:{sweep[T]["offdiag"]:.3f}' for T in [1, 4, 16]))

H10 = np.eye(10) - np.ones((10, 10)) / 10
spec_target = spectrum(H10 @ S4n @ H10)      # the spectrum of the TRANSFERRED kernel S(T)
G_teacher = class_mean_gram(teacher)
spec_teacher = spectrum(G_teacher)
cm_teacher_dir = confusion(logits_of(teacher, Xte).argmax(1))
cm_teacher_cen = confusion(t_cen_pred)

# ══ 2. STUDENT-KD (soft targets, no hard labels) ══════════════════════════════
print('\n=== student-KD (KL to teacher soft targets, no labels) ===')
kd = student_net(0)
kd_curve = []
kd_losses = train(kd, step_kd, jnp.asarray(p_tr), EPOCHS_S, 'kd',
                  per_epoch=lambda m, ep: (kd_curve.append(acc_direct(m)) or f'test {kd_curve[-1]:.1f}%'))
kd_acc = kd_curve[-1]
kd_lin, kd_cen, kd_cen_pred = probes(kd)
G_kd = class_mean_gram(kd)
print(f'student-KD: direct {kd_acc:.1f}%  linear {kd_lin:.1f}%  centroid {kd_cen:.1f}%  CKA {cka(G_teacher, G_kd):.3f}')

# ══ 3. STUDENT-KERNEL (relational only, no labels anywhere in the loss) ══════
print('\n=== student-kernel (batch Gram matching only) ===')
kernel_students, kernel_results = {}, {}
kern_curve = {'loss': [], 'linear': [], 'centroid': [], 'cka': [], 'ckaTarget': [],
              'spectra': [], 'handoffGram': []}
for seed in [0, 1, 2]:
    st = student_net(seed)
    if seed == 0:
        kern_curve['handoffGram'].append(student_handoff_gram(st).tolist())   # epoch 0 (untrained)
        def per_epoch(m, ep):
            lin, cen, _ = probes(m)
            Gs = class_mean_gram(m)
            kern_curve['linear'].append(lin); kern_curve['centroid'].append(cen)
            kern_curve['cka'].append(cka(G_teacher, Gs))
            kern_curve['ckaTarget'].append(cka(S4n, Gs))
            kern_curve['spectra'].append(spectrum(Gs).tolist())
            kern_curve['handoffGram'].append(student_handoff_gram(m).tolist())
            return (f'linear {lin:.1f}%  centroid {cen:.1f}%  CKA {kern_curve["cka"][-1]:.3f}'
                    f'  CKA->S(T) {kern_curve["ckaTarget"][-1]:.3f}')
        kern_curve['loss'] = train(st, step_kernel, jnp.asarray(pn_tr), EPOCHS_S,
                                   f'kernel s{seed}', per_epoch=per_epoch)
    else:
        train(st, step_kernel, jnp.asarray(pn_tr), EPOCHS_S, f'kernel s{seed}')
    lin, cen, cen_pred = probes(st)
    kernel_students[seed] = st
    kernel_results[seed] = {'linear': lin, 'centroid': cen}
    print(f'student-kernel seed {seed}: linear {lin:.1f}%  centroid {cen:.1f}%')
kern = kernel_students[0]
kern_lin, kern_cen, kern_cen_pred = probes(kern)
G_kern = class_mean_gram(kern)

# ══ 4. RANDOM control ═════════════════════════════════════════════════════════
print('\n=== random control (untrained student) ===')
rnd = student_net(0)
rnd_lin, rnd_cen, rnd_cen_pred = probes(rnd)
G_rnd = class_mean_gram(rnd)
print(f'random: linear {rnd_lin:.1f}%  centroid {rnd_cen:.1f}%')

# ══ 5. LABEL baseline ═════════════════════════════════════════════════════════
print('\n=== label baseline (same student arch, CE on true labels) ===')
lab = student_net(0)
lab_curve = []
train(lab, step_ce, None, EPOCHS_S, 'label',
      per_epoch=lambda m, ep: (lab_curve.append(acc_direct(m)) or f'test {lab_curve[-1]:.1f}%'))
lab_acc = lab_curve[-1]
lab_lin, lab_cen, lab_cen_pred = probes(lab)
G_lab = class_mean_gram(lab)

# ── the table ─────────────────────────────────────────────────────────────────
print('\n================= the five runs =================')
print(f'{"run":<16}{"direct":>8}{"linear":>8}{"centroid":>9}{"CKA":>7}{"confCorr":>9}')
rows = [
    ('teacher', t_acc, t_lin, t_cen, 1.0, 1.0),
    ('student-KD', kd_acc, kd_lin, kd_cen, cka(G_teacher, G_kd), conf_corr(cm_teacher_cen, confusion(kd_cen_pred))),
    ('student-kernel', None, kern_lin, kern_cen, cka(G_teacher, G_kern), conf_corr(cm_teacher_cen, confusion(kern_cen_pred))),
    ('label', lab_acc, lab_lin, lab_cen, cka(G_teacher, G_lab), conf_corr(cm_teacher_cen, confusion(lab_cen_pred))),
    ('random', None, rnd_lin, rnd_cen, cka(G_teacher, G_rnd), conf_corr(cm_teacher_cen, confusion(rnd_cen_pred))),
]
for n, d, l, c, k, cc in rows:
    print(f'{n:<16}{("--" if d is None else f"{d:.1f}%"):>8}{l:>7.1f}%{c:>8.1f}%{k:>7.3f}{cc:>9.3f}')
ks = [kernel_results[s] for s in [0, 1, 2]]
print(f'kernel-student across seeds: linear {[round(r["linear"],1) for r in ks]}  centroid {[round(float(r["centroid"]),1) for r in ks]}')
print(f'spectra L1 to teacher latent: kernel {np.abs(spectrum(G_kern)-spec_teacher).sum():.3f}  '
      f'kd {np.abs(spectrum(G_kd)-spec_teacher).sum():.3f}  label {np.abs(spectrum(G_lab)-spec_teacher).sum():.3f}  '
      f'random {np.abs(spectrum(G_rnd)-spec_teacher).sum():.3f}')
print(f'spectra L1 to the transferred kernel S(T={TEMP:g}): kernel {np.abs(spectrum(G_kern)-spec_target).sum():.3f}  '
      f'kd {np.abs(spectrum(G_kd)-spec_target).sum():.3f}  label {np.abs(spectrum(G_lab)-spec_target).sum():.3f}  '
      f'random {np.abs(spectrum(G_rnd)-spec_target).sum():.3f}')
print(f'CKA to S(T={TEMP:g}): kernel {cka(S4n, G_kern):.3f}  kd {cka(S4n, G_kd):.3f}  '
      f'label {cka(S4n, G_lab):.3f}  random {cka(S4n, G_rnd):.3f}  teacher {cka(S4n, G_teacher):.3f}')

# ── exports for the visualizations ───────────────────────────────────────────
os.makedirs(OUT, exist_ok=True)
r3 = lambda a: [[round(float(v), 4) for v in row] for row in np.asarray(a)]
r1 = lambda a: [round(float(v), 4) for v in np.asarray(a)]

# stratified sample of test garments + their real logits (WhatALogitLeaks / TemperatureLens)
pick = PICK                               # the same fixed sample used for the handoff checkpoints
from PIL import Image
COLS = 16
tiles = Xte[pick, :, :, 0]
rows_n = (len(pick) + COLS - 1) // COLS
im = np.zeros((rows_n * 28, COLS * 28), 'uint8')
for i, t in enumerate(tiles):
    r, c = divmod(i, COLS)
    im[r * 28:r * 28 + 28, c * 28:c * 28 + 28] = (t * 255).clip(0, 255).astype('uint8')
Image.fromarray(im, 'L').save(f'{OUT}/samples.png')

# handoff set: 30 of those examples (3 per class), PCA-16 pixel inputs + the target
# Gram the teacher's softened outputs assign them (KernelHandoff trains on this live);
# h_idx / G_h were fixed above so the per-epoch student Grams line up with them.
h_idx = H_IDX
pca = PCA(16, random_state=0).fit(Xtr.reshape(len(Xtr), -1))
Hx = pca.transform(Xte[pick[h_idx]].reshape(len(h_idx), -1))
Hx = (Hx - pca.transform(Xtr.reshape(len(Xtr), -1)).mean(0)) / (pca.transform(Xtr.reshape(len(Xtr), -1)).std(0) + 1e-9)

json.dump({
    'classes': CLS, 'temp': TEMP, 'seeds': [0, 1, 2],
    'accs': {
        'teacher': {'direct': round(t_acc, 1), 'linear': round(t_lin, 1), 'centroid': round(t_cen, 1)},
        'kd': {'direct': round(kd_acc, 1), 'linear': round(kd_lin, 1), 'centroid': round(kd_cen, 1)},
        'kernel': {'linear': round(kern_lin, 1), 'centroid': round(kern_cen, 1),
                   'seeds': {s: {k: round(v, 1) for k, v in kernel_results[s].items()} for s in kernel_results}},
        'label': {'direct': round(lab_acc, 1), 'linear': round(lab_lin, 1), 'centroid': round(lab_cen, 1)},
        'random': {'linear': round(rnd_lin, 1), 'centroid': round(rnd_cen, 1)},
    },
    'spectra': {'teacher': r1(spec_teacher), 'target': r1(spec_target), 'kd': r1(spectrum(G_kd)),
                'label': r1(spectrum(G_lab)), 'random': r1(spectrum(G_rnd)), 'kernelByEpoch': kern_curve['spectra']},
    'grams': {'teacher': r3(G_teacher), 'kernel': r3(G_kern)},
    'cka': {'kernelByEpoch': r1(kern_curve['cka']), 'kd': round(cka(G_teacher, G_kd), 3),
            'label': round(cka(G_teacher, G_lab), 3), 'random': round(cka(G_teacher, G_rnd), 3),
            'targetByEpoch': r1(kern_curve['ckaTarget']),
            'toTarget': {'kernel': round(cka(S4n, G_kern), 3), 'kd': round(cka(S4n, G_kd), 3),
                         'label': round(cka(S4n, G_lab), 3), 'random': round(cka(S4n, G_rnd), 3),
                         'teacher': round(cka(S4n, G_teacher), 3)}},
    'confCorr': {'kernel': round(conf_corr(cm_teacher_cen, confusion(kern_cen_pred)), 3),
                 'kd': round(conf_corr(cm_teacher_cen, confusion(kd_cen_pred)), 3),
                 'label': round(conf_corr(cm_teacher_cen, confusion(lab_cen_pred)), 3),
                 'random': round(conf_corr(cm_teacher_cen, confusion(rnd_cen_pred)), 3)},
    'confusion': {'teacher': r3(cm_teacher_cen), 'kernel': r3(confusion(kern_cen_pred))},
    'curves': {'kernel': {'loss': r1(kern_curve['loss']), 'linear': r1(kern_curve['linear']),
                          'centroid': r1(kern_curve['centroid'])},
               'kd': r1(kd_curve), 'label': r1(lab_curve)},
    'tempSweep': {'T': T_GRID, 'offdiag': [round(sweep[T]['offdiag'], 4) for T in T_GRID],
                  'effRank': [round(sweep[T]['effRank'], 3) for T in T_GRID],
                  'S': {str(T): r3(sweep[T]['S']) for T in [1, 4, 16]}},
    'logits': {'vals': r3(zt_te[pick]), 'labels': [int(v) for v in yte[pick]], 'cols': COLS},
    'handoff': {'X': r3(Hx), 'G': r3(G_h), 'labels': [int(v) for v in yte[pick[h_idx]]],
                'sprite': [int(v) for v in h_idx],
                'studentGramByEpoch': [r3(g) for g in kern_curve['handoffGram']]},
}, open(f'{OUT}/distill.json', 'w'))
print(f'\nwrote {OUT}/distill.json + samples.png   ({time.time() - t0:.0f}s total)')
