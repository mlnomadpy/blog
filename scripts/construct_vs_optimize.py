"""The construction-vs-optimization boundary, on Fashion-MNIST.

Trains a small CNN backbone and, at every epoch, freezes it and CONSTRUCTS a Yat
head on its features (k-means prototypes per class, one-hot readout, nearest-
prototype vote) with zero head training, comparing to the backbone's own trained
head. Also checks the edits (teach/forget) in feature space, and dumps
public/train-features/fold.json for the interactive engine visualization in the
explainer (per-epoch features, a 2-D layout, the prototypes, and the accuracies).

Run: python scripts/construct_vs_optimize.py
"""
import warnings; warnings.filterwarnings('ignore')
import os, json, numpy as np, torch, torch.nn as nn, torchvision
from sklearn.cluster import KMeans

dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLS = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
B, PER = 0.5, 20

tr = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=True, download=True)
te = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=False, download=True)
Xtr = (tr.data.float() / 255.0)[:, None]; ytr = tr.targets; ytr_np = ytr.numpy()
Xte = (te.data.float() / 255.0)[:, None]; yte = te.targets; yte_np = yte.numpy()


class Net(nn.Module):
    def __init__(s, d=64):
        super().__init__()
        s.feat = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, d), nn.ReLU())
        s.head = nn.Linear(d, 10)
    def forward(s, x): return s.head(s.feat(x))


def features(net, X, bs=2000):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs): out.append(net.feat(X[i:i+bs].to(dev)).cpu().numpy())
    return np.concatenate(out)


def build_head(Ftr, ytr_np, enabled, per=PER):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    W, vote = [], []
    for c in range(10):
        if not enabled[c]: continue
        Fc = ((Ftr[ytr_np == c] - mu) / sd)[:2500]
        W += list(KMeans(per, n_init=2, random_state=0).fit(Fc).cluster_centers_); vote += [c] * per
    return np.array(W, 'float32'), np.array(vote), mu, sd


def classify(Fte, W, vote, mu, sd):
    Z = (Fte - mu) / sd
    dot = Z @ W.T
    d2 = (Z ** 2).sum(1, keepdims=True) + (W ** 2).sum(1) - 2 * dot
    ker = (dot + B) ** 2 / (d2 + np.median(d2) * 0.1)
    score = np.full((len(Z), 10), -1e9)
    for c in np.unique(vote): score[:, c] = ker[:, vote == c].max(1)
    return score.argmax(1)


def eval_trained(net):
    net.eval(); ok = 0
    with torch.no_grad():
        for i in range(0, len(Xte), 2000):
            ok += (net(Xte[i:i+2000].to(dev)).argmax(1).cpu() == yte[i:i+2000]).sum().item()
    return 100 * ok / len(Xte)


def pca2(F):                                                # 2-D layout for display
    Fc = F - F.mean(0); U, S, Vt = np.linalg.svd(Fc, full_matrices=False)
    Y = Fc @ Vt[:2].T
    if Y[np.argmax(np.abs(Y[:, 0])), 0] < 0: Y[:, 0] *= -1   # deterministic sign
    if Y[np.argmax(np.abs(Y[:, 1])), 1] < 0: Y[:, 1] *= -1
    return Y / (np.abs(Y).max() + 1e-9)                      # to roughly [-1,1]


torch.manual_seed(0)
net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3); lossf = nn.CrossEntropyLoss()
sub = np.random.RandomState(0).permutation(len(Xtr))[:20000]; ytr_sub = ytr_np[sub]
viz = np.random.RandomState(1).permutation(len(Xte))[:600]   # points shown in the viz
EPOCHS, full = 6, np.ones(10, bool)
print(f"device {dev}\n{'epoch':>5} {'trained head':>13} {'constructed head':>17}")
curve, frames = [], []

for ep in range(EPOCHS + 1):
    if ep > 0:
        net.train(); perm = np.random.RandomState(ep).permutation(len(sub))
        for i in range(0, len(sub), 256):
            idx = sub[perm[i:i+256]]
            opt.zero_grad(); lossf(net(Xtr[idx].to(dev)), ytr[idx].to(dev)).backward(); opt.step()
    Ftr = features(net, Xtr[sub]); Fte = features(net, Xte)
    W, vote, mu, sd = build_head(Ftr, ytr_sub, full, per=PER)   # 20/class, matches the text
    cons = 100 * (classify(Fte, W, vote, mu, sd) == yte_np).mean()
    trained = eval_trained(net); curve.append((ep, trained, cons))
    print(f"{ep:>5} {trained:>12.1f}% {cons:>16.1f}%")
    if ep in (0, 1, 2, 4, 6):                                  # snapshot for the interactive viz
        Zv = (Fte[viz] - mu) / sd
        with torch.no_grad(): tp = net(Xte[viz].to(dev)).argmax(1).cpu().numpy()
        frames.append({'ep': ep, 'trained': round(trained, 1), 'constructed': round(cons, 1),
                       'embed': [[round(float(a), 3) for a in r] for r in pca2(Zv)],
                       'feat': [[round(float(a), 2) for a in r] for r in Zv],
                       'protos': [[round(float(a), 2) for a in r] for r in W],
                       'protoEmbed': [[round(float(a), 3) for a in r] for r in pca2(np.vstack([Zv, W]))[len(Zv):]],
                       'vote': [int(v) for v in vote], 'trainedPred': [int(v) for v in tp]})

print("\n(raw-pixel constructed head was ~79% in the editable-network post)")
W, vote, mu, sd = build_head(Ftr, ytr_sub, full)
pr_full = classify(Fte, W, vote, mu, sd)
en = full.copy(); en[5] = False
Wd, vd, md, sdd = build_head(Ftr, ytr_sub, en); pr_del = classify(Fte, Wd, vd, md, sdd)
rec = lambda pr, c: 100 * (pr[yte_np == c] == c).mean()
oth = [c for c in range(10) if c != 5]
print("=== edits in feature space ===")
print(f"forget Sandal: {rec(pr_full,5):.0f}% -> 0% ; others {np.mean([rec(pr_full,c) for c in oth]):.1f}% -> {np.mean([rec(pr_del,c) for c in oth]):.1f}%")
en8 = np.array([c < 8 for c in range(10)]); W8, v8, m8s, s8 = build_head(Ftr, ytr_sub, en8); pr8 = classify(Fte, W8, v8, m8s, s8)
m8 = np.isin(yte_np, range(8))
print(f"teach: 8-class head {100*(pr8[m8]==yte_np[m8]).mean():.1f}%; add Bag,Boot -> {rec(pr_full,8):.0f}%, {rec(pr_full,9):.0f}%, all-10 {100*(pr_full==yte_np).mean():.1f}%")

# ── pixel-space layout + the most pixel-confusable pair, for the "what is a feature" panel ──
Xte_flat = Xte.numpy().reshape(len(Xte), 784); Xtr_flat = Xtr.numpy().reshape(len(Xtr), 784)
means = np.stack([Xtr_flat[sub][ytr_sub == c].mean(0) for c in range(10)])          # class means in pixel space
pix_d2 = np.stack([((Xte_flat - means[c]) ** 2).sum(1) for c in range(10)], 1)       # test -> nearest pixel mean
pix_pred = pix_d2.argmin(1)
cm = np.zeros((10, 10), int)
for t, p in zip(yte_np, pix_pred): cm[t, p] += 1
sym = cm + cm.T; np.fill_diagonal(sym, 0)                                            # symmetric off-diagonal confusion
ci, cj = np.unravel_index(sym.argmax(), sym.shape); confuse = sorted([int(ci), int(cj)])
pixel_embed = pca2(Xte_flat[viz])                                                    # stage-independent 2-D pixel layout
print(f"\nmost pixel-confusable pair: {CLS[confuse[0]]} / {CLS[confuse[1]]} ({sym[ci, cj]} swaps)")

os.makedirs('public/train-features', exist_ok=True)
json.dump({'classes': CLS, 'dim': 64, 'labels': [int(v) for v in yte_np[viz]], 'frames': frames,
           'rawPixel': 79.0, 'confusePair': confuse,
           'pixelEmbed': [[round(float(a), 3) for a in r] for r in pixel_embed]},
          open('public/train-features/fold.json', 'w'))
print(f"\nwrote public/train-features/fold.json ({len(frames)} snapshots, {len(viz)} points)")

# ── sprites for the pixel-vs-exemplar panel ──
from PIL import Image
PE = 8
def sprite(tiles, cols, path):
    n = len(tiles); rows = (n + cols - 1) // cols; im = np.zeros((rows * 28, cols * 28), 'uint8')
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols); im[r*28:r*28+28, c*28:c*28+28] = (t * 255).clip(0, 255).astype('uint8')
    Image.fromarray(im, 'L').save(path)

Xnp = Xtr.numpy()[:, 0]
pix = []
for c in range(10):
    cen = KMeans(PE, n_init=3, random_state=0).fit(Xnp[ytr_np == c][:2500].reshape(-1, 784)).cluster_centers_
    pix += [cc.reshape(28, 28) for cc in cen]                  # blurry pixel-space averages
sprite(pix, PE, 'public/train-features/pixel-protos.png')

Fn = (Ftr - Ftr.mean(0)) / (Ftr.std(0) + 1e-6); imgs = Xnp[sub]
exe = []
for c in range(10):
    Fc, Ic = Fn[ytr_sub == c], imgs[ytr_sub == c]
    cen = KMeans(PE, n_init=3, random_state=0).fit(Fc[:2500]).cluster_centers_
    for j in range(PE):
        exe.append(Ic[((Fc - cen[j]) ** 2).sum(1).argmin()])   # nearest real image to each feature prototype
sprite(exe, PE, 'public/train-features/feat-exemplars.png')
print("wrote pixel-protos.png, feat-exemplars.png")
