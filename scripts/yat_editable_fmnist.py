"""Editing a Yat-kernel classifier by hand, with no training.

Backs the post "Your Network Is a List of Pictures. You Can Edit It." Every
number quoted there is printed here. A Yat unit is a kernel against a prototype
that lives in input space, so a classifier is just a labelled bank of pictures
plus a one-hot readout. That makes the network editable: you add a class by
appending its prototypes, and you unlearn a class by deleting them. No gradient
steps anywhere.

Run: python scripts/yat_editable_fmnist.py
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, torchvision
from sklearn.cluster import KMeans

tr = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=True, download=True)
te = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=False, download=True)
X = tr.data.numpy().reshape(-1, 784).astype('float32') / 255.0; y = tr.targets.numpy()
Xte = te.data.numpy().reshape(-1, 784).astype('float32') / 255.0; yte = te.targets.numpy()
CLS = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
B, EPS, PER = 0.5, 0.05, 20                            # kernel bias, floor, prototypes per class


def build(classes, per=PER):                           # place `per` k-means centroids per class
    W, vote = [], []
    for c in classes:
        cen = KMeans(per, n_init=3, random_state=0).fit(X[y == c][:2500]).cluster_centers_
        W += list(cen); vote += [c] * per
    return np.array(W, 'float32'), np.array(vote)


def kernel(W, Xq):                                     # [n, K] Yat activations
    dot = Xq @ W.T
    d2 = (Xq ** 2).sum(1, keepdims=True) + (W ** 2).sum(1) - 2 * dot
    return (dot + B) ** 2 / (d2 + EPS)


def predict(W, vote, Xq, mode='max'):                  # nearest-prototype ('max') or sum-vote
    K = kernel(W, Xq); score = np.full((len(Xq), 10), -1e9)
    for c in np.unique(vote):
        col = K[:, vote == c]
        score[:, c] = col.max(1) if mode == 'max' else col.sum(1)
    return score.argmax(1)


def acc_on(W, vote, Xq, yq, subset=None, mode='max'):
    m = np.isin(yq, subset) if subset is not None else np.ones(len(yq), bool)
    return float((predict(W, vote, Xq[m], mode) == yq[m]).mean()) * 100


def recall(W, vote, cls, mode='max'):
    m = yte == cls
    return float((predict(W, vote, Xte[m], mode) == cls).mean()) * 100


print("=== A) CLASS-INCREMENTAL: add 2 unseen classes by placing prototypes, no retraining ===")
W8, v8 = build(range(8))
old_before = acc_on(W8, v8, Xte, yte, subset=list(range(8)))
print(f"  base model on classes 0-7 (8x{PER} prototypes): {old_before:.1f}% on their test images")
Wadd, vadd = build([8, 9])
W10 = np.vstack([W8, Wadd]); v10 = np.concatenate([v8, vadd])
old_after = acc_on(W10, v10, Xte, yte, subset=list(range(8)))
print(f"  after adding {CLS[8]} and {CLS[9]} (placed {2 * PER} pictures, ZERO gradient steps):")
print(f"    old classes 0-7:  {old_before:.1f}% -> {old_after:.1f}%  (drop {old_before - old_after:+.1f})")
print(f"    new classes:      {CLS[8]} {recall(W10, v10, 8):.0f}%, {CLS[9]} {recall(W10, v10, 9):.0f}%")
print(f"    overall all 10:   {acc_on(W10, v10, Xte, yte):.1f}%")
Wg, vg = build(range(10))
print(f"  reference, built on all 10 from the start: {acc_on(Wg, vg, Xte, yte):.1f}%")

print("\n=== B) UNLEARNING: forget a class instantly by deleting its prototypes ===")
t = 5
others = [c for c in range(10) if c != t]
r_before, o_before = recall(W10, v10, t), acc_on(W10, v10, Xte, yte, subset=others)
keep = v10 != t
Wu, vu = W10[keep], v10[keep]
r_after, o_after = recall(Wu, vu, t), acc_on(Wu, vu, Xte, yte, subset=others)
print(f"  delete the {PER} '{CLS[t]}' prototypes and its readout rows:")
print(f"    '{CLS[t]}' recall:  {r_before:.0f}% -> {r_after:.0f}%   (forgotten)")
print(f"    other 9 classes:  {o_before:.1f}% -> {o_after:.1f}%  (untouched)")
