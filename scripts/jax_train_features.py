"""Construct the head on learned features, in JAX + Flax NNX.

Companion to "You Only Have to Train the Features". Trains a small conv backbone,
then on the FROZEN features builds a constructed Yat head (k-means prototypes,
nearest-prototype vote, no gradient steps) and compares it to the trained head.
Also shows the surprise: the constructed head on a RANDOM backbone.

Run: python scripts/jax_train_features.py
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp, optax, torchvision
from flax import nnx
from sklearn.cluster import KMeans


class Backbone(nnx.Module):
    """Two conv-pool blocks down to a 64-d feature. This is the part that is trained."""
    def __init__(self, rngs):
        self.c1 = nnx.Conv(1, 16, (3, 3), padding='SAME', rngs=rngs)
        self.c2 = nnx.Conv(16, 32, (3, 3), padding='SAME', rngs=rngs)
        self.lin = nnx.Linear(32 * 7 * 7, 64, rngs=rngs)

    def __call__(self, x):                                            # x: [N,28,28,1]
        x = nnx.max_pool(nnx.relu(self.c1(x)), (2, 2), (2, 2))
        x = nnx.max_pool(nnx.relu(self.c2(x)), (2, 2), (2, 2))
        return nnx.relu(self.lin(x.reshape(x.shape[0], -1)))           # [N,64]


class Net(nnx.Module):
    def __init__(self, rngs):
        self.feat = Backbone(rngs); self.head = nnx.Linear(64, 10, rngs=rngs)
    def __call__(self, x): return self.head(self.feat(x))


def build_head(Ftr, ytr, Fte, per=20):
    """A constructed Yat head on frozen features: k-means prototypes + nearest vote."""
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ztr, Zte = (Ftr - mu) / sd, (Fte - mu) / sd
    W, vote = [], []
    for c in range(10):
        W += list(KMeans(per, n_init=2, random_state=0).fit(Ztr[ytr == c][:2500]).cluster_centers_); vote += [c] * per
    W = np.array(W, 'float32'); vote = np.array(vote)
    dot = Zte @ W.T
    d2 = (Zte ** 2).sum(1, keepdims=True) + (W ** 2).sum(1) - 2 * dot
    ker = (dot + 0.5) ** 2 / (d2 + np.median(d2) * 0.1)
    score = np.full((len(Zte), 10), -1e9)
    for c in range(10): score[:, c] = ker[:, vote == c].max(1)
    return score.argmax(1)


if __name__ == '__main__':
    tr = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=True, download=True)
    te = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=False, download=True)
    Xtr = jnp.asarray(tr.data.numpy()[:20000], jnp.float32)[..., None] / 255.0; ytr = tr.targets.numpy()[:20000]
    Xte = jnp.asarray(te.data.numpy(), jnp.float32)[..., None] / 255.0; yte = te.targets.numpy()
    feats = lambda net, X: np.concatenate([np.asarray(net.feat(X[i:i + 2000])) for i in range(0, len(X), 2000)])

    # the surprise first: a RANDOM (untrained) backbone, constructed head
    rnd = Net(nnx.Rngs(0))
    con_rand = 100 * (build_head(feats(rnd, Xtr), ytr, feats(rnd, Xte)) == yte).mean()
    print(f"random backbone, constructed head: {con_rand:.1f}%  (its trained head is at chance)")

    # train the backbone
    net = Net(nnx.Rngs(0)); opt = nnx.Optimizer(net, optax.adam(1e-3), wrt=nnx.Param)

    @nnx.jit
    def step(net, opt, xb, yb):
        def loss(net): return optax.softmax_cross_entropy_with_integer_labels(net(xb), yb).mean()
        l, g = nnx.value_and_grad(loss)(net); opt.update(net, g); return l

    for ep in range(6):
        perm = np.random.RandomState(ep).permutation(len(Xtr))
        for i in range(0, len(Xtr) - 256, 256):
            j = perm[i:i + 256]; step(net, opt, Xtr[j], jnp.asarray(ytr[j]))

    trained = 100 * np.mean([np.asarray(net(Xte[i:i + 2000]).argmax(1)) == yte[i:i + 2000] for i in range(0, len(Xte), 2000)])
    con = 100 * (build_head(feats(net, Xtr), ytr, feats(net, Xte)) == yte).mean()
    print(f"trained backbone, trained head:     {trained:.1f}%")
    print(f"trained backbone, constructed head: {con:.1f}%  (no head training)")
