"""The hand-built, training-free image classifier, in JAX + Flax NNX.

Companion to "You Don't Even Have to Train the Features". The feature extractor is
pure JAX (Sobel gradients via lax.conv, soft orientation binning, patch pooling);
the classifier is a Flax NNX module holding k-means prototypes and voting with the
Yat kernel. Nothing is trained. Reproduces the 83.3% from handbuilt_vision.py.

Run: python scripts/jax_handbuilt.py
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp, torchvision
from flax import nnx
from sklearn.cluster import KMeans

NB, GRID, NCH = 6, 7, 7
CENTERS = jnp.linspace(0, jnp.pi, NB, endpoint=False)
SX = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], jnp.float32)
SY = SX.T


def _conv(imgs, k):
    """Edge-replicated 3x3 correlation, [N,28,28] -> [N,28,28] (matches scipy mode='nearest')."""
    x = jnp.pad(imgs, ((0, 0), (1, 1), (1, 1)), mode='edge')[:, None]   # [N,1,30,30]
    return jax.lax.conv_general_dilated(x, k[None, None], (1, 1), 'VALID',
                                        dimension_numbers=('NCHW', 'OIHW', 'NCHW'))[:, 0]


@jax.jit
def features(imgs):
    """[N,28,28] in [0,1] -> [N,343] hand-built features. No parameters."""
    gx, gy = _conv(imgs, SX), _conv(imgs, SY)
    mag = jnp.sqrt(gx ** 2 + gy ** 2) + 1e-6
    ang = jnp.mod(jnp.arctan2(gy, gx), jnp.pi)                          # undirected edge angle
    # soft-assign each pixel to the NB orientation bins -> [N,NB,28,28]
    d = jnp.abs(jnp.mod(ang[:, None] - CENTERS[None, :, None, None] + jnp.pi / 2, jnp.pi) - jnp.pi / 2)
    edges = jnp.clip(1 - d / (jnp.pi / NB), 0, 1) * mag[:, None]
    corner = (jnp.abs(gx) * jnp.abs(gy))[:, None]
    chans = jnp.concatenate([edges, corner], 1)                        # [N,NCH,28,28]
    # mean-pool over a GRID x GRID patch grid
    n = imgs.shape[0]; ps = 28 // GRID
    pooled = chans[:, :, :GRID * ps, :GRID * ps].reshape(n, NCH, GRID, ps, GRID, ps).mean((3, 5))
    V = pooled.reshape(n, -1)
    return V / (jnp.linalg.norm(V, axis=1, keepdims=True) + 1e-6)       # per-image L2 norm


class YatHead(nnx.Module):
    """A constructed head: prototype rows W (k-means centroids), one-hot votes A,
    and a nearest-prototype vote under the Yat kernel. No trainable parameters."""
    def __init__(self, W, vote, b=0.5, eps=1.0):
        self.W = nnx.Variable(jnp.asarray(W))                          # [K, d] prototypes
        self.A = nnx.Variable(jax.nn.one_hot(jnp.asarray(vote), 10))   # [K, 10] class votes
        self.b, self.eps = b, eps

    def __call__(self, z):                                            # z: [N, d] features
        dot = z @ self.W.value.T
        d2 = (z ** 2).sum(1, keepdims=True) + (self.W.value ** 2).sum(1) - 2 * dot
        ker = (dot + self.b) ** 2 / (d2 + self.eps)                    # [N, K] Yat kernel
        # per-class max kernel, then argmax (nearest-prototype vote)
        scores = jnp.where(self.A.value.T[None].astype(bool), ker[:, None, :], -jnp.inf).max(-1)
        return scores.argmax(-1)


if __name__ == '__main__':
    tr = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=True, download=True)
    te = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=False, download=True)
    Xtr = jnp.asarray(tr.data.numpy()[:20000], jnp.float32) / 255.0; ytr = tr.targets.numpy()[:20000]
    Xte = jnp.asarray(te.data.numpy(), jnp.float32) / 255.0; yte = te.targets.numpy()

    Ftr, Fte = np.asarray(features(Xtr)), np.asarray(features(Xte))
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd                        # z-score into the head's space

    PER = 20; W, vote = [], []                                         # build the head by hand: k-means per class
    for c in range(10):
        W += list(KMeans(PER, n_init=2, random_state=0).fit(Ftr[ytr == c][:2500]).cluster_centers_); vote += [c] * PER
    W = np.array(W, 'float32'); vote = np.array(vote)
    eps = float(np.median(((Fte ** 2).sum(1, keepdims=True) + (W ** 2).sum(1) - 2 * Fte @ W.T)) * 0.1)

    head = YatHead(W, vote, b=0.5, eps=eps)
    pred = np.asarray(head(jnp.asarray(Fte)))
    print(f"hand-built, no training anywhere (JAX): {100 * (pred == yte).mean():.1f}%")
