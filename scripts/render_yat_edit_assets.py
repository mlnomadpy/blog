"""Pixel assets for the EDIT-A-NETWORK post's interactive visualizations. The
main post computes the Yat kernel live in the browser, so this script only ships
pixels: the 200 prototype pictures (20 per class) and a bank of test images with
labels. The browser reads the pixels out of the sprites and runs the kernel and
the edits itself.

Run: python scripts/render_yat_edit_assets.py   (writes public/yat-edit/)
"""
import warnings; warnings.filterwarnings('ignore')
import os, json, numpy as np, torchvision
from PIL import Image
from sklearn.cluster import KMeans

OUT = 'public/yat-edit'
os.makedirs(OUT, exist_ok=True)
tr = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=True, download=True)
te = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=False, download=True)
X = tr.data.numpy().reshape(-1, 784).astype('float32') / 255.0; y = tr.targets.numpy()
Xte = te.data.numpy().reshape(-1, 784).astype('float32') / 255.0; yte = te.targets.numpy()
CLS = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
B, EPS, PER, NBANK = 0.5, 0.05, 20, 300


def sprite(tiles, cols, path):
    n = len(tiles); rows = (n + cols - 1) // cols
    img = np.zeros((rows * 28, cols * 28), 'uint8')
    for i, t in enumerate(tiles):
        r, k = divmod(i, cols)
        img[r * 28:r * 28 + 28, k * 28:k * 28 + 28] = (t.reshape(28, 28) * 255).clip(0, 255).astype('uint8')
    Image.fromarray(img, 'L').save(path)


# prototype bank: PER k-means centroids per class, class-major (row = class)
protos = np.zeros((10, PER, 784), 'float32')
for c in range(10):
    protos[c] = KMeans(PER, n_init=3, random_state=0).fit(X[y == c][:2500]).cluster_centers_
sprite(protos.reshape(200, 784), PER, f'{OUT}/protos.png')        # 10 rows x 20 cols

# balanced test bank: NBANK images, equal per class, shuffled
rng = np.random.RandomState(3); pick = []
for c in range(10):
    pick += list(rng.permutation(np.where(yte == c)[0])[:NBANK // 10])
pick = np.array(pick); rng.shuffle(pick)
sprite(Xte[pick], 20, f'{OUT}/testbank.png')                      # NBANK/20 rows x 20 cols

# sanity: reproduce the headline accuracy with the same kernel the browser uses
flat = protos.reshape(200, 784); Xb = Xte[pick]
dot = Xb @ flat.T
d2 = (Xb ** 2).sum(1, keepdims=True) + (flat ** 2).sum(1) - 2 * dot
ker = (dot + B) ** 2 / (d2 + EPS)                                  # [NBANK, 200]
vote = np.repeat(np.arange(10), PER)
lab = yte[pick]
def predict(mask):                                       # argmax over enabled classes
    sc = np.full((len(Xb), 10), -1e9)
    for c in range(10):
        if mask[c]:
            sc[:, c] = ker[:, vote == c].max(1)
    return sc.argmax(1)
def recall(pred, c):
    m = lab == c
    return (pred[m] == c).mean() * 100
on = [True] * 10
pr = predict(on)
print(f"all-10 accuracy on the {NBANK}-image bank: {(pr == lab).mean()*100:.1f}%  (post quotes 79.4% on full test)")
print("per-class recall:", {CLS[c]: round(recall(pr, c)) for c in range(10)})
# forget Sandal (class 5): its recall and the others' mean recall, before vs after
o5 = on.copy(); o5[5] = False; pr5 = predict(o5)
oth = [c for c in range(10) if c != 5]
print(f"forget Sandal: recall {recall(pr,5):.0f}% -> {recall(pr5,5):.0f}% ; "
      f"other 9 mean recall {np.mean([recall(pr,c) for c in oth]):.1f}% -> {np.mean([recall(pr5,c) for c in oth]):.1f}%")

json.dump({
    'classes': CLS, 'protoCols': PER, 'bankCols': 20, 'nbank': int(len(pick)),
    'b': B, 'eps': EPS, 'per': PER,
    'bankLabels': [int(v) for v in yte[pick]],
}, open(f'{OUT}/edit.json', 'w'))
print(f"wrote {OUT}/protos.png, testbank.png, edit.json  ({len(pick)} bank images)")
