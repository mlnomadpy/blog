# Contrastive-learning GIF generator

The runnable companion to the blog post
**[Organizing Randomness: Contrastive Learning in JAX](https://tahabouhsine.com/blog/organizing-randomness-jax/)**.

Six contrastive losses, implemented as pure `jax.numpy` functions, each
optimizing 2D embeddings from a random initialization. The optimization
trajectory is rendered to an animated GIF styled to match the blog.

## Run

```bash
pip install -r requirements.txt        # or: pixi add jax optax matplotlib imageio
python generate.py --loss all --grid
```

GIFs are written to `../../public/jax-contrastive/`, which Astro serves at
`/blog/jax-contrastive/<name>.gif`.

| flag | default | meaning |
|------|---------|---------|
| `--loss` | `all` | `pair`, `triplet`, `infonce`, `supcon`, `siglip`, `orthog`, or `all` |
| `--dataset` | per-loss | `random`, `random-4`, or `moons` |
| `--seed` | `7` | RNG seed (data + samplers) |
| `--steps` | per-loss | optimization steps |
| `--stride` | `4` | capture a frame every k steps |
| `--fps` | `20` | GIF frame rate |
| `--size` | `480` | square pixel size |
| `--grid` | off | also render the 2×3 race grid |

## Files

- `data.py` — toy datasets + nearest-centroid accuracy
- `losses.py` — the six losses + the `LOSSES` registry
- `train.py` — one jitted `optax.sgd` + `jax.value_and_grad` step, shared by all
- `render.py` — matplotlib → imageio, styled to the blog palette
- `generate.py` — argparse CLI

## Reproducibility

The optimization is deterministic given the seed (fixed JAX key, float32).
Rendered pixels may differ slightly across matplotlib / freetype versions, so
the guarantee is on the *trajectory*, not the bytes. Versions are pinned in
`requirements.txt`.
