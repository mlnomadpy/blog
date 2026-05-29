"""Generate the contrastive-learning GIFs for the blog post.

Examples
--------
    # every loss on its default dataset, plus the 2x3 race grid
    python generate.py --loss all --grid

    # a single loss
    python generate.py --loss infonce

Outputs land in ``public/jax-contrastive/`` so Astro serves them at
``/blog/jax-contrastive/<name>.gif``.
"""

from __future__ import annotations

import argparse
import os

from data import DATASETS, get_dataset
from losses import LOSSES
from render import render_gif, render_grid_gif
from train import run

HERE = os.path.dirname(__file__)
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "..", "public", "jax-contrastive"))


def generate_one(name, *, dataset=None, seed=7, steps=None, stride=4,
                 fps=20, size=480, out=DEFAULT_OUT):
    cfg = LOSSES[name]
    ds = dataset or cfg.dataset
    if steps:
        cfg.steps = steps
    use_seed = cfg.seed if cfg.seed is not None else seed
    z0, labels = get_dataset(ds, cfg.n, use_seed)
    frames = run(cfg, z0, labels, use_seed, stride)
    path = os.path.join(out, f"{name}.gif")
    size_bytes = render_gif(frames, labels, path, title=cfg.title,
                            on_sphere=cfg.on_sphere, fps=fps, size_px=size)
    print(f"  {name:8s} {ds:9s} {len(frames):3d} frames  ->  "
          f"{path}  ({size_bytes / 1024:.0f} KB)")
    return frames, labels, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loss", default="all",
                    choices=["all", *LOSSES.keys()])
    ap.add_argument("--dataset", default=None, choices=list(DATASETS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--grid", action="store_true",
                    help="also render the 2x3 race grid (all six losses)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    names = list(LOSSES) if args.loss == "all" else [args.loss]

    print(f"Rendering to {args.out}")
    captured = {}
    for name in names:
        frames, labels, cfg = generate_one(
            name, dataset=args.dataset, seed=args.seed, steps=args.steps,
            stride=args.stride, fps=args.fps, size=args.size, out=args.out)
        captured[name] = (frames, labels, cfg)

    if args.grid and len(captured) == len(LOSSES):
        order = list(LOSSES)
        per = [captured[n][0] for n in order]
        labs = [captured[n][1] for n in order]
        titles = [LOSSES[n].title.split(" (")[0] for n in order]
        sphere = [LOSSES[n].on_sphere for n in order]
        path = os.path.join(args.out, "grid-six.gif")
        size_bytes = render_grid_gif(per, labs, titles, path,
                                     on_sphere_by_loss=sphere, fps=args.fps)
        print(f"  {'grid':8s} {'six':9s}      ->  {path}  "
              f"({size_bytes / 1024:.0f} KB)")

    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out) if f.endswith(".gif"))
    print(f"Total GIF size: {total / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
