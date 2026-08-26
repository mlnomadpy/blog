"""Build staged-reading GIFs from verified companion figures.

These are not decorative loops. Each animation reveals a multi-panel computation or
measured comparison in the order described by its companion caption. Source PNGs
are outputs of the experiment-specific renderers; this script changes presentation,
not values.
"""
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "public"

# Rectangles are normalized (left, top, right, bottom) and follow the reading order.
JOBS = {
    "depth-junction-assembly.png": ("depth-junction-build.gif", [(0, 0, .23, 1), (.23, 0, .47, 1), (.47, 0, .69, 1), (.69, 0, 1, 1)]),
    "depth-ladder.png": ("depth-ladder-build.gif", [(0, 0, .45, 1), (.45, 0, .72, 1), (.72, 0, 1, 1)]),
    "depth-layer2-alone.png": ("depth-layer2-build.gif", [(0, 0, .42, 1), (.42, 0, .68, 1), (.68, 0, 1, 1)]),
    "calib-channels.png": ("calib-channels-read.gif", [(0, 0, .5, 1), (.5, 0, 1, 1)]),
    "calib-seeds.png": ("calib-seeds-read.gif", [(0, 0, .52, 1), (.52, 0, 1, 1)]),
    "handbuilt-coarsegrain.png": ("handbuilt-coarsegrain-build.gif", [(0, 0, 1, .38), (0, .38, 1, .67), (0, .67, 1, 1)]),
    "handbuilt-yat-well.png": ("handbuilt-yat-well-read.gif", [(0, 0, .5, .52), (.5, 0, 1, .52), (0, .52, .5, 1), (.5, .52, 1, 1)]),
    "spectral-codebook.png": ("spectral-codebook-build.gif", [(0, 0, .42, 1), (.42, 0, .72, 1), (.72, 0, 1, 1)]),
    "mds-reconstruction.png": ("mds-reconstruction-build.gif", [(0, 0, .34, 1), (.34, 0, .67, 1), (.67, 0, 1, 1)]),
    "velocity-ledger-steps.png": ("velocity-ledger-steps-build.gif", [(0, 0, .34, 1), (.34, 0, .67, 1), (.67, 0, 1, 1)]),
    "velocity-ledger-pathbars.png": ("velocity-ledger-pathbars-read.gif", [(0, 0, 1, .42), (0, .42, 1, .72), (0, .72, 1, 1)]),
    "gravity-attention-bookkeeping.png": ("gravity-attention-bookkeeping-build.gif", [(0, 0, .25, 1), (.25, 0, .5, 1), (.5, 0, .75, 1), (.75, 0, 1, 1)]),
    "yat-ffn-attribution.png": ("yat-ffn-attribution-build.gif", [(0, 0, 1, .34), (0, .34, 1, .68), (0, .68, 1, 1)]),
    "trial-calibration.png": ("trial-calibration-read.gif", [(0, 0, .5, 1), (.5, 0, 1, 1)]),
    "train-head-swap.png": ("train-head-swap-read.gif", [(0, 0, .5, 1), (.5, 0, 1, 1)]),
    "weight-be-bill.png": ("weight-be-bill-build.gif", [(0, 0, .38, 1), (.38, 0, .7, 1), (.7, 0, 1, 1)]),
    "weight-lives-rings.png": ("weight-lives-rings-build.gif", [(0, 0, .34, 1), (.34, 0, .67, 1), (.67, 0, 1, 1)]),
}


def ease(t):
    return t * t * (3 - 2 * t)


def px(rect, width, height):
    l, t, r, b = rect
    return tuple(round(v) for v in (l * width, t * height, r * width, b * height))


def render(source, target, stages):
    base = Image.open(PUB / source).convert("RGB")
    if base.width > 880:
        scale = 880 / base.width
        base = base.resize((880, round(base.height * scale)), Image.Resampling.LANCZOS)
    dark = Image.blend(base, Image.new("RGB", base.size, "#080807"), .78)
    frames = []
    done = []
    for rect in stages:
        l, t, r, b = px(rect, *base.size)
        horizontal = (r - l) >= (b - t)
        for step in range(9):
            q = ease(step / 8)
            mask = Image.new("L", base.size, 0)
            md = ImageDraw.Draw(mask)
            for prior in done:
                md.rectangle(px(prior, *base.size), fill=255)
            active = (l, t, round(l + (r - l) * q), b) if horizontal else (l, t, r, round(t + (b - t) * q))
            md.rectangle(active, fill=255)
            frame = Image.composite(base, dark, mask)
            if step:
                fd = ImageDraw.Draw(frame)
                fd.rectangle((l, t, r - 1, b - 1), outline="#36d6c4", width=max(2, base.width // 360))
            frames.append(frame)
        done.append(rect)
        frames.extend([frames[-1]] * 3)
    frames.extend([base] * 12)
    out = PUB / target
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=85, loop=0,
                   optimize=True, disposal=2)
    base.save(PUB / target.replace(".gif", "-preview.png"))
    print(f"wrote {out.name}: {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    for source, (target, stages) in JOBS.items():
        render(source, target, stages)
