# Separate or represent: the modality gap as a choice

Runnable experiments behind
**[Separate or Represent: The Modality Gap Is a Choice, Not a Bug](https://tahabouhsine.com/blog/modality-gap-complementary/)**.

A latent space can *separate* concepts (classification: collapse within-class
variation, keep only the boundary) or *represent* them (retrieval / reconstruction:
keep the within-class detail). It cannot do both, and the modality gap is the
readout of which one you chose.

## Run

```bash
pip install -r requirements.txt          # jax, optax, matplotlib, numpy, pillow

# experiment 1 — separate vs represent (the choice, probed)
python toy.py            # trains a SigLIP projector and a linear autoencoder
python toy_render.py     # toy_data.png, toy_collapse.png, toy_jobs.png

# experiment 2 — the modality gap as a learned dynamic (the choice, in motion)
python gap.py            # trains SigLIP, recording embeddings across steps
python gap_render.py     # gap_dynamics.gif (hero) + gap_cones.png
```

Outputs land in `../../public/modality-gap/` (served at `/blog/modality-gap/`).
Both interactives compute **live in the browser** with hand-written autodiff —
`ModalityToy.astro` (single linear head: separate vs represent) and
`ModalityGapLive.astro` (two deep ReLU towers: the gap opening). No JSON replay.
These Python scripts are the **reference JAX implementation** — they validate the
numbers and can render static figures, but the blog post no longer embeds any of
that output: it renders everything live in the browser. All plots (live or static)
use one encoding: **shape = class (○ / ✕), colour = modality**.

## Experiment 1 — `toy.py`: separate vs represent

Four distributions: **two classes** × **two modalities**. A within-class factor
`t` (position along each cluster) is drawn *independently* per modality, so a
matched pair shares only its class — `t` is modality-unique. The same linear
projector (a 2-D bottleneck) is trained two ways:

- **separate** — align the modalities with the SigLIP pairwise-sigmoid loss
- **represent** — reconstruct each modality's own input (a linear autoencoder)

Probed on modality A:

| job | metric | separate (SigLIP) | represent (autoencoder) |
|---|---|---|---|
| classify | class accuracy ↑ | **100%** | **100%** |
| represent | recover `t`, R² ↑ | 0.00 | **0.97** |
| retrieve | within-class Δt ratio ↓ | 1.03 (chance) | **0.21** |

Both spaces separate the classes; only the representing space keeps the
within-class factor. Same data, same capacity — the objective decides.

## Experiment 2 — `gap.py`: the gap as a dynamic

Two modalities of a 2-class signal, two small ReLU towers started from the **same
initialization** (so there is no gap at step 0), trained with SigLIP. The gap is
*opened* by the objective: it grows from ~0.3 to ~1.9 while matched-pair cosine
falls just to the random line (close enough to keep the ranking, far enough to
separate the cones). `gap.py` records embeddings across training and writes the
animated `gap_dynamics.gif` (the post's hero) and the static `gap_cones.png`. The
in-page gap viz (`ModalityGapLive.astro`) trains the same deep-tower setup live.

## Files

- `toy.py`, `toy_render.py` — experiment 1
- `gap.py`, `gap_render.py` — experiment 2
- `demo.py`, `radioml.py`, `render.py` — the original radio-denoising study
  (DenoMAE-style complementary modalities; its own post, kept for reference)
- `requirements.txt` — pinned deps
