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
initialization** (so there is no gap at step 0), trained with SigLIP (learnable
logit scale, init 2.0; Adam, 1400 steps). The gap is *opened* by the objective:
it grows from 0.27 to ~1.9 within the first hundred steps, while matched-pair
cosine falls through zero (around step 12) and tracks the shuffled-pair baseline
down into negative territory, ending at −0.78 with random at −0.88, negative in
absolute terms but still ranked above random. `gap.py` records embeddings across
training and writes the animated `gap_dynamics.gif` (the post's hero) and the
static `gap_cones.png`. The in-page gap viz (`ModalityGapLive.astro`) trains the
same deep-tower setup live.

`python gap.py --sweep` trains the same setup with a **fixed** (untrained) logit
scale at 1/τ ∈ {0.5, 1, 2, 4, 8}. The gap opens at every sharpness; soft
temperatures open it widest (fully antipodal, gap → 2.0) but lose the ranking
entirely (matched = random = −1.00), while sharpness preserves the
matched-above-random margin (at 1/τ = 8: matched −0.49 vs random −0.66,
gap 1.65). Temperature does not open the gap; it buys back the ranking.

## Files

- `toy.py`, `toy_render.py` — experiment 1
- `gap.py`, `gap_render.py` — experiment 2
- `demo.py`, `radioml.py`, `render.py` — the original radio-denoising study
  (DenoMAE-style complementary modalities; its own post, kept for reference)
- `requirements.txt` — pinned deps
