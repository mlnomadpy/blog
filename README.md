# Records of the !mmortal Data Scientist

A blog about machine learning, geometry, and what neural networks are actually
doing under the hood. Most posts share a thread: take a familiar piece of a
modern model — attention, MLPs, contrastive losses, activations — and ask
*what mathematical object is it, really?* The answers usually point at kernels,
RKHS geometry, or the shape of the loss landscape.

Live at <https://tahabouhsine.com/blog/>.

## Posts

### [What an MLP Knows, When It's a Kernel](https://tahabouhsine.com/blog/what-an-mlp-knows/)

The MLP block is illegible because its primitive — affine + pointwise
nonlinearity — does not carry a kernel. Replace it with a Yat unit (a quadratic
similarity divided by an induced distance) and the four objects that make
attention legible follow for free: pairwise scores, normalised contributions,
named geometry, prototype units. Ships with animated stage-by-stage diagrams of
the transformer / attention / MLP blocks and a kernel playground.

### [Not All Infinities Are Equal](https://tahabouhsine.com/blog/not-all-infinities-are-equal/)

Cross-entropy has a singularity at $p \to 0$, not at $p \to 1$. That asymmetry
is not a quirk of the loss surface — it's the reason language models
hallucinate, the reason CLIP-style contrastive training needs enormous batches,
and the reason the modality gap is geometrically inevitable. The post separates
what's a theorem from what's a hypothesis, and visualises the growth rates,
asymmetry, and vector-vs-probability gradients side by side.

### [Opposite Is Not Different](https://tahabouhsine.com/blog/opposite-is-not-different/)

The cosine-similarity scale has *three* landmarks, not two. Maximum difference
between two unit vectors is orthogonality, not opposition. The most influential
contrastive losses spent years optimizing for the wrong target — pushing
negatives toward $-1$ when they should have been pushing toward $0$. Includes
a simplex-packing viz showing why $k$ classes want $k-1$ dimensions, not $k$.

### [Activations Are Bad for Geometry](https://tahabouhsine.com/blog/activations-are-bad-for-geometry/)

Pointwise activations factor into the layer's Jacobian as a diagonal modulation.
The same modulation that buys selectivity destroys geometric structure on the
data manifold. With a 3D Jacobian-warp viz showing exactly how ReLU and friends
crumple the input space.

### [Attention is Explainable Because it is a Kernel](https://tahabouhsine.com/blog/attention-is-a-kernel/)

Self-attention is a Nadaraya–Watson smoother. The score $QK^\top / \sqrt{d}$ is
a kernel, the softmax row is a normalised contribution, and the output is a
kernel-weighted average of values. RKHS framing explains why attention is
inherently more interpretable than the MLP that follows it.

### [Morocco and AI Illiteracy — Part I](https://tahabouhsine.com/blog/ai-illiteracy-pt1/)

On AI education in Morocco and the cost of staying behind.

## Companion papers

A few posts have longer write-ups as PDFs in [`papers/`](./papers):

- *Painting Arithmetic with Kernel MLPs* — the experiment behind *What an MLP Knows*
- *Not All Infinities Are Equal* — the cross-entropy asymmetry result
- *Opposite Is Not Different* — the three-landmark argument
- *Activations Are Bad for Geometry* — the Jacobian-modulation theorem

## Stack

Astro 5 + MDX, KaTeX for math, giscus for comments, hand-rolled canvas viz.
See [`CONTRIBUTING.md`](./CONTRIBUTING.md) if you want to know how to build
and deploy.
