// Tag metadata for topic pages and breadcrumbs. Gives each tag a readable name
// and a keyword-rich description so /blog/tag/<tag>/ pages can rank for topic
// queries (instead of a thin "Posts tagged #tag"), and so per-post breadcrumbs
// can name the topic level. Curated descriptions for the main topical tags; a
// templated fallback for the long tail.

const NAMES: Record<string, string> = {
  ml: 'Machine Learning',
  ai: 'Artificial Intelligence',
  jax: 'JAX',
  flax: 'Flax',
  nnx: 'Flax NNX',
  clip: 'CLIP',
  siglip: 'SigLIP',
  simclr: 'SimCLR',
  supcon: 'SupCon',
  infonce: 'InfoNCE',
  rkhs: 'RKHS',
  mlp: 'MLP',
  rope: 'RoPE',
  afcl: 'AFCL',
  simo2: 'SimO₂',
  optax: 'Optax',
  'simplex-etf': 'Simplex ETF',
  'yat-kernel': 'Yat Kernel',
  'yat-unit': 'Yat Unit',
  'query-key': 'Query–Key',
};

const titleCase = (s: string) =>
  s.split('-').map((w) => (w ? w[0].toUpperCase() + w.slice(1) : '')).join(' ');

export const tagName = (tag: string): string => NAMES[tag] ?? titleCase(tag);

// Curated, keyword-front-loaded descriptions (~150 chars) for the main topics.
const DESCRIPTIONS: Record<string, string> = {
  ml: 'Machine learning research notes: interpretability, kernels, contrastive learning, attention, and representation geometry, each with live interactive visualizations.',
  kernels: 'Kernel methods in deep learning: the softmax/attention kernel, RKHS feature maps, the Yat kernel, and Nadaraya–Watson smoothing, explained with interactive demos.',
  transformers: 'How transformers actually work: attention as a kernel, Q/K bilinear forms, the MLP readout, and linear-attention approximations, with live visualizations.',
  interpretability: 'Mechanistic and white-box interpretability: reading attention heads, MLP prototypes, induction circuits, and the geometry of learned representations.',
  'representation-learning': 'Representation learning geometry: neural collapse, the simplex and Welch bound, contrastive objectives, and what a good latent space actually looks like.',
  'contrastive-learning': 'Contrastive learning explained: InfoNCE, SimCLR, SupCon, CLIP, SigLIP, alignment and uniformity, and the geometry they impose, with interactive demos.',
  contrastive: 'Contrastive learning, loss by loss: pair, triplet, InfoNCE, CLIP, SupCon, SigLIP, and alignment/uniformity, watched as they organize embeddings live.',
  attention: 'Attention mechanisms from scratch: scaled dot-product attention, Q/K projections as a bilinear form, attention as a kernel smoother, and linear-time variants.',
  'self-attention': 'Self-attention as a learned bilinear relation and a Nadaraya–Watson kernel smoother: why Q and K projections matter and how heads become readable.',
  jax: 'Runnable JAX and Flax NNX implementations: contrastive losses, linear attention, Q/K attention, and prototype readouts, with the math running live.',
  flax: 'Flax NNX implementations of attention, linear attention, and prototype readouts, written as nnx.Module with the linear algebra running live.',
  nnx: 'Flax NNX implementations of attention, linear attention, and prototype readouts, written as nnx.Module with the linear algebra running live.',
  implementation: 'Runnable JAX / Flax NNX companions that turn each theory post into executable, checkable code.',
  'neural-collapse': 'Neural collapse and the terminal phase of training: why class means form a simplex equiangular tight frame, and what the Welch bound says about it.',
  embeddings: 'Embedding geometry: the modality gap, simplex codebooks, the Welch bound, and what trained features spend their dimensions on.',
  clip: 'CLIP and the modality gap: why image and text embeddings sit in separate cones, and when the gap is a choice rather than a bug.',
  siglip: 'SigLIP and the sigmoid contrastive loss: how it differs from InfoNCE, the modality gap it leaves, and the constellation geometry it targets.',
  infonce: 'InfoNCE and friends: the contrastive objective behind CLIP and SimCLR, its alignment/uniformity decomposition, and the geometry it learns.',
  mlp: 'The transformer MLP as a kernel and a prototype readout: reading W_out as output prototypes and the convex/conic/affine/linear regimes.',
  'mechanistic-interpretability': 'Mechanistic interpretability: induction heads, the QK and OV circuits, feed-forward key-value memories, and the bilinear forms that route attention.',
  'latent-space': 'What makes a good latent space: the simplex, the Welch bound, spectral codebooks, and the information that lives between prototypes.',
  'simplex-etf': 'The simplex equiangular tight frame: the maximally-separated configuration of class prototypes, the Welch-bound optimum, and neural collapse.',
  'welch-bound': 'The Welch bound: the coherence lower bound from radio engineering that sets the best latent geometry when concepts outnumber dimensions.',
  rkhs: 'Reproducing kernel Hilbert spaces in deep learning: feature maps, the softmax kernel, positive-definite vs nonnegative kernels, and Mercer theory.',
  'linear-attention': 'Linear attention and Performers: approximating the softmax kernel with random features so the N×N attention matrix never forms.',
  performer: 'Performers and FAVOR+: positive random features that make softmax attention linear-time while staying an unbiased kernel estimate.',
  'random-features': 'Random features: approximating an infinite-dimensional kernel feature map with a finite sketch, the lever behind linear attention.',
  'modality-gap': 'The modality gap in CLIP-style models: separated cones for image and text, why it appears, and whether to separate or represent.',
  bilinear: 'Attention as a learned bilinear form x_iᵀW_QW_Kᵀx_j: directional, low-rank, role-aware, and split into a symmetric metric and an antisymmetric part.',
  'query-key': 'Q and K projections as a learned bilinear form: directional, low-rank, role-aware attention, and the symmetric/antisymmetric split.',
  prototypes: 'Prototype geometry in neural networks: output prototypes in the MLP readout, class-mean codebooks, and convex combinations.',
  'training-dynamics': 'Training dynamics: loss plateaus as phase transitions, saddle-to-saddle learning, and the random, organized, and structured states of a representation.',
  orthogonality: 'Why orthogonality, not opposition, is maximal difference for unit vectors, and what that means for CLIP, InfoNCE, and SimCLR.',
  geometry: 'The geometry of representations: simplices, frames, cones, and the metrics learned objectives actually impose.',
  ai: 'Notes on AI literacy, education, and policy, with a focus on Morocco and Africa.',
};

export interface TagMeta {
  name: string;
  title: string;
  description: string;
}

export function tagMeta(tag: string, count: number): TagMeta {
  const name = tagName(tag);
  const s = count === 1 ? '' : 's';
  const description =
    DESCRIPTIONS[tag] ??
    `${count} long-form post${s} on ${name}: machine-learning research by Taha Bouhsine, each built around live, in-browser interactive visualizations.`;
  return { name, title: `${name} (${count} post${s})`, description };
}
