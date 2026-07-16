// Central registry of blog series ("arcs").
//
// Each series lists its EXPLAINER slugs only, in reading order. JAX companions
// attach to their explainer automatically through the existing frontmatter
// pair mechanism (see src/utils/posts.ts), so they never appear here: a
// companion inherits its explainer's position in the series.
//
// Slugs are content ids without extension. Draft filtering is NOT done here;
// callers must intersect these slugs with the visible (postFilter-ed) post
// list so part numbers always match what is actually published.

export interface SeriesDef {
  id: string;
  /** Full series title, shown in the series navigation on post pages. */
  title: string;
  /** Compact label for tight spots like index-row chips. */
  short: string;
  /** One-or-two-sentence arc description for home cards and the series page hero. */
  description: string;
  /** 'ongoing' renders an "in progress" note; omit when the arc is complete. */
  status?: 'ongoing';
  /** Explainer slugs in reading order. */
  slugs: string[];
}

export const SERIES: SeriesDef[] = [
  {
    id: 'representation-geometry',
    title: 'Geometry of Representations',
    short: 'Representation Geometry',
    description:
      'Where do representations live, and what makes a latent space good? ' +
      'Contrastive learning, the geometry of embedding spaces, and why the usual activation functions work against it.',
    slugs: [
      'activations-are-bad-for-geometry',
      'opposite-is-not-different',
      'not-all-infinities-are-equal',
      'untangling-the-moons',
      'welch-bound-good-latent-space',
      'latent-on-the-spectrum',
      'three-states-of-information',
      'distillation-is-kernel-transfer',
      'modality-gap-complementary',
      'simo2-geometry-by-construction',
    ],
  },
  {
    id: 'attention-as-a-kernel',
    title: 'Attention Is a Kernel',
    short: 'Attention Kernel',
    description:
      'Attention, read as kernel regression: what the softmax is really computing, ' +
      'why that makes it explainable, and what happens when you make the kernel cheap.',
    slugs: [
      'attention-is-a-kernel',
      'what-an-mlp-knows',
      'cheap-attention-is-linear-attention',
      'why-attention-needs-qk-projections',
    ],
  },
  {
    id: 'weights-in-kernel-space',
    title: 'Weights in Kernel Space',
    short: 'Kernel Weights',
    description:
      'Once everything is a kernel, what is a weight? An interlude on RKHS foundations: ' +
      'where a weight lives, what it can be, and why the MLP block is a representer theorem.',
    slugs: [
      'readout-as-convex-combination',
      'where-does-a-weight-live',
      'what-can-a-weight-be',
      'mlp-block-is-a-representer-theorem',
      'regularization-is-a-price-list',
    ],
  },
  {
    id: 'prototype-networks',
    title: 'The Prototype Network',
    short: 'Prototype Network',
    description:
      'Replace the activation with a finite, positive-definite kernel and a network becomes ' +
      'a list of prototypes you can read, edit by hand, and finally collapse into a single fixed-point operator.',
    status: 'ongoing',
    slugs: [
      'what-a-finite-kernel-buys-an-mlp',
      'your-neuron-is-a-picture',
      'edit-a-network-by-hand',
      'train-the-features',
      'you-dont-have-to-train-the-features',
      'depth-by-construction',
      'calibration-of-a-bounded-net',
      'a-risk-model-that-names-its-reasons',
      'survival-model-on-trial',
      'your-network-is-a-fixed-point',
      'edit-a-fixed-point',
      'you-dont-have-to-solve-a-kernel-machine',
    ],
  },
  {
    id: 'networks-as-integrators',
    title: 'Networks as Integrators',
    short: 'Integrators',
    description:
      'Numerical analysis as an architecture catalog: skip connections as an Euler step, ' +
      'momentum nets as half of Newton, and conservation laws as testable predictions about hidden states.',
    status: 'ongoing',
    slugs: [
      'skip-connections-are-half-of-newton',
      'transformers-with-a-velocity-ledger',
      'a-network-that-conserves-energy',
      'backprop-without-the-memory',
      'depth-on-demand',
    ],
  },
];

const bySlug = new Map<string, SeriesDef>();
for (const s of SERIES) for (const slug of s.slugs) bySlug.set(slug, s);

/** The series a given EXPLAINER slug belongs to, or undefined. */
export function seriesForSlug(slug: string): SeriesDef | undefined {
  return bySlug.get(slug);
}
