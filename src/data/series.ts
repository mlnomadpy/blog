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
  /** Explainer slugs in reading order. */
  slugs: string[];
}

export const SERIES: SeriesDef[] = [
  {
    id: 'representation-geometry',
    title: 'Geometry of Representations',
    short: 'Representation Geometry',
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
    slugs: [
      'what-a-finite-kernel-buys-an-mlp',
      'your-neuron-is-a-picture',
      'edit-a-network-by-hand',
      'train-the-features',
      'you-dont-have-to-train-the-features',
      'depth-by-construction',
      'calibration-of-a-bounded-net',
      'a-risk-model-that-names-its-reasons',
      'your-network-is-a-fixed-point',
      'edit-a-fixed-point',
    ],
  },
  {
    id: 'networks-as-integrators',
    title: 'Networks as Integrators',
    short: 'Integrators',
    slugs: [
      'skip-connections-are-half-of-newton',
    ],
  },
];

const bySlug = new Map<string, SeriesDef>();
for (const s of SERIES) for (const slug of s.slugs) bySlug.set(slug, s);

/** The series a given EXPLAINER slug belongs to, or undefined. */
export function seriesForSlug(slug: string): SeriesDef | undefined {
  return bySlug.get(slug);
}
