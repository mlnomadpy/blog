// Draft visibility.
//
// A post with `draft: true` in its frontmatter is a work in progress: it should
// be fully viewable locally but must never appear in the deployed site (no index
// entry, no page, no RSS item, no OG image, no tag listing).
//
// `SHOW_DRAFTS` is on when running the dev server (`astro dev`) or when any build
// is run with the env var `SHOW_DRAFTS=true` (handy for `astro preview`). A normal
// production build — what the deploy runs — leaves it off, so drafts are excluded.
//
//   astro dev                         → drafts visible
//   SHOW_DRAFTS=true astro build      → drafts visible (preview them locally)
//   astro build                       → drafts hidden  (deploy)
export const SHOW_DRAFTS: boolean =
  import.meta.env.DEV ||
  (typeof process !== 'undefined' && process.env.SHOW_DRAFTS === 'true');

// Predicate for getCollection('blog', postFilter): always keep published posts;
// keep drafts only when SHOW_DRAFTS is on.
export const postFilter = ({ data }: { data: { draft?: boolean } }): boolean =>
  SHOW_DRAFTS || !data.draft;

// ── explainer ↔ JAX-companion pairing ──────────────────────────────────────
// An explainer post declares `companion: <slug>` in its frontmatter pointing at
// its runnable JAX companion. From the visible post list we derive both
// directions, so the companion does not need to point back.
type PairPost = { id: string; data: { title: string; companion?: string } };
const stripId = (s: string) => s.replace(/\.(md|mdx)$/, '');

export interface Pairs<T extends PairPost> {
  companionOf: Map<string, T>; // stripped explainer id → companion post
  mainOf: Map<string, T>;      // stripped companion id → explainer post
}

// Build the pairing maps from the (already draft-filtered) post list. A pair is
// only formed when BOTH posts are present, so a link never points at a hidden
// draft.
export function buildPairs<T extends PairPost>(posts: T[]): Pairs<T> {
  const byId = new Map(posts.map((p) => [stripId(p.id), p]));
  const companionOf = new Map<string, T>();
  const mainOf = new Map<string, T>();
  for (const p of posts) {
    const cid = p.data.companion;
    if (!cid) continue;
    const comp = byId.get(stripId(cid));
    if (comp) {
      companionOf.set(stripId(p.id), comp);
      mainOf.set(stripId(comp.id), p);
    }
  }
  return { companionOf, mainOf };
}

// Resolve the pairing for one post: is it an explainer (with a companion) or a
// companion (of an explainer)? Returns null for standalone posts.
export function pairFor<T extends PairPost>(
  post: T,
  pairs: Pairs<T>,
): { role: 'explainer' | 'companion'; other: T } | null {
  const id = stripId(post.id);
  const comp = pairs.companionOf.get(id);
  if (comp) return { role: 'explainer', other: comp };
  const main = pairs.mainOf.get(id);
  if (main) return { role: 'companion', other: main };
  return null;
}
