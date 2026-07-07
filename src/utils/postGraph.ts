// Build-time extraction and layout of the post cross-reference graph.
//
// Nodes are the visible series explainers (companions fold into their
// explainer). Edges are the internal /blog/<slug>/ links found in post bodies,
// deduped, with adjacent-in-series pairs dropped because the lane spine
// already draws reading-order adjacency. Everything here is deterministic and
// runs at build; the client only gets static SVG.

import { SERIES, type SeriesDef } from '../data/series';

export interface GraphPost {
  id: string;
  body?: string;
  data: { title: string; draft?: boolean; companion?: string };
}

export interface MapNode {
  slug: string;
  title: string;
  arcId: string;
  /** 0-based position among the visible explainers of its arc. */
  part: number;
  draft: boolean;
  hasCompanion: boolean;
}

export interface MapEdge {
  /** Explainer slug of the post that cites... */
  source: string;
  /** ...this explainer slug. */
  target: string;
  kind: 'intra' | 'inter';
}

export interface PostGraph {
  arcs: { def: SeriesDef; nodes: MapNode[] }[];
  edges: MapEdge[];
}

const stripId = (s: string) => s.replace(/\.(md|mdx)$/, '');

export function buildPostGraph(posts: GraphPost[]): PostGraph {
  const byId = new Map(posts.map((p) => [stripId(p.id), p]));

  // companion slug -> explainer slug, for folding both edge endpoints.
  const mainOf = new Map<string, string>();
  const companionOf = new Map<string, string>();
  for (const p of posts) {
    const cid = p.data.companion && stripId(p.data.companion);
    if (cid && byId.has(cid)) {
      mainOf.set(cid, stripId(p.id));
      companionOf.set(stripId(p.id), cid);
    }
  }

  const arcs = SERIES.map((def) => ({
    def,
    nodes: def.slugs
      .filter((s) => byId.has(s))
      .map((s, i): MapNode => ({
        slug: s,
        title: byId.get(s)!.data.title,
        arcId: def.id,
        part: i,
        draft: Boolean(byId.get(s)!.data.draft),
        hasCompanion: companionOf.has(s),
      })),
  })).filter((a) => a.nodes.length > 0);

  const nodeIndex = new Map<string, MapNode>();
  for (const a of arcs) for (const n of a.nodes) nodeIndex.set(n.slug, n);

  const fold = (slug: string) => mainOf.get(slug) ?? slug;
  const seen = new Set<string>();
  const edges: MapEdge[] = [];

  for (const p of posts) {
    const source = fold(stripId(p.id));
    const src = nodeIndex.get(source);
    for (const m of (p.body ?? '').matchAll(/\]\(\/blog\/([a-z0-9-]+)/g)) {
      const raw = m[1];
      if (!byId.has(raw) && !nodeIndex.has(raw)) {
        console.warn(`[postGraph] ${stripId(p.id)} links to unknown post: /blog/${raw}/`);
        continue;
      }
      const target = fold(raw);
      const tgt = nodeIndex.get(target);
      // Keep only edges between series explainers; standalone posts and
      // writeups have no lane to sit on.
      if (!src || !tgt || source === target) continue;
      const key = `${source}→${target}`;
      if (seen.has(key)) continue;
      seen.add(key);
      // Adjacent parts of the same arc are drawn by the spine, not an edge.
      if (src.arcId === tgt.arcId && Math.abs(src.part - tgt.part) === 1) continue;
      edges.push({ source, target, kind: src.arcId === tgt.arcId ? 'intra' : 'inter' });
    }
  }

  return { arcs, edges };
}

// ── deterministic layout ────────────────────────────────────────────────────

export interface LaidOutNode extends MapNode {
  x: number;
  y: number;
  /** Short label under/over the node; full title stays in <title>. */
  label: string;
  /** Whether the label sits above or below the node (alternates). */
  labelAbove: boolean;
}

export interface LaidOutArc {
  def: SeriesDef;
  y: number;
  nodes: LaidOutNode[];
  spine: { x1: number; x2: number };
}

export interface LaidOutEdge extends MapEdge {
  path: string;
}

export interface Layout {
  width: number;
  height: number;
  arcs: LaidOutArc[];
  edges: LaidOutEdge[];
}

const WIDTH = 1080;
const MARGIN_X = 70;
const LANE_TOP = 96;
const LANE_GAP = 128;

/** First few words of a title, budgeted to stay readable at map scale. */
function shortLabel(title: string, budget = 18): string {
  const words = title.replace(/[.:!?]+$/, '').split(/\s+/);
  let out = words[0];
  for (let i = 1; i < words.length && out.length + 1 + words[i].length <= budget; i++) {
    out += ' ' + words[i];
  }
  return out;
}

export function layoutGraph(graph: PostGraph): Layout {
  const maxLen = Math.max(...graph.arcs.map((a) => a.nodes.length));
  const step = maxLen > 1 ? (WIDTH - 2 * MARGIN_X) / (maxLen - 1) : 0;
  const nodeX = (part: number) => MARGIN_X + part * step;

  const pos = new Map<string, { x: number; y: number }>();
  const arcs: LaidOutArc[] = graph.arcs.map((a, i) => {
    const y = LANE_TOP + i * LANE_GAP;
    const nodes = a.nodes.map((n): LaidOutNode => {
      const x = nodeX(n.part);
      pos.set(n.slug, { x, y });
      return { ...n, x, y, label: shortLabel(n.title), labelAbove: n.part % 2 === 1 };
    });
    return {
      def: a.def,
      y,
      nodes,
      spine: { x1: nodeX(0), x2: nodeX(a.nodes.length - 1) },
    };
  });

  const edges: LaidOutEdge[] = graph.edges.map((e) => {
    const s = pos.get(e.source)!;
    const t = pos.get(e.target)!;
    let path: string;
    if (e.kind === 'intra') {
      // Quadratic arc above the lane; taller for longer jumps, capped so it
      // stays inside the lane gap.
      const span = Math.abs(s.x - t.x) / (step || 1);
      const h = Math.min(16 + 5 * span, 48);
      path = `M ${s.x} ${s.y} Q ${(s.x + t.x) / 2} ${s.y - h} ${t.x} ${t.y}`;
    } else {
      // Vertical S-curve between lanes.
      const my = (s.y + t.y) / 2;
      path = `M ${s.x} ${s.y} C ${s.x} ${my}, ${t.x} ${my}, ${t.x} ${t.y}`;
    }
    return { ...e, path };
  });

  const height = LANE_TOP + (arcs.length - 1) * LANE_GAP + 72;
  return { width: WIDTH, height, arcs, edges };
}
