import { getCollection } from 'astro:content';

export async function GET() {
  const posts = await getCollection('blog', ({ data }) => !data.draft);
  // Strip MDX/JSX imports + component invocations so they don't pollute search.
  const stripMdx = (s: string) =>
    s
      .replace(/^import\s.+?$/gm, '')
      .replace(/<[A-Z][\w]*\b[^>]*\/?>/g, '')
      .replace(/<\/[A-Z][\w]*>/g, '')
      .replace(/```[\s\S]*?```/g, ' ')
      .replace(/`[^`]*`/g, ' ')
      .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/[#*_>`]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

  const data = posts
    .sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf())
    .map((p) => ({
      id: p.id,
      title: p.data.title,
      description: p.data.description ?? '',
      tags: p.data.tags ?? [],
      pubDate: p.data.pubDate.toISOString(),
      body: stripMdx(p.body ?? ''),
    }));

  return new Response(JSON.stringify(data), {
    headers: { 'Content-Type': 'application/json' },
  });
}
