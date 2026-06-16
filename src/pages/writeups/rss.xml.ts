import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import { SITE_TITLE } from '../../consts';
import { postFilter } from '../../utils/posts';
import type { APIContext } from 'astro';

// Separate feed for the personal writeups collection, so subscribers to the
// AI-theory blog (/blog/rss.xml) and the writeups (/blog/writeups/rss.xml)
// never get mixed streams.
export async function GET(context: APIContext) {
  const writeups = await getCollection('writeups', postFilter);
  return rss({
    title: `${SITE_TITLE} — Writeups`,
    description: 'Personal writeups — opinions, notes, and reflections that aren’t AI theory.',
    site: context.site!,
    items: writeups
      .sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf())
      .map((post) => ({
        title: post.data.title,
        description: post.data.description ?? '',
        pubDate: post.data.pubDate,
        link: `/writeups/${post.id}/`,
      })),
  });
}
