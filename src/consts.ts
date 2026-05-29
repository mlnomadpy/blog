export const SITE_TITLE = 'Records of the !mmortal Data Scientist';
export const SITE_TAGLINE = 'Notes on AI, math, and the long road. Live slow, die whenever.';
// Long-form description used in <meta name="description">, OG, and JSON-LD on
// the homepage. Keywords here are what SERPs index for the root URL.
export const SITE_DESCRIPTION =
  "Machine learning research notes by Taha Bouhsine — neural network interpretability, kernel methods, contrastive learning, attention mechanisms, RKHS, and transformer architectures. Long-form pieces with interactive visualisations.";
export const SITE_AUTHOR = 'Taha Bouhsine';
export const SITE_AUTHOR_URL = 'https://github.com/mlnomadpy';
export const SITE_AUTHOR_JOB_TITLE = 'Machine Learning Researcher';

// sameAs URLs — used in JSON-LD Person/Publisher schema and as <link rel="me">
// for IndieWeb / Mastodon verification and Google's author-discovery signal.
export const SITE_SAMEAS: readonly string[] = [
  'https://github.com/mlnomadpy',
  'https://www.linkedin.com/in/tahabsn/',
  'https://scholar.google.com/citations?user=IsBjb3EAAAAJ',
  'https://g.dev/tahabsn',
];

// Default Open Graph image, served from public/ at the site base path.
// Relative to BASE_URL — see src/components/BaseHead.astro for the URL build.
export const SITE_OG_IMAGE = 'og-default.svg';

// Google Analytics 4 Measurement ID, e.g. 'G-XXXXXXXXXX'. Leave empty to
// disable. The tag only loads in production builds (see BaseHead.astro), so
// local dev traffic never reaches GA. Per-post numbers come for free: every
// post is its own URL, so the GA "Pages and screens" report breaks down by
// post path automatically.
export const GA4_MEASUREMENT_ID = 'G-373438VHJB';

// giscus — comments via GitHub Discussions on mlnomadpy/blog.
// IDs fetched via the GitHub GraphQL API; "Announcements" category chosen
// because only maintainers + the giscus app can create top-level threads.
export const GISCUS = {
  repo: 'mlnomadpy/blog',
  repoId: 'MDEwOlJlcG9zaXRvcnk0MDQ4MzY2NDU=',
  category: 'Announcements',
  categoryId: 'DIC_kwDOGCFRJc4C9EiR',
  mapping: 'pathname',
  reactionsEnabled: '1',
  emitMetadata: '0',
  inputPosition: 'top',
  theme: 'preferred_color_scheme',
  lang: 'en',
} as const;
