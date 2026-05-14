export const SITE_TITLE = 'Records of the !mmortal Data Scientist';
export const SITE_DESCRIPTION = 'Notes on AI, math, and the long road. Live slow, die whenever.';
export const SITE_AUTHOR = 'Taha Bouhsine';

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
