// No-op for now. URLs are prefixed with /blog via astro.config base, but the
// dist/ tree is flat — the gh-pages branch root holds index.html, _astro/, etc.
// Whoever serves it is responsible for mapping /blog/* requests to those files
// (e.g. project-repo URL mlnomadpy.github.io/blog/).
//
// Search is JSON-based (see src/pages/search.json.ts) and emitted directly by
// astro build, so no Pagefind step is needed here.
console.log('[postbuild] nothing to do');
