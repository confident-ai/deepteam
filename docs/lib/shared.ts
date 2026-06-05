export const appName = 'DeepTeam';

/**
 * Canonical public origin for the site. Single source of truth for
 * every absolute URL we emit (sitemap, robots, JSON-LD, `metadataBase`,
 * OG/image URLs, etc.) so a domain change only needs one edit.
 */
export const siteUrl = 'https://trydeepteam.com';

/**
 * Site title used as the default `<title>` on routes that don't set
 * their own, and as the suffix in the root layout's title template
 * (`%s | {siteTitle}`). Kept verbatim from the old Docusaurus
 * `config.title` for SERP continuity.
 */
export const siteTitle =
  'DeepTeam by Confident AI - The LLM Red Teaming Framework';

/**
 * Short meta-description used on the homepage and as the fallback for
 * pages without a frontmatter `description:` and no extractable body
 * paragraph.
 */
export const siteDescription =
  'DeepTeam is the open-source LLM red teaming framework for testing and securing LLM applications.';

export const docsRoute = '/docs';
export const docsImageRoute = '/og/docs';

/**
 * Raw-markdown API route prefix for any section. We host a Next.js
 * route handler at `/llms.mdx/<section>/<slug>/content.md` for every
 * section that wants the "Copy as Markdown" button.
 *
 * Pass either a section name (`"docs"`) or a source's `baseUrl`
 * (`"/guides"`) — both work.
 */
export function contentRouteFor(sectionOrBaseUrl: string) {
  const section = sectionOrBaseUrl.replace(/^\/+/, '').split('/')[0];
  return `/llms.mdx/${section}`;
}

/** Back-compat alias. */
export const docsContentRoute = contentRouteFor('docs');

export const gitConfig = {
  user: 'confident-ai',
  repo: 'deepteam',
  branch: 'main',
};

/** Community Discord invite — used by the `<DiscordButton>` CTA.
 *  Single source of truth so rotating the invite is a one-line change. */
export const discordUrl = 'https://discord.gg/3SEyvpgu2f';
