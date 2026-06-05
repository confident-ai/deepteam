import type { Metadata } from 'next';
import Script from 'next/script';
import { RootProvider } from 'fumadocs-ui/provider/next';
import './global.css';
import 'katex/dist/katex.css';
import { Geist, Space_Grotesk } from 'next/font/google';
import { GeistPixelGrid } from 'geist/font/pixel';
import UtmCapture from '@/src/layouts/UtmCapture';
import SchemaInjector from '@/src/components/SchemaInjector/SchemaInjector';
import { buildWebSiteSchema } from '@/src/utils/schema-helpers';
import {
  appName,
  siteDescription,
  siteTitle,
  siteUrl,
} from '@/lib/shared';

const sans = Geist({
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap',
});

const heading = Space_Grotesk({
  subsets: ['latin'],
  weight: ['300', '400', '500', '600', '700'],
  variable: '--font-heading',
  display: 'swap',
});

const disabledSearchHotKey = [
  {
    key: "__disabled__",
    display: null,
  },
];

// `%s` template mirrors Docusaurus' default `<title>` format so every
// SERP entry still reads "Page Title | {siteTitle}".
//
// `openGraph` / `twitter` defaults here set the site-wide baseline that
// every section inherits (Next's `generateMetadata` deep-merges onto
// this object). Per-page routes can override individual fields — the
// blog section adds `openGraph.type = 'article'` + publishedTime, the
// docs section adds per-page OG images, etc.
export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: siteTitle,
    template: `%s | ${siteTitle}`,
  },
  description: siteDescription,
  alternates: { canonical: '/' },
  openGraph: {
    type: 'website',
    siteName: appName,
    url: siteUrl,
    title: siteTitle,
    description: siteDescription,
    // Site-wide fallback preview image. Every section/page inherits
    // this unless it overrides `openGraph.images` (the docs section
    // does, swapping in a per-page `/og/docs/.../image.png`). Mirrors
    // the old Docusaurus `themeConfig.image = 'img/social_card.png'`
    // default so guides/blog/home never end up with a blank link
    // preview on social shares.
    images: '/img/social_card.png',
  },
  twitter: {
    card: 'summary_large_image',
    title: siteTitle,
    description: siteDescription,
    // Deliberately no `images:` here — we rely on X's documented
    // fallback to `og:image` when `twitter:image` is absent. Setting
    // it explicitly would stick the generic social card even on
    // docs pages whose `og:image` is overridden per-page (Next
    // replaces the whole `twitter` block across nested
    // `generateMetadata` calls instead of deep-merging, so a section
    // override of just `twitter.images` would clobber the `card` /
    // `site` / `creator` fields here). LinkedIn / Slack / Discord /
    // Facebook read `og:image` directly, so the single `og:image`
    // source-of-truth covers every surface.
  },
};

// Organization schema mirrored from the old Docusaurus `headTags` block
// (docusaurus.config.ts:161-181). Rendered once in <head> via the App
// Router layout — Next will keep JSON-LD scripts where they are placed.
const organizationJsonLd = {
  '@context': 'https://schema.org',
  '@type': 'Organization',
  name: 'Confident AI Inc.',
  alternateName: 'Confident AI',
  url: siteUrl,
  logo: `${siteUrl}/icons/DeepTeam.svg`,
  sameAs: [
    'https://github.com/confident-ai/deepteam',
    'https://discord.gg/3SEyvpgu2f',
  ],
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html
      lang="en"
      className={`${sans.variable} ${heading.variable} ${GeistPixelGrid.variable}`}
      suppressHydrationWarning
    >
      <head>
        {/*
          Two site-wide JSON-LD blocks rendered once per page:
          `Organization` (mirrored from the old Docusaurus `headTags`)
          and `WebSite` (so crawlers have a canonical top-level entity
          to hang everything else off of). Both use the shared
          `SchemaInjector` helper, which safely escapes `</` inside the
          serialized JSON.
        */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(organizationJsonLd) }}
        />
        <SchemaInjector schema={buildWebSiteSchema()} />
      </head>
      <body className="flex flex-col min-h-screen font-sans">
        <UtmCapture />
        <RootProvider search={{ hotKey: disabledSearchHotKey }}>
          {children}
        </RootProvider>
        {/*
          Analytics parity with the old Docusaurus site
          (docusaurus.config.ts:111-127). `afterInteractive` keeps these
          out of the critical path while still firing on every page
          navigation — same effective behavior as the old
          `<script defer>` tags.
        */}
        <Script
          src="https://www.googletagmanager.com/gtag/js?id=G-N2EGDDYG9M"
          strategy="afterInteractive"
        />
        <Script id="ga-init" strategy="afterInteractive">
          {`window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', 'G-N2EGDDYG9M');`}
        </Script>
        <Script
          src="https://plausible.io/js/script.tagged-events.js"
          data-domain="trydeepteam.com"
          strategy="afterInteractive"
        />
      </body>
    </html>
  );
}
