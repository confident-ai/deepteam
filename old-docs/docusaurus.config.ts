import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const config: Config = {
  plugins: [
    "docusaurus-plugin-sass",
    [
      "@docusaurus/plugin-content-docs",
      {
        id: "guides",
        path: "guides",
        routeBasePath: "guides",
        sidebarPath: require.resolve("./sidebarGuides.js"),
        editUrl: "https://github.com/confident-ai/deepteam/edit/main/docs/",
        showLastUpdateAuthor: true,
        showLastUpdateTime: true,
      },
    ],
    [
      "@docusaurus/plugin-content-blog",
      {
        id: "blogs",
        path: "blog",
        routeBasePath: "blog",
        blogSidebarCount: 0,
      },
    ],
  ],

  title: "DeepTeam by Confident AI - The LLM Red Teaming Framework",
  tagline: "Red Teaming Framework for LLMs",
  favicon: "img/fav.ico",

  url: "https://trydeepteam.com",
  baseUrl: "/",

  onBrokenLinks: "warn",

  headTags: [
    {
      tagName: "meta",
      attributes: {
        name: "algolia-site-verification",
        content: "0BD7859165065E7F",
      },
    },
    // GEO / SEO Organization Schema injected here
    {
      tagName: "script",
      attributes: {
        type: "application/ld+json",
      },
      innerHTML: JSON.stringify({
        "@context": "https://schema.org",
        "@type": "Organization",
        name: "DeepTeam by Confident AI",
        alternateName: "DeepTeam - The LLM Security and Red Teaming Framework",
        url: "https://trydeepteam.com",
        logo: "https://trydeepteam.com/icons/DeepTeam.svg",
        sameAs: [
          "https://github.com/confident-ai/deepteam",
          "https://discord.gg/3SEyvpgu2f",
        ],
      }),
    },
  ],

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "@docusaurus/preset-classic",
      {
        blog: false,
        docs: {
          path: "docs",
          editUrl: "https://github.com/confident-ai/deepteam/edit/main/docs/",
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
          sidebarPath: require.resolve("./sidebars.js"),
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        theme: {
          customCss: require.resolve("./src/css/custom.scss"),
        },
        gtag: {
          trackingID: "G-N2EGDDYG9M",
          anonymizeIP: false,
        },
      } satisfies Preset.Options,
    ],
  ],
  themes: ["@docusaurus/theme-mermaid"],
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: "warn",
    },
  },
  scripts: [
    {
      src: "https://plausible.io/js/script.tagged-events.js",
      defer: true,
      "data-domain": "trydeepteam.com",
    },
    {
      src: "https://unpkg.com/lucide@latest",
      async: true,
    },
  ],
  stylesheets: [
    {
      href: "https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css",
      type: "text/css",
      integrity:
        "sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM",
      crossorigin: "anonymous",
    },
    {
      href: "https://fonts.googleapis.com/css2?family=Lexend+Deca:wght@500&display=swap",
      type: "text/css",
    },
  ],
  themeConfig: {
    image: "img/social_card.png",
    navbar: {
      logo: {
        alt: "DeepTeam Logo",
        src: "icons/DeepTeam.svg",
      },
      items: [
        {
          to: "docs/getting-started",
          position: "left",
          label: "Docs",
          activeBasePath: "docs",
        },
        {
          to: "guides/guide-agentic-ai-red-teaming",
          position: "left",
          label: "Guides",
          activeBasePath: "guides",
        },
        { to: "blog", label: "Blog", position: "left" },
        {
          href: "https://discord.gg/3SEyvpgu2f",
          className: "header-discord-link",
          position: "right",
        },
        {
          href: "https://github.com/confident-ai/deepteam",
          position: "right",
          className: "header-github-link",
        },
      ],
    },
    algolia: {
      appId: "5CUTDUUUNG",
      apiKey: "94dc139a8be236f7f48eb9013014e517",
      indexName: "Deepteam",
      contextualSearch: true,
    },
    colorMode: {
      defaultMode: "light",
      disableSwitch: false,
      respectPrefersColorScheme: false,
    },
    announcementBar: {
      id: "announcementBar-1",
      content:
        '⭐️ If you like DeepTeam, give it a star on <a target="_blank" rel="noopener noreferrer" href="https://github.com/confident-ai/deepteam">GitHub</a>! ⭐️',
      backgroundColor: "#de0064",
      textColor: "#000",
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Documentation",
          items: [
            {
              label: "Introduction",
              to: "/docs/getting-started",
            },
            {
              label: "Guides",
              to: "guides/guide-agentic-ai-red-teaming",
            },
          ],
        },
        {
          title: "Articles You Must Read",
          items: [
            {
              label: "How to jailbreak LLMs",
              to: "https://www.confident-ai.com/blog/how-to-jailbreak-llms-one-step-at-a-time",
            },
            {
              label: "OWASP Top 10 for LLMs",
              to: "https://www.confident-ai.com/blog/owasp-top-10-2025-for-llm-applications-risks-and-mitigation-techniques",
            },
            {
              label: "The comprehensive LLM safety guide",
              to: "https://www.confident-ai.com/blog/the-comprehensive-llm-safety-guide-navigate-ai-regulations-and-best-practices-for-llm-safety",
            },
            {
              label: "LLM evaluation metrics",
              to: "https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation",
            },
          ],
        },
        {
          title: "Red Teaming Community",
          items: [
            {
              label: "GitHub",
              to: "https://github.com/confident-ai/deepteam",
            },
            {
              label: "Discord",
              to: "https://discord.gg/a3K9c8GRGt",
            },
            {
              label: "Newsletter",
              to: "https://confident-ai.com/blog",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Confident AI Inc. Built with ❤️ and confidence.`,
    },
    prism: {
      theme: require("prism-react-renderer/themes/nightOwl"),
      additionalLanguages: ["python"],
      magicComments: [
        {
          className: "theme-code-block-highlighted-line",
          line: "highlight-next-line",
          block: { start: "highlight-start", end: "highlight-end" },
        },
        {
          className: "code-block-error-message",
          line: "highlight-next-line-error-message",
        },
        {
          className: "code-block-info-line",
          line: "highlight-next-line-info",
          block: {
            start: "highlight-info-start",
            end: "highlight-info-end",
          },
        },
      ],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
