import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";
import { BookOpen, Compass } from "lucide-react";
import { appName, gitConfig } from "./shared";

// Nav items rendered in the middle column of the top nav, between the
// logo and the search bar. Exported so our custom header slot
// (`src/components/NavHeader`) can consume it; deliberately NOT
// passed via Fumadocs' `links` option, because that flow places text
// items on the far right of the header — we want the classic "Logo |
// Nav — — Search | Icons" layout (Tailwind / Next.js docs style) with
// the items aligned under the main content column.
//
// Icons chosen for semantic clarity + visual distinction at 16px:
//   Docs   → BookOpen  (reading reference material)
//   Guides → Compass   (directional walkthroughs)
//
// The Enterprise route still exists at `/enterprise` but is
// intentionally hidden from the nav for now.
export const navLinks = [
  {
    text: "Docs",
    url: "/docs/getting-started",
    activeBase: "/docs",
    icon: <BookOpen />,
  },
  {
    text: "Guides",
    url: "/guides/guide-agentic-ai-red-teaming",
    activeBase: "/guides",
    icon: <Compass />,
  },
];

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <span
          role="img"
          aria-label={appName}
          style={{
            display: "block",
            height: "24px",
            width: "102px",
            backgroundColor: "var(--color-fd-foreground)",
            WebkitMask: 'url("/icons/DeepTeam.svg") no-repeat center / contain',
            mask: 'url("/icons/DeepTeam.svg") no-repeat center / contain',
          }}
        />
      ),
      // NOTE: no `nav.children` here — the nav link strip is rendered
      // directly inside our custom header slot (`NavHeader`) so it
      // lands in the middle grid column, right under the main content.
      // Fumadocs would otherwise stash `children` next to `navTitle`
      // in the left cell, which is the wrong column.
    },
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
    // `links` intentionally omitted — text items live in `navLinks`
    // (rendered by `NavHeader`); only the GitHub icon flows through
    // Fumadocs' `navItems` via `githubUrl`, and our header picks it
    // up from `useNotebookLayout().navItems`.
  };
}
