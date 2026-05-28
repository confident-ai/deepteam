import Image from "next/image";
import Link from "next/link";
import type { ComponentType, SVGProps } from "react";
import {
  CircleCIMark,
  GitHubMark,
  OpenAIMark,
} from "@site/src/components/BrandMarks";
import styles from "./IntegrationGrid.module.scss";

type Integration = {
  name: string;
  /**
   * Static file in /public for use with next/image. Kept as a fallback
   * and so the icon exists on disk even for integrations we render
   * inline (next/image is still nice for preloading / link previews).
   */
  logo: string;
  /** Docs page the card links to. */
  href: string;
  /**
   * Optional inline SVG component. When set, the icon is rendered as
   * real SVG in the React tree so `fill="currentColor"` picks up the
   * page's foreground color and survives light/dark mode toggles.
   * Loading via <img>/next/image puts the SVG in a separate document
   * context where `currentColor` can't inherit from the host page,
   * which is why monochrome brand marks (OpenAI, GitHub, CircleCI,
   * Vercel AI SDK) need to be inlined.
   */
  inline?: ComponentType<SVGProps<SVGSVGElement>>;
};

type Category = {
  label: string;
  items: Integration[];
  columns: number;
};

/* ---------- Category config ---------- */

/* Destination docs pages. DeepTeam's CLI/YAML config covers both
 * model-provider setup (every simulator hits the same model API) and
 * CI/CD integration, so model + CI/CD tiles all land on the same
 * page until dedicated per-integration pages exist. Safety
 * frameworks each have a dedicated docs page already. */
const MODEL_DOCS = "/docs/red-teaming-yaml-cli#using-custom-models";
const CI_CD_DOCS = "/docs/red-teaming-yaml-cli";

const MODEL_PROVIDERS: Category = {
  label: "Model Providers",
  columns: 4,
  items: [
    { name: "OpenAI", logo: "/icons/integrations/openai.svg", href: MODEL_DOCS, inline: OpenAIMark },
    { name: "Claude", logo: "/icons/integrations/claude.svg", href: MODEL_DOCS },
    { name: "Gemini", logo: "/icons/integrations/gemini.svg", href: MODEL_DOCS },
    { name: "Azure OpenAI", logo: "/icons/integrations/azure.svg", href: MODEL_DOCS },
    { name: "AWS Bedrock", logo: "/icons/integrations/bedrock.svg", href: MODEL_DOCS },
    { name: "Vertex AI", logo: "/icons/integrations/vertext_ai.svg", href: MODEL_DOCS },
    { name: "Mistral", logo: "/icons/integrations/mistral.svg", href: MODEL_DOCS },
    { name: "LiteLLM", logo: "/icons/integrations/litellm.svg", href: MODEL_DOCS },
    { name: "Portkey", logo: "/icons/integrations/portkey.svg", href: MODEL_DOCS },
  ],
};

/* Safety, security, and compliance frameworks deepteam ships
 * support for out of the box. Each tile links to its dedicated docs
 * page under `content/docs/`. */
const SAFETY_FRAMEWORKS: Category = {
  label: "Safety Frameworks",
  columns: 3,
  items: [
    { name: "OWASP Top 10", logo: "/icons/integrations/owasp.svg", href: "/docs/frameworks-owasp-top-10-for-llms" },
    { name: "MITRE ATLAS", logo: "/icons/integrations/mitre.png", href: "/docs/frameworks-mitre-atlas" },
    { name: "NIST AI RMF", logo: "/icons/integrations/nist.png", href: "/docs/frameworks-nist-ai-rmf" },
    { name: "EU AI Act", logo: "/icons/integrations/eu-ai-act.png", href: "/docs/frameworks-eu-ai-act" },
    { name: "BeaverTails", logo: "/icons/integrations/hf-logo.svg", href: "/docs/frameworks-beavertails" },
    { name: "Aegis", logo: "/icons/integrations/nvidia.png", href: "/docs/frameworks-aegis" },
  ],
};

const CI_CD: Category = {
  label: "CI / CD",
  columns: 6,
  items: [
    { name: "GitHub Actions", logo: "/icons/integrations/github.svg", href: CI_CD_DOCS, inline: GitHubMark },
    { name: "GitLab CI", logo: "/icons/integrations/gitlab.svg", href: CI_CD_DOCS },
    { name: "Jenkins", logo: "/icons/integrations/jenkins.svg", href: CI_CD_DOCS },
    { name: "CircleCI", logo: "/icons/integrations/circleci.svg", href: CI_CD_DOCS, inline: CircleCIMark },
    { name: "Buildkite", logo: "/icons/integrations/buildkite.svg", href: CI_CD_DOCS },
    { name: "Azure Pipelines", logo: "/icons/integrations/azure-pipelines.svg", href: CI_CD_DOCS },
  ],
};

const IntegrationTile: React.FC<{ item: Integration }> = ({ item }: { item: Integration }) => {
  const Inline = item.inline;
  return (
    <Link
      href={item.href}
      className={styles.tile}
      aria-label={`${item.name} integration docs`}
    >
      <div
        className={`${styles.logoWrap}${Inline ? ` ${styles.logoWrapInline}` : ""}`}
      >
        {Inline ? (
          <Inline className={styles.logoInline} aria-label={`${item.name} logo`} />
        ) : (
          <Image
            src={item.logo}
            alt={`${item.name} logo`}
            width={32}
            height={32}
            className={styles.logo}
          />
        )}
      </div>
      <span className={styles.tileName}>{item.name}</span>
    </Link>
  );
};

const Panel: React.FC<{
  category: Category;
  className?: string;
}> = ({
  category,
  className,
}: {
  category: Category;
  className?: string;
}) => {
  return (
    <section
      className={`${styles.panel}${className ? ` ${className}` : ""}`}
      aria-labelledby={`integration-${category.label}`}
    >
      <header className={styles.panelHeader}>
        <span id={`integration-${category.label}`} className={styles.panelLabel}>
          {category.label}
        </span>
      </header>
      <div
        className={styles.tiles}
        style={{ ["--tile-cols" as string]: category.columns }}
      >
        {category.items.map((item) => (
          <IntegrationTile key={item.name} item={item} />
        ))}
      </div>
    </section>
  );
};

const IntegrationGrid: React.FC = () => {
  return (
    <div className={styles.grid}>
      <Panel category={SAFETY_FRAMEWORKS} className={styles.left} />
      <Panel category={MODEL_PROVIDERS} className={styles.right} />
      <Panel category={CI_CD} className={styles.full} />
    </div>
  );
};


export default IntegrationGrid;
