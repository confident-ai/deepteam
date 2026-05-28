import React from "react";
import Link from "@docusaurus/Link";
import styles from "./styles.module.scss";

const LINKS = {
  product: [
    { label: "Getting Started", to: "/docs/getting-started" },
    { label: "Vulnerabilities", to: "/docs/red-teaming-vulnerabilities" },
    {
      label: "Adversarial Attacks",
      to: "/docs/red-teaming-adversarial-attacks",
    },
    { label: "Guardrails", to: "/docs/guardrails-introduction" },
    { label: "Frameworks", to: "/docs/frameworks-introduction" },
  ],
  reads: [
    {
      label: "Red Teaming AI Agents",
      to: "/guides/guide-agentic-ai-red-teaming",
    },
    { label: "Red Teaming RAG", to: "/guides/guide-red-teaming-agentic-rag" },
    { label: "Safety Frameworks", to: "/guides/guide-safety-frameworks" },
    { label: "Building Custom Attacks", to: "/guides/guide-custom-attacks" },
    { label: "Deploying Guardrails", to: "/guides/guide-deploying-guardrails" },
  ],
  ecosystem: [
    { label: "Confident AI", href: "https://www.confident-ai.com" },
    { label: "DeepEval", href: "https://deepeval.com" },
  ],
};

function GitHubIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
      <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
    </svg>
  );
}

function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={styles.inner}>
        <div className={styles.top}>
          <div className={styles.brand}>
            <img
              src="/icons/DeepTeam.svg"
              alt="DeepTeam"
              className={styles.logo}
            />
            <p className={styles.tagline}>
              Open-source LLM red teaming framework. Apache 2.0 licensed.
            </p>
            <Link
              href="https://github.com/confident-ai/deepteam"
              className={styles.starButton}
            >
              <GitHubIcon /> Star us on GitHub
            </Link>
          </div>

          <div className={styles.columns}>
            <div className={styles.column}>
              <h4 className={styles.columnTitle}>Product</h4>
              <ul className={styles.columnList}>
                {LINKS.product.map((link) => (
                  <li key={link.label}>
                    <Link to={link.to} className={styles.link}>
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>

            <div className={styles.column}>
              <h4 className={styles.columnTitle}>Very Useful Reads</h4>
              <ul className={styles.columnList}>
                {LINKS.reads.map((link) => (
                  <li key={link.label}>
                    <Link to={link.to} className={styles.link}>
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>

            <div className={styles.column}>
              <h4 className={styles.columnTitle}>Ecosystem</h4>
              <ul className={styles.columnList}>
                {LINKS.ecosystem.map((link) => (
                  <li key={link.label}>
                    <Link href={link.href} className={styles.link}>
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        <div className={styles.divider} />

        <div className={styles.bottom}>
          <span className={styles.copyright}>
            &copy; {new Date().getFullYear()} Confident AI Inc. Made with 🖤 and
            confidence.
          </span>
        </div>
      </div>

      {/* <img
        src="/icons/DeepTeam.svg"
        alt=""
        className={styles.wordmark}
        aria-hidden="true"
      /> */}
    </footer>
  );
}

export default React.memo(Footer);
