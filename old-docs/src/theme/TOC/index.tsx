import { type ReactNode } from "react";
import clsx from "clsx";
import TOCItems from "@theme/TOCItems";
import type { Props } from "@theme/TOC";
import styles from "./styles.module.scss";

// Using a custom className
// This prevents TOCInline/TOCCollapsible getting highlighted by mistake
const LINK_CLASS_NAME = "table-of-contents__link toc-highlight";
const LINK_ACTIVE_CLASS_NAME = "table-of-contents__link--active";

export default function TOC({ className, ...props }: Props): ReactNode {
  return (
    <div className={clsx(styles.tableOfContents, className)}>
      {/* Scrollable container for TOC items */}
      <div className={clsx(styles.tocItemsContainer, "thin-scrollbar")}>
        <TOCItems
          {...props}
          linkClassName={LINK_CLASS_NAME}
          linkActiveClassName={LINK_ACTIVE_CLASS_NAME}
        />
      </div>
      <div className={styles.promoCard}>
        <img
          src="/icons/red-logo.svg"
          alt="Confident AI"
          className={styles.logo}
        />
        <div className={styles.heading}>
          Try DeepTeam on Confident AI Enterprise
        </div>
        <div className={styles.description}>
          Run red teaming assessments against safety frameworks, view risk
          reports, schedule recurring audits, and deploy guardrails in
          production.
        </div>
        <div
          className={styles.button}
          onClick={() =>
            (window.location.href =
              "https://calendly.com/d/cqbp-t88-y4j/confident-ai-intro-call")
          }
        >
          Book a demo
        </div>
      </div>
    </div>
  );
}
