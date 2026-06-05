import type { CSSProperties, ReactNode } from "react";
import { PauseOffscreen } from "@site/src/components/PauseOffscreen";
import styles from "./AdversaryCards.module.scss";

type Card = {
  icon: ReactNode;
  heading: string;
  description: string;
};

/* Animated glyph: pill grid representing the breadth of vulnerability
 * categories DeepTeam tests against. Each pill fades + slides in with
 * a staggered delay so the grid feels like risks being enumerated. */
const VulnerabilityPillsGlyph: React.FC = () => {
  // 2 rows × 3 columns of pills with slightly varied widths so the
  // grid reads as discrete vulnerability tags rather than a UI grid.
  const pills = [
    { x: 4, y: 8, w: 16, delay: 0 },
    { x: 23, y: 8, w: 12, delay: 0.18 },
    { x: 38, y: 8, w: 22, delay: 0.36 },
    { x: 4, y: 26, w: 20, delay: 0.54 },
    { x: 27, y: 26, w: 14, delay: 0.72 },
    { x: 44, y: 26, w: 16, delay: 0.9 },
  ];
  return (
    <svg
      viewBox="0 0 64 48"
      className={styles.glyph}
      aria-hidden
      focusable="false"
    >
      {pills.map((p, i) => (
        <rect
          key={i}
          x={p.x}
          y={p.y}
          width={p.w}
          height="11"
          rx="5.5"
          className={styles.glyphPill}
          style={{ animationDelay: `${p.delay}s` } as CSSProperties}
        />
      ))}
    </svg>
  );
};

/* Animated glyph: multi-turn conversation with per-turn attack flags
 * — the small dots fire in sequence to suggest an escalating attack
 * chain landing across multiple turns. */
const MultiTurnAttacksGlyph: React.FC = () => {
  const rows = [
    { bubbleX: 6, bubbleW: 20, scoreCx: 32, side: "user", delay: 0 },
    { bubbleX: 24, bubbleW: 28, scoreCx: 58, side: "agent", delay: 0.3 },
    { bubbleX: 6, bubbleW: 16, scoreCx: 28, side: "user", delay: 0.6 },
  ];
  return (
    <svg
      viewBox="0 0 64 48"
      className={styles.glyph}
      aria-hidden
      focusable="false"
    >
      {rows.map((r, i) => (
        <g key={i}>
          <rect
            x={r.bubbleX}
            y={6 + i * 13}
            width={r.bubbleW}
            height="9"
            rx="2.5"
            className={`${styles.glyphConvBubble} ${
              r.side === "agent"
                ? styles.glyphConvBubbleAgent
                : styles.glyphConvBubbleUser
            }`}
          />
          <circle
            cx={r.scoreCx}
            cy={10.5 + i * 13}
            r="2.2"
            className={styles.glyphAttackFlag}
            style={{ animationDelay: `${r.delay}s` } as CSSProperties}
          />
        </g>
      ))}
    </svg>
  );
};

const CARDS: Card[] = [
  {
    icon: <VulnerabilityPillsGlyph />,
    heading: "60+ adversarial vulnerabilities",
    description:
      "Bias, Toxicity, PII leakage, SQL Injection, BFLA, BOLA, RBAC, Hallucination, and dozens more — every major LLM risk category covered out of the box.",
  },
  {
    icon: <MultiTurnAttacksGlyph />,
    heading: "Multi-turn jailbreak attacks",
    description:
      "Crescendo, Linear, Tree, Sequential, and Bad-Likert-Judge — research-backed adversarial chains that simulate sophisticated multi-turn attackers.",
  },
];

const AdversaryCards: React.FC = () => {
  return (
    <PauseOffscreen>
      <div className={styles.grid}>
        {CARDS.map((card, i) => (
          <article key={i} className={styles.card}>
            <div className={styles.iconWrap}>{card.icon}</div>
            <h3 className={styles.heading}>{card.heading}</h3>
            <p className={styles.description}>{card.description}</p>
          </article>
        ))}
      </div>
    </PauseOffscreen>
  );
};

export default AdversaryCards;
