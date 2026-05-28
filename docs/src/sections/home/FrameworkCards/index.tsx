import type { CSSProperties, ReactNode } from "react";
import { PauseOffscreen } from "@site/src/components/PauseOffscreen";
import styles from "./FrameworkCards.module.scss";

type Card = {
  icon: ReactNode;
  heading: string;
  description: string;
};

/* Shield + crosshair glyph — adversarial / security framework feel.
 * The crosshair pulses inside a static shield outline so the card
 * reads as "industry-standard threat taxonomies". */
const ShieldGlyph: React.FC = () => {
  return (
    <svg
      viewBox="0 0 64 48"
      className={styles.glyph}
      aria-hidden
      focusable="false"
    >
      {/* Shield outline */}
      <path
        d="M32 6 L48 12 L48 26 C48 34 41 41 32 44 C23 41 16 34 16 26 L16 12 Z"
        className={styles.shieldOutline}
      />
      {/* Crosshair: ring + two perpendicular ticks + center dot */}
      <g className={styles.shieldCrosshair}>
        <circle cx="32" cy="24" r="5" className={styles.crosshairRing} />
        <line x1="32" y1="17" x2="32" y2="20" className={styles.crosshairTick} />
        <line x1="32" y1="28" x2="32" y2="31" className={styles.crosshairTick} />
        <line x1="25" y1="24" x2="28" y2="24" className={styles.crosshairTick} />
        <line x1="36" y1="24" x2="39" y2="24" className={styles.crosshairTick} />
        <circle cx="32" cy="24" r="1.4" className={styles.crosshairDot} />
      </g>
    </svg>
  );
};

/* Document with seal glyph — compliance / regulation feel. The seal
 * scales in after the text lines settle so the card reads as
 * "audit-ready, attested by a regulator". */
const ComplianceDocGlyph: React.FC = () => {
  const lines = [
    { y: 14, w: 18, delay: 0 },
    { y: 19, w: 14, delay: 0.15 },
    { y: 24, w: 20, delay: 0.3 },
    { y: 29, w: 12, delay: 0.45 },
  ];
  return (
    <svg
      viewBox="0 0 64 48"
      className={styles.glyph}
      aria-hidden
      focusable="false"
    >
      {/* Document outline with a folded corner */}
      <path
        d="M14 6 L40 6 L48 14 L48 42 L14 42 Z"
        className={styles.docOutline}
      />
      <path
        d="M40 6 L40 14 L48 14"
        className={styles.docOutline}
      />
      {/* Text lines */}
      {lines.map((l, i) => (
        <line
          key={i}
          x1="18"
          y1={l.y}
          x2={18 + l.w}
          y2={l.y}
          className={styles.docLine}
          style={{ animationDelay: `${l.delay}s` } as CSSProperties}
        />
      ))}
      {/* Approval seal */}
      <circle cx="40" cy="36" r="5" className={styles.docSealRing} />
      <path
        d="M37.5 36 L39.5 38 L42.5 34"
        className={styles.docSealCheck}
      />
    </svg>
  );
};

/* Stacked dataset glyph — three layered rows representing curated
 * benchmark corpora (BeaverTails, Aegis). Each row dims/brightens in
 * sequence so the stack feels like a deck of test cases. */
const DatasetStackGlyph: React.FC = () => {
  const layers = [
    { y: 11, delay: 0 },
    { y: 22, delay: 0.25 },
    { y: 33, delay: 0.5 },
  ];
  return (
    <svg
      viewBox="0 0 64 48"
      className={styles.glyph}
      aria-hidden
      focusable="false"
    >
      {layers.map((l, i) => (
        <g
          key={i}
          className={styles.stackLayer}
          style={{ animationDelay: `${l.delay}s` } as CSSProperties}
        >
          <rect
            x="12"
            y={l.y}
            width="40"
            height="8"
            rx="1.5"
            className={styles.stackPlate}
          />
          {/* Two short ticks per layer to suggest tabular rows */}
          <line
            x1="16"
            y1={l.y + 4}
            x2="24"
            y2={l.y + 4}
            className={styles.stackTick}
          />
          <line
            x1="28"
            y1={l.y + 4}
            x2="40"
            y2={l.y + 4}
            className={styles.stackTick}
          />
        </g>
      ))}
    </svg>
  );
};

const CARDS: Card[] = [
  {
    icon: <ShieldGlyph />,
    heading: "Security & threat frameworks",
    description:
      "OWASP Top 10 for LLMs and MITRE ATLAS — industry-standard adversarial taxonomies in one framework.",
  },
  {
    icon: <ComplianceDocGlyph />,
    heading: "Compliance & governance",
    description:
      "Map your model's risk posture against NIST AI RMF and EU AI Act controls — audit-ready, no separate tooling required.",
  },
  {
    icon: <DatasetStackGlyph />,
    heading: "Safety benchmark datasets",
    description:
      "Test against curated red-teaming corpora — BeaverTails and Aegis — straight from the runner, no glue code.",
  },
];

const FrameworkCards: React.FC = () => {
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

export default FrameworkCards;
