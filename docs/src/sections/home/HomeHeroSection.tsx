"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { ArrowUpRight } from "lucide-react";
import { PrimaryButton, SecondaryButton } from "@site/src/components/Buttons";
import { PauseOffscreen } from "@site/src/components/PauseOffscreen";
import { DYNAMIC_LOGOS } from "./CompanyLogos";
import styles from "./HomeSection.module.scss";

/**
 * One-shot "encrypted text decoder" effect for the install command.
 * Each character index has a lock time proportional to its position;
 * before its lock time the slot renders a random ASCII char from
 * `DECODER_CHARSET`, after it settles to the real character. Total
 * cycle is `durationMs`; the loop falls through once the last char
 * locks, so the effect only plays on mount.
 */
const DECODER_CHARSET =
  "abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*-=_+[]{}|;:,.<>?/~`";

function useDecodedText(target: string, durationMs: number): string {
  // Initial render uses the final string so the SSR-flushed HTML matches
  // the post-decode state — the useEffect kicks in client-side, swaps
  // it for scrambled text, then walks it back to `target`.
  const [display, setDisplay] = useState(target);

  useEffect(() => {
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = (now - start) / durationMs;
      if (t >= 1) {
        setDisplay(target);
        return;
      }
      // Lock the final ~15% of progress for the last char so the very
      // end of the animation feels deliberate, not a sudden snap.
      const lockProgress = t / 0.85;
      const next = Array.from(target)
        .map((c, i) => {
          if (c === " ") return " ";
          const lockAt = i / target.length;
          if (lockProgress >= lockAt) return c;
          return DECODER_CHARSET[
            Math.floor(Math.random() * DECODER_CHARSET.length)
          ];
        })
        .join("");
      setDisplay(next);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, durationMs]);

  return display;
}

type Brand = {
  name: string;
  slug: string;
};

const BRANDS: Brand[] = [
  // Row 1 — required anchors (LEGO col 3, Uber col 5, Google and OpenAI split)
  { name: "Google", slug: "google" },
  { name: "Uber", slug: "uber" },
  { name: "OpenAI", slug: "openai" },
  { name: "LEGO", slug: "lego" },
  { name: "Visa", slug: "visa" },
  // Row 2 — blue / red / silver / orange / red-yellow
  { name: "Toyota", slug: "toyota" },
  { name: "Adobe", slug: "adobe" },
  { name: "Walmart", slug: "walmart" },
  { name: "Mastercard", slug: "mastercard" },
  { name: "AWS", slug: "aws" },
  // Row 3 — mono / yellow-dark / green / blue-yellow / multi
  { name: "Samsung", slug: "samsung" },
  { name: "EY", slug: "ey" },
  { name: "Mercedes-Benz", slug: "benz" },
  { name: "NVIDIA", slug: "nvidia" },
  { name: "Microsoft", slug: "microsoft" },
  // Row 4 — blue / red / blue / red / teal (alternating)
  { name: "Bosch", slug: "bosch" },
  { name: "Pfizer", slug: "pfizer" },
  { name: "AXA", slug: "axa" },
  { name: "Siemens", slug: "siemens" },
  { name: "CVS Health", slug: "cvs-health" },
];

const BANNER_ITEMS = [
  "60+ adversarial attacks & vulnerabilities",
  "Used by leading security & AI teams",
  "Aligned with OWASP, NIST, MITRE ATLAS",
];

const HomeHeroSection: React.FC = () => {
  const installText = useDecodedText("pip install deepteam", 2600);

  return (
    <section className={styles.hero}>
      <div className={styles.main}>
        <h1 className={styles.title}>
          The LLM{" "}
          <span className={styles.titleAccent}>Red Teaming</span> Framework
        </h1>

        <p className={styles.description}>
          Used by some of the world&apos;s leading AI companies, DeepTeam
          enables teams to red team any LLM application — surfacing
          vulnerabilities, jailbreaks, and agentic risks before attackers do.
        </p>

        <div className={styles.actions}>
          <PrimaryButton
            href="/docs/getting-started"
            shortkey="Enter"
            endIcon={<ArrowUpRight aria-hidden />}
          >
            Get Started
          </PrimaryButton>
          <SecondaryButton href="/guides/guide-agentic-ai-red-teaming">
            Explore Guides
          </SecondaryButton>
        </div>
      </div>

      {/* Hero footer — install command in a rectangular box with a
       *  rotating gradient border (deepteam-red sweep), plus the stat
       *  dial below. Shield experiment was reverted in favor of this
       *  simpler border-glow treatment. */}
      <div className={styles.heroFooter}>
        <div className={styles.installBlock}>
          <div className={styles.installCommand}>
            <span className={styles.installPrompt}>$</span>
            <code>{installText}</code>
          </div>
        </div>
        <div
          className={styles.statRoller}
          aria-label="DeepTeam by the numbers"
        >
          <span className={styles.statWindow}>
            <ul className={styles.statTrack}>
              <li>60+ adversarial attacks</li>
              <li>30+ vulnerability types</li>
              <li>6+ safety frameworks</li>
              <li aria-hidden="true">60+ adversarial attacks</li>
            </ul>
          </span>
        </div>
      </div>
      {/* Scrolling stats banner hidden for now — at deepteam's current
       *  traction, even the soft claims ("Used by leading security & AI
       *  teams") read as aspirational. The BANNER_ITEMS array stays
       *  defined above so reviving the ticker later is a one-block
       *  uncomment + tweak. */}
      {/*
      <PauseOffscreen
        className={styles.banner}
        aria-label="DeepTeam by the numbers"
      >
        <div className={styles.bannerTrack}>
          {[...BANNER_ITEMS, ...BANNER_ITEMS].map((item, i) => (
            <span
              key={i}
              className={styles.bannerItem}
              aria-hidden={i >= BANNER_ITEMS.length}
            >
              {item}
            </span>
          ))}
        </div>
      </PauseOffscreen>
      */}
      {/* Brand logo grid hidden for now — the 20-brand strip (Google,
       *  Uber, OpenAI, LEGO, etc.) was inherited verbatim from deepeval
       *  and overstates deepteam's current adoption. Keep the JSX +
       *  BRANDS array so it's a one-edit revival once we have honest
       *  customer logos to show. */}
      {/*
      <div className={styles.logoGrid} aria-label="Companies using DeepTeam">
        {BRANDS.map((brand) => {
          const DynamicLogo = DYNAMIC_LOGOS[brand.slug];
          return (
            <div key={brand.slug} className={styles.cell}>
              {DynamicLogo ? (
                <DynamicLogo
                  role="img"
                  aria-label={brand.name}
                  className={styles.logo}
                />
              ) : (
                <Image
                  src={`/icons/companies/${brand.slug}.svg`}
                  alt={brand.name}
                  width={120}
                  height={40}
                  className={styles.logo}
                />
              )}
            </div>
          );
        })}
      </div>
      */}
    </section>
  );
};

export default HomeHeroSection;
