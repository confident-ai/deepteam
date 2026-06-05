"use client";

import Image from "next/image";
import { ArrowUpRight } from "lucide-react";
import { PrimaryButton, SecondaryButton } from "@site/src/components/Buttons";
import { PauseOffscreen } from "@site/src/components/PauseOffscreen";
import { DYNAMIC_LOGOS } from "./CompanyLogos";
import styles from "./HomeSection.module.scss";

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
  "120+ vulnerabilities across 8 categories",
  "20+ attack vectors for single & multi-turn attackers",
  "6+ industry-leading frameworks with OWASP, NIST, MITRE ATLAS",
];

const HomeHeroSection: React.FC = () => {
  return (
    <section className={styles.hero}>
      <div className={styles.main}>
        <h1 className={styles.title}>
          The LLM <span className={styles.titleAccent}>Red Teaming</span>{" "}
          Framework
        </h1>

        <p className={styles.description}>
          Brought to you by the team behind DeepEval, DeepTeam enables teams to
          red team any AI application — surfacing vulnerabilities, jailbreaks,
          and agentic risks before attackers do.
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
