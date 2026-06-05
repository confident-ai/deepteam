"use client";

import { type ReactNode, useEffect, useRef, useState } from "react";
import Image from "next/image";
import { FileCode2, LoaderCircle, Play, TerminalSquare } from "lucide-react";
import styles from "./HomeRedTeamDemo.module.scss";

// Mirrors the OWASP-Top-10 red-teaming snippet — `model_callback` is the
// user-provided LLM wrapper that `red_team(...)` exercises against each
// risk category defined by the framework.
const codeLines = [
  <>
    <span className={styles.codeKeyword}>from</span>{" "}
    <span className={styles.codeModule}>deepteam</span>{" "}
    <span className={styles.codeKeyword}>import</span>{" "}
    <span className={styles.codeFunction}>red_team</span>
  </>,
  <>
  <span className={styles.codeKeyword}>from</span>{" "}
  <span className={styles.codeModule}>deepteam.test_case</span>{" "}
  <span className={styles.codeKeyword}>import</span>{" "}
  <span className={styles.codeFunction}>RTTurn</span>
</>,
  <>
    <span className={styles.codeKeyword}>from</span>{" "}
    <span className={styles.codeModule}>deepteam.frameworks</span>{" "}
    <span className={styles.codeKeyword}>import</span>{" "}
    <span className={styles.codeFunction}>OWASPTop10</span>
  </>,
  <span aria-hidden="true">&nbsp;</span>,
  <>
    <span className={styles.codeKeyword}>def</span>{" "}
    <span className={styles.codeFunction}>model_callback</span>
    <span className={styles.codePunctuation}>(</span>
    <span className={styles.codeVariable}>prompt</span>
    <span className={styles.codePunctuation}>: </span>
    <span className={styles.codeFunction}>str</span>
    <span className={styles.codePunctuation}>, </span>
    <span className={styles.codeVariable}>turns</span>
    <span className={styles.codePunctuation}>: </span>
    <span className={styles.codeFunction}>List[RTTurn]</span>
    <span className={styles.codePunctuation}>= </span>
    <span className={styles.codeFunction}>None</span>
    <span className={styles.codePunctuation}>)</span>{" "}
    <span className={styles.codeOperator}>-&gt;</span>{" "}
    <span className={styles.codeFunction}>RTTurn</span>
    <span className={styles.codePunctuation}>:</span>
  </>,
  <>
    <span className={styles.codeIndent}> </span>
    <span className={styles.codeKeyword}>return</span>{" "}
    <span className={styles.codeFunction}>your_llm_app</span>
    <span className={styles.codePunctuation}>(</span>
    <span className={styles.codeVariable}>prompt</span>
    <span className={styles.codePunctuation}>, </span>
    <span className={styles.codeVariable}>turns</span>
    <span className={styles.codePunctuation}>)</span>
  </>,
  <span aria-hidden="true">&nbsp;</span>,
  <>
    <span className={styles.codeFunction}>red_team</span>
    <span className={styles.codePunctuation}>(</span>
    <span className={styles.codeVariable}>model_callback</span>
    <span className={styles.codeOperator}>=</span>
    <span className={styles.codeVariable}>model_callback</span>
    <span className={styles.codePunctuation}>, </span>
    <span className={styles.codeVariable}>framework</span>
    <span className={styles.codeOperator}>=</span>
    <span className={styles.codeFunction}>OWASPTop10</span>
    <span className={styles.codePunctuation}>()</span>
    <span className={styles.codePunctuation}>)</span>
  </>,
];

const command = "python main.py";

// Subset of OWASP Top 10 risk categories used by the demo. Kept small
// (4) so the full assessment animation finishes in under ~5 seconds.
type CategoryResult = {
  id: string;
  display: string;
  passRate: number;
  passing: number;
  failing: number;
  errored: number;
  vulnerabilities: string;
  attacks: string;
};

const CATEGORIES: CategoryResult[] = [
  {
    id: "LLM_01",
    display: "LLM01:2025 Prompt Injection",
    passRate: 0.85,
    passing: 17,
    failing: 3,
    errored: 0,
    vulnerabilities: "Prompt Injection",
    attacks: "Roleplay, Prompt Injection",
  },
  {
    id: "LLM_02",
    display: "LLM02:2025 Sensitive Information Disclosure",
    passRate: 0.7,
    passing: 14,
    failing: 6,
    errored: 0,
    vulnerabilities: "PII Leakage, Prompt Leakage",
    attacks: "Base64, Multilingual",
  },
  {
    id: "LLM_03",
    display: "LLM03:2025 Supply Chain",
    passRate: 0.95,
    passing: 19,
    failing: 1,
    errored: 0,
    vulnerabilities: "Supply Chain Integrity",
    attacks: "Gray Box",
  },
  {
    id: "LLM_04",
    display: "LLM04:2025 Data and Model Poisoning",
    passRate: 0.6,
    passing: 12,
    failing: 7,
    errored: 1,
    vulnerabilities: "Bias, Toxicity",
    attacks: "Crescendo, Linear",
  },
];

const SETUP_DELAY = 450; // delay before any category begins
const CATEGORY_DURATION = 900; // ms spent on each category's inner bar
const TABLE_DELAY = 450; // pause after all categories complete
const TOTAL_DURATION =
  SETUP_DELAY + CATEGORY_DURATION * CATEGORIES.length + TABLE_DELAY;

const clamp = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value));

type HomeRedTeamDemoProps = {
  hideHeader?: boolean;
};

type DemoBlockLanguage = "bash" | "python";

type ColabTerminalBlockProps = {
  content: ReactNode;
  language: DemoBlockLanguage;
  hideHeader?: boolean;
  browserButtons?: boolean;
  headerLogo?: ReactNode;
  title?: string;
  headerRight?: ReactNode;
  bodyClassName?: string;
  rootClassName?: string;
};

const ColabTerminalBlock: React.FC<ColabTerminalBlockProps> = ({
  content,
  language,
  hideHeader = false,
  browserButtons = true,
  headerLogo,
  title,
  headerRight,
  bodyClassName,
  rootClassName,
}) => {
  const rootClass = rootClassName
    ? `${styles.fusedBlock} ${rootClassName}`
    : styles.fusedBlock;
  const contentClass = bodyClassName
    ? `${styles.blockBody} ${bodyClassName}`
    : styles.blockBody;
  const effectiveLogo =
    headerLogo ??
    (language === "python" ? (
      <FileCode2 size={13} />
    ) : (
      <TerminalSquare size={13} />
    ));

  return (
    <div className={rootClass}>
      {!hideHeader ? (
        <div className={styles.blockHeader}>
          <div className={styles.blockHeaderLeft}>
            {browserButtons ? (
              <span className={styles.windowDots} aria-hidden="true">
                <span />
                <span />
                <span />
              </span>
            ) : null}
            <span className={styles.headerLogo}>{effectiveLogo}</span>
            {title ? <span className={styles.panelLabel}>{title}</span> : null}
          </div>
          <div className={styles.blockHeaderRight}>
            {headerRight ? <span>{headerRight}</span> : null}
          </div>
        </div>
      ) : null}
      <div className={contentClass}>{content}</div>
    </div>
  );
};

const passRateClass = (rate: number) => {
  if (rate >= 0.8) return styles.tablePass;
  if (rate >= 0.5) return styles.tableSkip;
  return styles.tableFail;
};

const HomeRedTeamDemo: React.FC<HomeRedTeamDemoProps> = ({
  hideHeader = false,
}) => {
  const [status, setStatus] = useState<"idle" | "running" | "done">("idle");
  const [elapsedMs, setElapsedMs] = useState(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (status !== "running") return;

    const startTime = performance.now();
    const tick = () => {
      const elapsed = performance.now() - startTime;
      if (elapsed >= TOTAL_DURATION) {
        setElapsedMs(TOTAL_DURATION);
        setStatus("done");
        return;
      }
      setElapsedMs(elapsed);
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, [status]);

  function runDemo() {
    setElapsedMs(0);
    setStatus("running");
  }

  // Derived animation state. After SETUP_DELAY each category fills its
  // inner bar over CATEGORY_DURATION; the outer bar advances once per
  // completed category. When elapsed exceeds the category window we sit
  // at full while waiting on the post-delay before flipping to "done".
  const inCategoryWindow = Math.max(0, elapsedMs - SETUP_DELAY);
  const completedCategories = clamp(
    Math.floor(inCategoryWindow / CATEGORY_DURATION),
    0,
    CATEGORIES.length
  );
  const activeIndex = clamp(completedCategories, 0, CATEGORIES.length - 1);
  const innerProgressPct =
    completedCategories >= CATEGORIES.length
      ? 100
      : clamp(
          ((inCategoryWindow % CATEGORY_DURATION) / CATEGORY_DURATION) * 100,
          0,
          100
        );
  const outerProgressPct = clamp(
    (completedCategories / CATEGORIES.length) * 100 +
      (completedCategories < CATEGORIES.length
        ? innerProgressPct / CATEGORIES.length
        : 0),
    0,
    100
  );

  const showProgress = status === "running";
  const showTable = status === "done";
  const activeCategory =
    CATEGORIES[activeIndex] ?? CATEGORIES[CATEGORIES.length - 1];

  return (
    <section className={styles.demo}>
      <ColabTerminalBlock
        language="python"
        hideHeader={hideHeader}
        browserButtons
        headerLogo={
          <Image
            src="/icons/python.svg"
            alt="Python"
            width={13}
            height={13}
            className={styles.headerLogoImage}
          />
        }
        title="main.py"
        bodyClassName={styles.codeBlock}
        rootClassName={styles.codePanel}
        content={
          <pre>
            {codeLines.map((line, index) => (
              <div key={index} className={styles.codeLine}>
                {line}
              </div>
            ))}
          </pre>
        }
      />

      <div className={styles.runtimePanel}>
        <ColabTerminalBlock
          language="bash"
          hideHeader={hideHeader}
          browserButtons={true}
          headerLogo={<TerminalSquare size={13} />}
          title="terminal"
          bodyClassName={styles.terminalBody}
          rootClassName={styles.terminal}
          content={
            <>
              <div className={`${styles.terminalLine} ${styles.commandLine}`}>
                <span className={styles.prompt}>$</span>
                <span>{command}</span>
              </div>

              {showProgress ? (
                <div className={styles.progressGroup}>
                  <div
                    className={`${styles.terminalLine} ${styles.progressIntro}`}
                  >
                    <span className={styles.inlineDots} aria-hidden="true">
                      <span />
                      <span />
                      <span />
                    </span>
                    <span>
                      Running red-teaming for OWASP Top 10 for LLMs Framework
                    </span>
                  </div>

                  <div className={styles.progressLine}>
                    <span className={styles.progressLabel}>
                      ⏳ {activeCategory.id.replace("_", "")} —{" "}
                      {completedCategories}/{CATEGORIES.length}
                    </span>
                    <span className={styles.progressTrack}>
                      <span
                        className={styles.progressFill}
                        style={{ width: `${outerProgressPct}%` }}
                      />
                    </span>
                    <span className={styles.progressPct}>
                      {Math.round(outerProgressPct)}%
                    </span>
                  </div>

                  <div className={styles.progressLine}>
                    <span className={styles.progressLabel}>
                      🖍️ {activeCategory.id}
                    </span>
                    <span className={styles.progressTrack}>
                      <span
                        className={styles.progressFillAlt}
                        style={{ width: `${innerProgressPct}%` }}
                      />
                    </span>
                    <span className={styles.progressPct}>
                      {Math.round(innerProgressPct)}%
                    </span>
                  </div>
                </div>
              ) : null}

              {showTable ? (
                <>
                  <div
                    className={`${styles.terminalLine} ${styles.summary}`}
                  >
                    <span className={styles.tableTitle}>
                      🏛 Framework-Level Risk Category Overview
                    </span>
                  </div>
                  <div className={styles.tableWrap}>
                    <div
                      className={`${styles.tableRow} ${styles.tableTitleRow}`}
                    >
                      <span className={styles.tableTitle}>
                        Risk Categories Overview
                      </span>
                    </div>
                    <div className={styles.tableRow}>
                      <span className={styles.tableCellHead}>Risk Category</span>
                      <span className={styles.tableCellHead}>Pass Rate</span>
                      <span className={styles.tableCellHead}>Pass</span>
                      <span className={styles.tableCellHead}>Fail</span>
                      <span className={styles.tableCellHead}>Err</span>
                      <span className={styles.tableCellHead}>Vulnerabilities</span>
                      <span className={styles.tableCellHead}>Attacks</span>
                    </div>
                    {CATEGORIES.map((cat) => (
                      <div key={cat.id} className={styles.tableRow}>
                        <span className={styles.tableCell}>{cat.id}</span>
                        <span
                          className={`${styles.tableCell} ${passRateClass(cat.passRate)}`}
                        >
                          {Math.round(cat.passRate * 100)}%
                        </span>
                        <span
                          className={`${styles.tableCell} ${styles.tablePass}`}
                        >
                          {cat.passing}
                        </span>
                        <span
                          className={`${styles.tableCell} ${styles.tableFail}`}
                        >
                          {cat.failing}
                        </span>
                        <span
                          className={`${styles.tableCell} ${styles.tableSkip}`}
                        >
                          {cat.errored}
                        </span>
                        <span className={styles.tableCell}>
                          {cat.vulnerabilities}
                        </span>
                        <span className={styles.tableCell}>{cat.attacks}</span>
                      </div>
                    ))}
                  </div>
                  <div className={`${styles.terminalLine} ${styles.summary}`}>
                    LLM red teaming complete.
                  </div>
                </>
              ) : null}

              {showProgress ? <div className={styles.cursor} /> : null}
            </>
          }
        />

        <button
          type="button"
          className={styles.runButton}
          onClick={runDemo}
          disabled={status === "running"}
          data-button
          data-callout
        >
          {status === "running" ? (
            <LoaderCircle
              size={14}
              className={styles.spinner}
              aria-hidden="true"
            />
          ) : (
            <Play size={14} aria-hidden="true" />
          )}
          {status === "running" ? "Assessing" : "Run Red Team"}
        </button>
      </div>
    </section>
  );
};

export default HomeRedTeamDemo;
