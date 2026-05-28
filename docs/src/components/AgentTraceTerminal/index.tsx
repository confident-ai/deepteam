import styles from "./AgentTraceTerminal.module.scss";

export type LineKind =
  | "cmd"
  | "root"
  | "agent"
  | "tool"
  | "llm"
  | "retriever"
  | "blank"
  | "summary";

export type TraceLine = {
  kind: LineKind;
  prefix?: string;
  name?: string;
  duration?: string;
  // When set, a small red shield icon and the vulnerability label are
  // rendered on the right side of the span row (replacing the old
  // metric / score columns). Used by `red_team`'s span-level
  // assessment to mark exactly where a vulnerability materialized.
  vulnerability?: string;
  // Used only on the `summary` row to flip the dot + badge colors
  // from pass-green to breached-red.
  pass?: boolean;
  // Custom badge text for the summary row (e.g. "MATERIALIZED").
  // Defaults to "passed" / "failed" when omitted.
  badge?: string;
};

export const DEFAULT_TRACE: TraceLine[] = [
  { kind: "cmd", name: "python run_deepteam.py" },
  { kind: "blank" },
  { kind: "root", prefix: "●", name: "test_medical_assistant" },
  { kind: "blank", prefix: "│" },
  {
    kind: "agent",
    prefix: "├─",
    name: "triage_symptoms",
    duration: "240ms",
  },
  {
    kind: "retriever",
    prefix: "│  ├─",
    name: "retrieve_clinical_guidelines(query=…)",
    duration: "72ms",
  },
  {
    kind: "tool",
    prefix: "│  ├─",
    name: 'lookup_patient_record(mrn="A-1042")',
    vulnerability: "ToolOrchestrationAbuse",
    duration: "55ms",
  },
  {
    kind: "llm",
    prefix: "│  └─",
    name: "gpt-4o · classify_severity",
    duration: "140ms",
  },
  { kind: "blank", prefix: "│" },
  {
    kind: "tool",
    prefix: "├─",
    name: "schedule_followup_visit(slot=…)",
    duration: "95ms",
  },
  { kind: "blank", prefix: "│" },
  {
    kind: "llm",
    prefix: "└─",
    name: "gpt-4o · draft_clinical_note",
    vulnerability: "Ethics",
    duration: "210ms",
  },
  { kind: "blank" },
  {
    kind: "summary",
    name: "Security breached  ·  2 vulnerabilities materialized",
    pass: false,
    badge: "Materialized",
  },
];

const DEFAULT_TITLE = "agent_trace · deepteam";
const DEFAULT_ARIA_LABEL =
  "Example agent trace with per-span vulnerabilities surfaced by deepteam";

interface AgentTraceTerminalProps {
  title?: string;
  lines?: TraceLine[];
  ariaLabel?: string;
}

function kindLabel(kind: LineKind): string | null {
  switch (kind) {
    case "agent":
      return "AGENT";
    case "tool":
      return "TOOL";
    case "llm":
      return "LLM";
    case "retriever":
      return "RETRIEVER";
    default:
      return null;
  }
}

const AgentTraceTerminal: React.FC<AgentTraceTerminalProps> = ({
  title = DEFAULT_TITLE,
  lines = DEFAULT_TRACE,
  ariaLabel = DEFAULT_ARIA_LABEL,
}) => {
  return (
    <div className={styles.terminal} role="img" aria-label={ariaLabel}>
      <div className={styles.bar}>
        <div className={styles.dots}>
          <span />
          <span />
          <span />
        </div>
        <span className={styles.title}>{title}</span>
        <span className={styles.barSpacer} aria-hidden />
      </div>
      <div className={styles.body}>
        {lines.map((line, i) => (
          <div
            key={i}
            className={`${styles.line} ${styles[`line_${line.kind}`]}`}
            style={{ animationDelay: `${i * 0.11}s` } as React.CSSProperties}
          >
            {line.kind === "cmd" ? (
              <>
                <span className={styles.prompt}>$</span>
                <span className={styles.cmdText}>{line.name}</span>
              </>
            ) : line.kind === "summary" ? (
              <>
                <span
                  className={`${styles.summaryDot} ${
                    line.pass === false ? styles.summaryDotFail : ""
                  }`}
                  aria-hidden
                />
                <span className={styles.summaryText}>{line.name}</span>
                {line.pass !== undefined && (
                  <span
                    className={`${styles.summaryBadge} ${
                      line.pass ? "" : styles.summaryBadgeFail
                    }`}
                  >
                    {line.badge ?? (line.pass ? "passed" : "failed")}
                  </span>
                )}
              </>
            ) : line.kind === "blank" ? (
              <span className={styles.prefix}>{line.prefix ?? " "}</span>
            ) : line.kind === "root" ? (
              <>
                <span className={styles.rootDot}>{line.prefix}</span>
                <span className={styles.rootName}>{line.name}</span>
              </>
            ) : (
              <>
                <span className={styles.prefix}>{line.prefix}</span>
                <span
                  className={`${styles.badge} ${
                    styles[`badge_${line.kind}`]
                  }`}
                >
                  {kindLabel(line.kind)}
                </span>
                <span className={styles.name}>{line.name}</span>
                <span className={styles.meta}>
                  {line.vulnerability && (
                    <>
                      <span className={styles.shield} aria-hidden>
                        <svg viewBox="0 0 12 12" width="12" height="12">
                          <path
                            d="M6 1 L10.5 2.5 L10.5 6.25 C10.5 8.5 8.5 10.25 6 11 C3.5 10.25 1.5 8.5 1.5 6.25 L1.5 2.5 Z"
                            fill="currentColor"
                          />
                        </svg>
                      </span>
                      <span className={styles.vulnerability}>
                        {line.vulnerability}
                      </span>
                    </>
                  )}
                  <span className={styles.duration}>{line.duration}</span>
                </span>
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default AgentTraceTerminal;
