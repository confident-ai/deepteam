import { Check, X } from "lucide-react";
import styles from "./EnterpriseComparisonTable.module.scss";

const ROWS = [
  {
    confident: "Shared risk assessments",
    deepteam: "Results live in local files",
  },
  {
    confident: "No-code red teaming",
    deepteam: "Python scripts and the CLI",
  },
  {
    confident: "Scheduled, recurring assessments",
    deepteam: "One-off runs you trigger yourself",
  },
  {
    confident: "Custom frameworks and risk categories",
    deepteam: "Frameworks configured in code",
  },
  {
    confident: "Org-wide risk profile with CVSS scores",
    deepteam: "Per-run scores in the terminal",
  },
  {
    confident: "Test cases to review and triage",
    deepteam: "Raw output to sift through",
  },
  {
    confident: "Red team any AI app over HTTP",
    deepteam: "Wrap your app in a Python callback",
  },
  {
    confident: "Enterprise controls",
    deepteam: "Single-user by design",
  },
];

const EnterpriseComparisonTable: React.FC = () => {
  return (
    <div
      className={styles.table}
      role="table"
      aria-label="DeepTeam and Confident AI comparison"
    >
      <div className={styles.header} role="row">
        <div role="columnheader">Confident AI</div>
        <div role="columnheader">DeepTeam</div>
      </div>
      <div className={styles.body}>
        {ROWS.map((row) => (
          <div key={row.confident} className={styles.row} role="row">
            <div className={styles.item} role="cell">
              <span className={styles.icon} data-tone="positive" aria-hidden>
                <Check />
              </span>
              <strong>{row.confident}</strong>
            </div>
            <div className={styles.item} role="cell">
              <span className={styles.icon} data-tone="negative" aria-hidden>
                <X />
              </span>
              <strong>{row.deepteam}</strong>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default EnterpriseComparisonTable;
