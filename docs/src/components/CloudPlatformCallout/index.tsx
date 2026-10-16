import { Cloud, Compass } from "lucide-react";
import { PrimaryButton } from "@/src/components/Buttons";
import MascotPeek from "@/src/components/MascotPeek";
import { CAST } from "@/src/components/Mascot/_cast";
import styles from "./CloudPlatformCallout.module.scss";

const ENTERPRISE_HREF = "/enterprise";

const CloudPlatformCallout: React.FC = () => {
  return (
    <div className={styles.root}>
      <span className={styles.icon}>
        <Cloud aria-hidden="true" />
      </span>
      <span className={styles.content}>
        <span className={styles.title}>Collaborate in Confident Cloud</span>
        <span className={styles.body}>
          Manage AI security and governance across your organization, from
          scheduled red teaming to framework-aligned risk assessments.
        </span>
      </span>
      <span className={styles.cta}>
        <MascotPeek mascot={CAST.classic}>
          <PrimaryButton
            href={ENTERPRISE_HREF}
            startIcon={<Compass aria-hidden />}
            data-utm-content="toc_explore_enterprise"
          >
            Explore Enterprise
          </PrimaryButton>
        </MascotPeek>
      </span>
    </div>
  );
};

export default CloudPlatformCallout;
