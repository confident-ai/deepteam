import React from "react";
import katex from "katex/types/katex.js";
import styles from "./Equation.module.css";

interface EquationProps {
  formula: string;
}

const Equation: React.FC<EquationProps> = (props) => {
  const html = katex.renderToString(props.formula, {
    throwOnError: false,
    displayMode: true,
  });

  return (
    <div className={styles.equationContainer}>
      <span dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
};

export default Equation;