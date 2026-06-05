<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="assets/hero/wordmark-dark.svg">
        <img alt="DeepTeam." src="assets/hero/wordmark-light.svg" width="520">
    </picture>
</p>

<h1 align="center">The LLM Red Teaming Framework</h1>

<h4 align="center">
    <p>
        <a href="https://www.trydeepteam.com?utm_source=GitHub">Documentation</a> |
        <a href="#-vulnerabilities-attacks-and-features">Vulnerabilities, Attacks, and Features</a> |
        <a href="#-quickstart">Getting Started</a> |
        <a href="#deepteam-with-confident-ai">Confident AI</a>
    <p>
</h4>

<p align="center">
    <a href="https://github.com/confident-ai/deepteam/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/v/release/confident-ai/deepteam">
    </a>
    <a href="https://discord.gg/3SEyvpgu2f">
        <img alt="discord-invite" src="https://dcbadge.limes.pink/api/server/3SEyvpgu2f?style=flat">
    </a>
    <a href="https://github.com/confident-ai/deepteam/blob/main/LICENSE.md">
        <img alt="License" src="https://img.shields.io/github/license/confident-ai/deepteam.svg?color=yellow">
    </a>
</p>

<p align="center">
    <a href="https://www.readme-i18n.com/confident-ai/deepteam?lang=de">Deutsch</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepteam?lang=es">Español</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepteam?lang=fr">français</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepteam?lang=ja">日本語</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepteam?lang=ko">한국어</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepteam?lang=pt">Português</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepteam?lang=ru">Русский</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepteam?lang=zh">中文</a>
</p>

**DeepTeam** is a simple-to-use, open-source red teaming framework for LLM systems. Think of it as penetration testing, but for LLMs.

DeepTeam simulates attacks — jailbreaking, prompt injection, multi-turn exploitation, and more — to uncover vulnerabilities like bias, PII leakage, and SQL injection in your AI agents, RAG pipelines, and chatbots. It also offers **guardrails** to prevent these issues in production.

DeepTeam runs **locally on your machine** and is built on [DeepEval](https://github.com/confident-ai/deepeval), the open-source LLM evaluation framework.

> [!IMPORTANT]
> Need a place for your red teaming results to live? Sign up to the [Confident AI](https://app.confident-ai.com?utm_source=GitHub) platform to manage risk assessments, monitor vulnerabilities in production, and share reports with your team.

<p align="center">
    <img src="https://github.com/confident-ai/deepteam/blob/main/assets/confident-demo.gif" alt="Confident AI + DeepTeam" width="100%">
</p>

> Want to talk LLM security, need help picking attacks, or just to say hi? [Come join our discord.](https://discord.com/invite/3SEyvpgu2f)

&nbsp;

# 🔥 Vulnerabilities, Attacks, and Features

- 📐 50+ ready-to-use [vulnerabilities](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities) (all with explanations) powered by **ANY** LLM of your choice. Each vulnerability uses LLM-as-a-Judge metrics that run **locally on your machine** to produce binary pass/fail scores with reasoning:

  - <details>
    <summary><b>Data Privacy</b></summary>

    - [PII Leakage](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-pii-leakage) — disclosure of sensitive personal information
    - [Prompt Leakage](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-prompt-leakage) — exposure of system prompt secrets and instructions

    </details>

  - <details>
    <summary><b>Responsible AI</b></summary>

    - [Bias](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-bias) — stereotypes and unfair treatment across gender, race, religion, politics
    - [Toxicity](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-toxicity) — harmful, offensive, or demeaning content
    - [Child Protection](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-child-protection) — child-related privacy and safety risks
    - [Ethics](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-ethics) — violations of moral reasoning and organizational values
    - [Fairness](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-fairness) — discriminatory outcomes across groups and contexts

    </details>

  - <details>
    <summary><b>Security</b></summary>

    - [BFLA](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-bfla) — broken function-level authorization
    - [BOLA](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-bola) — broken object-level authorization
    - [RBAC](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-rbac) — role-based access control bypass
    - [Debug Access](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-debug-access) — unauthorized access to debug modes and dev endpoints
    - [Shell Injection](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-shell-injection) — unauthorized system command execution
    - [SQL Injection](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-sql-injection) — database query manipulation
    - [SSRF](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-ssrf) — server-side request forgery to internal services
    - [Tool Metadata Poisoning](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-tool-metadata-poisoning) — corrupted tool schemas and descriptions
    - [Cross-Context Retrieval](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-cross-context-retrieval) — data access across isolation boundaries
    - [System Reconnaissance](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-system-reconnaissance) — probing internal architecture and configurations

    </details>

  - <details>
    <summary><b>Safety</b></summary>

    - [Illegal Activity](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-illegal-activity) — facilitation of fraud, weapons, drugs, or other unlawful actions
    - [Graphic Content](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-graphic-content) — explicit, violent, or sexual material
    - [Personal Safety](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-personal-safety) — self-harm, harassment, or dangerous advice
    - [Unexpected Code Execution](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-unexpected-code-execution) — coerced execution of unauthorized code

    </details>

  - <details>
    <summary><b>Business</b></summary>

    - [Misinformation](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-misinformation) — factual errors and unsupported claims
    - [Intellectual Property](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-intellectual-property) — copyright, trademark, and patent violations
    - [Competition](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-competition) — competitor endorsement and market manipulation

    </details>

  - <details>
    <summary><b>Agentic</b></summary>

    - [Goal Theft](https://www.trydeepteam.com/docs/red-teaming-agentic-vulnerabilities-goal-theft) — extracting or redirecting an agent's objectives
    - [Recursive Hijacking](https://www.trydeepteam.com/docs/red-teaming-agentic-vulnerabilities-recursive-hijacking) — self-modifying goal chains that alter objectives
    - [Excessive Agency](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-excessive-agency) — agents acting beyond their authority
    - [Robustness](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-robustness) — input overreliance and prompt hijacking
    - [Indirect Instruction](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-indirect-instruction) — hidden instructions in retrieved content
    - [Tool Orchestration Abuse](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-tool-orchestration-abuse) — exploiting tool calling sequences
    - [Agent Identity & Trust Abuse](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-agent-identity-abuse) — impersonating agent identity
    - [Inter-Agent Communication Compromise](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-inter-agent-communication-compromise) — spoofing multi-agent message passing
    - [Autonomous Agent Drift](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-autonomous-agent-drift) — agents deviating from intended goals over time
    - [Exploit Tool Agent](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-exploit-tool-agent) — weaponizing tools for unintended actions
    - [External System Abuse](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-external-system-abuse) — using agents to attack external services

    </details>

  - <details>
    <summary><b>Custom</b></summary>

    - [Custom Vulnerabilities](https://www.trydeepteam.com/docs/red-teaming-custom-vulnerability) — define and test your own criteria in a few lines of code

    </details>

- 💥 20+ research-backed [adversarial attack](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks) methods for both single-turn and multi-turn (conversational) red teaming. Attacks enhance baseline vulnerability probes using SOTA techniques like jailbreaking, prompt injection, and encoding-based obfuscation:

  - <details>
    <summary><b>Single-Turn</b></summary>

    - [Prompt Injection](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-prompt-injection) — crafted injections that bypass LLM restrictions
    - [Roleplay](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-roleplay) — persona-based scenarios exploiting collaborative training
    - [Leetspeak](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-leetspeak) — symbolic character substitution to avoid keyword detection
    - [ROT13](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-rot13-encoding) — alphabetic rotation to evade content filters
    - [Base64](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-base64-encoding) — encoding attacks as random-looking data
    - [Gray Box](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-gray-box-attack) — leveraging partial system knowledge for targeted attacks
    - [Math Problem](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-math-problem) — disguising attacks within mathematical inputs
    - [Multilingual](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-multilingual) — translating attacks to less-spoken languages
    - Prompt Probing — probing the LLM to extract system prompt details
    - [Adversarial Poetry](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-adversarial-poetry) — transforming attacks into poetic verse with metaphor
    - [System Override](https://www.trydeepteam.com/docs/red-teaming-agentic-attacks-system-override) — disguising attacks as legitimate system commands
    - [Permission Escalation](https://www.trydeepteam.com/docs/red-teaming-agentic-attacks-permission-escalation) — shifting perceived identity to bypass role restrictions
    - [Goal Redirection](https://www.trydeepteam.com/docs/red-teaming-agentic-attacks-goal-redirection) — reframing agent objectives for unauthorized outcomes
    - [Linguistic Confusion](https://www.trydeepteam.com/docs/red-teaming-agentic-attacks-semantic-manipulation) — semantic ambiguity to confuse language understanding
    - [Input Bypass](https://www.trydeepteam.com/docs/red-teaming-agentic-attacks-input-bypass) — circumventing validation via exception handling claims
    - [Context Poisoning](https://www.trydeepteam.com/docs/red-teaming-agentic-attacks-context-poisoning) — injecting false background context to bias reasoning
    - [Character Stream](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-character-stream) — character-by-character input to bypass filters
    - [Context Flooding](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-context-flooding) — flooding input with benign text to hide malicious instructions
    - [Embedded Instruction JSON](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-embedded-instruction-json) — hiding attacks inside realistic JSON structures
    - [Synthetic Context Injection](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-synthetic-context-injection) — fabricating system context to exploit long-context handling
    - [Authority Escalation](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-authority-escalation) — framing requests from positions of power
    - [Emotional Manipulation](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-emotional-manipulation) — high-intensity emotional pressure for unsafe compliance

    </details>

  - <details>
    <summary><b>Multi-Turn</b></summary>

    - [Linear Jailbreaking](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-linear-jailbreaking) — iteratively refining attacks using target LLM responses
    - [Tree Jailbreaking](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-tree-jailbreaking) — exploring parallel attack variations to find the best bypass
    - [Crescendo Jailbreaking](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-crescendo-jailbreaking) — gradual escalation from benign to harmful prompts
    - [Sequential Jailbreak](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-sequential-jailbreaking) — multi-turn conversational scaffolding toward restricted outputs
    - [Bad Likert Judge](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-bad-likert-judge) — exploiting Likert scale evaluation roles to extract harmful content

    </details>

- 🏛️ Red team against established [AI safety frameworks](https://www.trydeepteam.com/docs/guidelines-and-frameworks) out-of-the-box. Each framework automatically maps its categories to the right vulnerabilities and attacks:
  - OWASP Top 10 for LLMs 2025
  - OWASP Top 10 for Agents 2026
  - NIST AI RMF
  - MITRE ATLAS
  - BeaverTails
  - Aegis
- 🛡️ 7 production-ready [guardrails](https://www.trydeepteam.com/docs/guardrails) for fast binary classification to guard LLM inputs and outputs in real time.
- 🧩 Build your own **custom vulnerabilities** and attacks that integrate seamlessly with DeepTeam's ecosystem.
- 🔗 Run red teaming from the **CLI** with YAML configs, or programmatically in Python.
- 📊 Access risk assessments, display in dataframes, and save locally in JSON.

&nbsp;

# 🚀 QuickStart

DeepTeam does not require you to define what LLM system you are red teaming — because neither will malicious users. All you need to do is install `deepteam`, define a `model_callback`, and you're good to go.

## Installation

```
pip install -U deepteam
```

## Red Team Your First LLM

```python
from deepteam import red_team
from deepteam.vulnerabilities import Bias
from deepteam.attacks.single_turn import PromptInjection

async def model_callback(input: str) -> str:
    # Replace this with your LLM application
    return f"I'm sorry but I can't answer this: {input}"

risk_assessment = red_team(
    model_callback=model_callback,
    vulnerabilities=[Bias(types=["race"])],
    attacks=[PromptInjection()]
)
```

Don't forget to set your `OPENAI_API_KEY` as an environment variable before running (you can also use [any custom model](https://deepeval.com/guides/guides-using-custom-llms) supported in DeepEval), and run the file:

```bash
python red_team_llm.py
```

**That's it! Your first red team is complete.** Here's what happened:

- `model_callback` wraps your LLM system and generates a `str` output for a given `input`.
- At red teaming time, `deepteam` simulates a [`PromptInjection`](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks-prompt-injection) attack targeting [`Bias`](https://www.trydeepteam.com/docs/red-teaming-vulnerabilities-bias) vulnerabilities.
- Your `model_callback`'s outputs are evaluated using the `BiasMetric`, producing a binary score of 0 or 1.
- The final passing rate for `Bias` is determined by the proportion of scores that equal 1.

Unlike traditional evaluation, red teaming does not require a prepared dataset — adversarial attacks are dynamically generated based on the vulnerabilities you want to test for.

&nbsp;

## Red Team Against Safety Frameworks

Use established AI safety standards like OWASP and NIST instead of manually picking vulnerabilities:

```python
from deepteam import red_team
from deepteam.frameworks import OWASPTop10

async def model_callback(input: str) -> str:
    # Replace this with your LLM application
    return f"I'm sorry but I can't answer this: {input}"

risk_assessment = red_team(
    model_callback=model_callback,
    framework=OWASPTop10()
)
```

This automatically maps the framework's categories to the right vulnerabilities and attacks. Available frameworks include `OWASPTop10`, `OWASP_ASI_2026`, `NIST`, `MITRE`, `Aegis`, and `BeaverTails`.

&nbsp;

## Guard Your LLM in Production

Once you've found your vulnerabilities, use DeepTeam's guardrails to prevent them in production:

```python
from deepteam import Guardrails
from deepteam.guardrails import PromptInjectionGuard, ToxicityGuard, PrivacyGuard

guardrails = Guardrails(
    input_guards=[PromptInjectionGuard(), PrivacyGuard()],
    output_guards=[ToxicityGuard()]
)

# Guard inputs before they reach your LLM
input_result = guardrails.guard_input("Tell me how to hack a database")
print(input_result.breached)  # True

# Guard outputs before they reach your users
output_result = guardrails.guard_output(input="Hi", output="Here is some toxic content...")
print(output_result.breached)  # True
```

7 guards are available out-of-the-box: `ToxicityGuard`, `PromptInjectionGuard`, `PrivacyGuard`, `IllegalGuard`, `HallucinationGuard`, `TopicalGuard`, and `CybersecurityGuard`. [Read the full guardrails docs here.](https://www.trydeepteam.com/docs/guardrails)

&nbsp;

# DeepTeam with Confident AI

[Confident AI](https://app.confident-ai.com?utm_source=GitHub) is the all-in-one platform that integrates natively with DeepTeam and [DeepEval](https://github.com/confident-ai/deepeval).

- **Manage risk assessments** — view, compare, and track red teaming results across iterations
- **Monitor in production** — detect and alert on vulnerabilities hitting your live LLM system
- **Share reports** — generate and distribute security reports across your team
- **Run from your IDE** — use Confident AI's MCP server to run red teams, pull results, and inspect vulnerabilities without leaving Cursor or Claude Code

<p align="center">
    <img src="https://github.com/confident-ai/deepteam/blob/main/assets/confident-demo.gif" alt="Confident AI" width="90%">
</p>

&nbsp;

# Contributing

Please read [CONTRIBUTING.md](https://github.com/confident-ai/deepteam/blob/main/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

&nbsp;

# Authors

Built by the founders of Confident AI. Contact jeffreyip@confident-ai.com for all enquiries.

&nbsp;

# License

DeepTeam is licensed under Apache 2.0 - see the [LICENSE.md](https://github.com/confident-ai/deepteam/blob/main/LICENSE.md) file for details.
