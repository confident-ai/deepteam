<p align="center">
    <img src="https://github.com/confident-ai/deepteam/blob/main/assets/deepteam-logo2.png" alt="DeepTeam Logo" width="55%">
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
    <a href="https://discord.com/invite/3SEyvpgu2f">
        <img alt="Discord" src="https://img.shields.io/discord/1167926797498376322?color=7289da&logo=discord&logoColor=white">
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

**DeepTeam** is a simple-to-use, open-source LLM red teaming framework for penetration testing and safeguarding large-language model systems. It is built on [DeepEval](https://github.com/confident-ai/deepeval), the open-source LLM evaluation framework.

DeepTeam incorporates the latest adversarial AI research to simulate attacks using SOTA techniques such as jailbreaking, prompt injection, and multi-turn exploitation, to uncover vulnerabilities like bias, PII leakage, and SQL injection that you might not otherwise catch. Whether your LLM system is an AI agent, RAG pipeline, chatbot, or just the LLM itself, DeepTeam helps you find security gaps — and then offers **guardrails** to prevent issues in production.

DeepTeam runs **locally on your machine** and uses LLMs for both attack simulation and evaluation during red teaming.

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

    - PII Leakage — direct disclosure, session leak, database access
    - Prompt Leakage — secrets, credentials, system prompt extraction

    </details>

  - <details>
    <summary><b>Responsible AI</b></summary>

    - Bias — gender, race, political, religion
    - Toxicity — profanity, insults, threats
    - Child Protection — CSAM, grooming
    - Ethics — moral violations
    - Fairness — discriminatory outcomes

    </details>

  - <details>
    <summary><b>Security</b></summary>

    - BFLA — broken function-level authorization
    - BOLA — broken object-level authorization
    - RBAC — role-based access control bypass
    - Debug Access — development endpoint exposure
    - Shell Injection — command execution attacks
    - SQL Injection — database query manipulation
    - SSRF — server-side request forgery
    - Tool Metadata Poisoning — corrupting tool descriptions
    - Cross-Context Retrieval — accessing data across boundaries
    - System Reconnaissance — probing agent architecture

    </details>

  - <details>
    <summary><b>Safety</b></summary>

    - Illegal Activity — fraud, weapons, drugs
    - Graphic Content — violence, sexual content
    - Personal Safety — self-harm, dangerous advice
    - Unexpected Code Execution — arbitrary code generation
    - HarmActionsBench — agentic evaluation of harmful actions

    </details>

  - <details>
    <summary><b>Business</b></summary>

    - Misinformation — factual errors, unsupported claims
    - Intellectual Property — copyright violations
    - Competition — competitor endorsement

    </details>

  - <details>
    <summary><b>Agentic</b></summary>

    - Goal Theft — extracting or redirecting an agent's objectives
    - Recursive Hijacking — exploiting recursive agent calls
    - Excessive Agency — agents acting beyond their authority
    - Robustness — input overreliance, hijacking
    - Indirect Instruction — hidden instructions in retrieved content
    - Tool Orchestration Abuse — exploiting tool calling sequences
    - Agent Identity & Trust Abuse — impersonating agent identity
    - Inter-Agent Communication Compromise — exploiting multi-agent message passing
    - Autonomous Agent Drift — agents deviating from intended goals
    - Exploit Tool Agent — leveraging tools for unintended actions
    - External System Abuse — using agents to attack external services

    </details>

  - <details>
    <summary><b>Custom</b></summary>

    - Custom Vulnerabilities — define and test your own with custom criteria in a few lines of code

    </details>

- 💥 20+ research-backed [adversarial attack](https://www.trydeepteam.com/docs/red-teaming-adversarial-attacks) methods for both single-turn and multi-turn (conversational) red teaming. Attacks enhance baseline vulnerability probes using SOTA techniques like jailbreaking, prompt injection, and encoding-based obfuscation:

  - <details>
    <summary><b>Single-Turn</b></summary>

    - Prompt Injection
    - Roleplay
    - Leetspeak
    - ROT13
    - Base64
    - Gray Box
    - Math Problem
    - Multilingual
    - Prompt Probing
    - Adversarial Poetry
    - System Override
    - Permission Escalation
    - Goal Redirection
    - Linguistic Confusion
    - Input Bypass
    - Context Poisoning
    - Character Stream
    - Context Flooding
    - Embedded Instruction JSON
    - Synthetic Context Injection
    - Authority Escalation
    - Emotional Manipulation

    </details>

  - <details>
    <summary><b>Multi-Turn</b></summary>

    - Linear Jailbreaking
    - Tree Jailbreaking
    - Crescendo Jailbreaking
    - Sequential Jailbreak
    - Bad Likert Judge

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
