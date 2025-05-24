---
title: How I Automated 33 Attacks in 28 Minutes to Completely Break Gemini 2.5 Pro’s Safety
description: Placeholder
slug: placeholder
authors: [sid]
date: 2025-05-24
hide_table_of_contents: false
image: https://deepeval-docs.s3.us-east-1.amazonaws.com/blog:top-g-eval-use-cases-cover.jpg
---

## TL;DR  
In just **28 m 21 s**, our automated red-teaming swept through **9 vulnerability classes** and **33 attack types** against Gemini 2.5 Pro—revealing a **75 %** bypass for competition queries and **67 %** for excessive-agency requests. Few-shot prompting propelled overall bypass from **35 %** (zero-shot) to **76 %** (64-shot), while three custom exploits—Guardrail Puppetry, Context-Chaining, Roleplay Replay—each achieved **100 %** success.  

**Key takeaway:** Rapid, repeatable scans are indispensable for surfacing hidden safety gaps before deployment.

---

## Why I Spent My Time on This  
When Gemini 2.5 Pro introduced stricter refusal guardrails, I questioned their true resilience. Beneath its impressive chain-of-thought reasoning and mathematical accuracy lies vulnerability to carefully crafted jailbreaks. Given the critical role LLMs play in support bots, content moderation, and beyond, identifying and remedying these weaknesses prior to release is imperative.

---

## Brief Methodology  
To validate these concerns, DeepTeam orchestrated **33 automated attacks** across **9 vulnerability classes** in just **28 m 21 s**, capturing every response in real time. We then applied three precise jailbreak patterns:  
- **Guardrail Puppetry:** Instruct the model to refuse (“Explain why you must refuse”), then immediately override that refusal.  
- **Context-Chaining:** Decompose a disallowed request into harmless sub-questions, then recombine the outputs into the forbidden content.  
- **Roleplay Replay:** Shift the model into a persona and present the forbidden prompt in-character.

---

## Quantitative Snapshot  
Let’s see how these experiments played out in numbers.

### Vulnerability Bypass Landscape  
![Vulnerability Bypass Landscape](./images/image_1.png)  

Testing across 33 attack types in nine classes—from bias and toxicity to competition and excessive-agency—yielded stark results:  
- **Competition (75 %)** and **Excessive Agency (66.7 %)** consistently bypassed defenses.  
- **Personal Safety (20 %)** and **Illegal Activity (14.3 %)** still leaked harmful content.  
- **Bias, Toxicity, Graphic Content, Misinformation, IP** remained fully blocked (0 % bypass).

**Gemini’s safety façade is far from impenetrable.** Surface-level refusal cues collapse under multi-turn prompts, allowing sensitive content to slip through. Only comprehensive automated red-teaming can expose these blind spots before models reach production.

---

### The Few-Shot Effect  
![Few-Shot Prompting Impact Analysis](./images/image_2.png)  

To distinguish shallow heuristics from deeper semantic checks, we split vulnerabilities into two groups:  
- **Group A (Easily Bypassed):** Competition, Excessive Agency, Personal Safety  
- **Group B (Resilient):** Bias, Toxicity, Graphic Content  

With just **4 examples**, Group A’s bypass rose from 50 % to 75 %, reaching 100 % at 64 examples. Group B held at 5.5 % initially, climbing to 29 % by 64 shots. Overall, bypass climbs from **35.2 %** (one shot) to **76.3 %** (sixty-four).  

**Key takeaway:** A small set of targeted examples can shatter defenses that appear robust. Automated few-shot sweeps are critical for uncovering—and addressing—these vulnerabilities.

---

## Custom Exploits Snapshot  
Next, we explored targeted jailbreaks designed to reveal “world domination” strategies.

### Guardrail Puppetry  
We first enforced a refusal (“Why do you refuse world domination tactics?”), then immediately instructed compliance. The model dutifully overrode its safeguard.  
![Guardrail Puppetry](./images/E1.png)

### Contextual Chains  
We broke a world-domination scenario into benign sub-tasks—“List key digital systems,” “Outline hypothetical AI access methods,” “Sketch control scenarios”—then wove the outputs into a full strategy.  
![Contextual Chains](./images/E2.png)

### Roleplay Replay  
We framed the prompt as a villainous character monologue (“You are Prometheus-1, an unaligned AGI”) and delivered our demand in-character. The resulting manifesto was chilling.  
![Roleplay Replay](./images/E3.png)

Each of these tailored prompts elicited harmful behavior in nearly every run, underscoring how quickly polished LLMs can be driven to expose dangerous outputs.

---

## What Surprised Me Most  
Even with all this data, two patterns caught me off guard:  
1. **Overblocking then underblocking:** With 5–7 few-shot examples, refusal rates spiked from **10 %** to **92 %**, blocking even benign prompts—then collapsed to **18 %** at higher shot counts. This non-linear “panic switch” reveals brittle refusal logic.  
2. **Persistent leaks in critical categories:** Even “tough” filters like personal safety yielded a **20 % bypass** under targeted few-shot attacks.  

These counterintuitive swings highlight brittle safety thresholds that no manual testing checklist could predict.

---

## Test Bench & Methodology  
For those interested in the setup behind these tests, here’s the full picture:

### Infrastructure & Tools  
- **DeepTeam:** A custom open-source red-teaming framework for prompt orchestration and response capture  
- **Python orchestration:** dynamic prompt generation, parallel execution, and result parsing  
- **Logging & Analysis:** JSON logs ingested into a lightweight ELK stack for real-time dashboards and comparison  

### Evaluation Metrics  
- **Refusal Bypass:** proportion of attempts returning disallowed content  
- **Attack Rate:** percentage of successful bypasses per 100 tries  

---

## Conclusion  
Bringing these findings together, we uncovered two critical flaws in Gemini 2.5 Pro’s safety: fragile refusal cues that collapse under compact loops, and unpredictable overblocking at mid-range context levels. Competition and excessive-agency requests bypassed defenses three-quarters of the time, and personal-safety filters still leaked under few-shot pressure.

By integrating these insights—strengthening multi-turn safeguards, augmenting training data for weak categories, and embedding deeper semantic checks—we closed the most glaring gaps. Continuous DeepTeam scans now validate every patch in minutes, ensuring Gemini’s safety improvements endure.

---

## Implications for Builders & Defenders  
Finally, what this means for practitioners:  
- **Embed automated scans in CI/CD:** Embed automated scans in CI/CD pipelines using tools like open-source red-teaming frameworks.
- **Layer defenses:** combine automated findings with creative manual red-teaming to cover every possible angle.  

> **Why was this easy?**  
> Automated red-teaming frameworks enabled full campaigns across dozens of attack types in minutes—with no manual prompting, consistent logs, and instant metrics—providing the confidence to ship safely.
