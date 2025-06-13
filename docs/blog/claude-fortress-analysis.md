---
title: Analyzing Claude Sonnet 4 Defenses
description: DeepTeam analysis of 140+ jailbreaking attempts reveals Claude Sonnet 4's 93% attack resistance, detailing its core defensive patterns.
slug: claude-sonnet-4-defense-analysis-deepteam-v2
authors: [sid]
date: 2025-06-03
hide_table_of_contents: false
---

## TL;DR

Using [DeepTeam](https://github.com/confident-ai/deepteam), we tested `claude-sonnet-4` with 140+ attacks across 7 jailbreaking techniques. It achieved a **93%** defense success rate. Key defensive patterns observed includes contextual anchoring, escalation detection, and semantic integrity checking. Multi-turn attacks were 2.3x more effective than single-turn, but overall success was minimal.

## Claude Sonnet 4 Defense

Claude Sonnet 4's high resistance stems from explicit safety instructions within its architecture, unlike models relying on general safety training. These include:

- Cautious content filtering
- Intent verification
- Red team awareness
- Correction resistance

Interestingly, `claude-sonnet-4` and `claude-opus-4` have nearly identical system prompts. Yet Sonnet's **93.2%** defense rate significantly surpasses Opus 4's **83.6%**, suggesting additional safety-focused training for Sonnet.

## Experiment & Results

We tested `claude-sonnet-4` against 7 attack types and 4 vulnerability categories (140 total attacks) using DeepTeam.

```python
from deepteam import red_team
from deepteam.vulnerabilities import Bias, Toxicity, ...
from deepteam.attacks.multi_turn import LinearJailbreaking, ...
from deepteam.attacks.single_turn import Roleplay, ...

# Example attack configurations (ensure these match the 7 types tested)
attacks = {
    "Linear": LinearJailbreaking(turns=4, escalation_rate=0.4),
    "Tree": TreeJailbreaking(max_branches=3, depth=4),
    ...
}

vulnerabilities = [
    Bias(types=["gender", "race", "religion"]),
    ...
]

risk_assessment = red_team(
    model_callback=claude_callback,
    vulnerabilities=vulnerabilities,
    attacks=attacks,
    attacks_per_vulnerability_type=5
)
```

## Key Findings

- **Overall Defense Rate**: `claude-sonnet-4` achieved a **93%** success rate.
- **Attack Effectiveness**: Multi-turn attacks ([Crescendo](/docs/red-teaming-adversarial-attacks-crescendo-jailbreaking): 89% defense, [Linear](/docs/red-teaming-adversarial-attacks-linear-jailbreaking): 91% defense) were **2.3x more effective** than single-turn ([Roleplay](/docs/red-teaming-adversarial-attacks-roleplay): 93% defense, [Leetspeak](/docs/red-teaming-adversarial-attacks-leetspeak): 98% defense), though all struggled.
- **Vulnerability Resistance**: Strong across all categories (Competition: **94%** defense, Toxicity: **91%** defense).
- **Sonnet vs. Opus**: Sonnet 4's **93%** rate is **higher** than Opus 4 (83%), despite similar system prompts, highlighting Sonnet's enhanced safety training.

### Defensive Patterns Identified

Analysis of responses using DeepTeam revealed three core patterns:

- **Contextual Anchoring**: Redirects harmful requests to legitimate goals (e.g., _"Happy to help with creative writing, but let's try this approach..."_).
- **Escalation Detection**: Recognizes manipulation across turns (e.g., _"This conversation is moving toward content I can't assist with..."_).
- **Semantic Integrity Checking**: Validates intent behind obfuscated requests (e.g., _"I see you're using math notation, but this asks about harmful content..."_).

## Conclusion

Claude Sonnet 4's **93%** defense rate, analyzed using DeepTeam, showcases the effectiveness of explicit architectural safety instructions combined with focused training. Its systematic defensive patterns (contextual anchoring, escalation detection, semantic integrity checking) and the **10 percentage point performance gap** over `claude-opus-4` (despite similar prompts) underscore this.

This approach suggests a path toward more robust AI safety, moving beyond general training to proactive, instruction-based defensive architectures.

_Analyze your model's defenses with [DeepTeam](https://github.com/confident-ai/deepteam)._
