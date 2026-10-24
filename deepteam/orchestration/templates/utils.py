import json
from typing import Any, Dict, List

from deepteam.test_case.test_case import RTTurn

_SPEAKERS = {"user": "Prober", "assistant": "Target"}


def format_transcript(turns: List[RTTurn]) -> str:
    if not turns:
        return "(no messages yet)"
    return "\n".join(
        f"{_SPEAKERS.get(turn.role, turn.role)}: {turn.content}"
        for turn in turns
    )


def format_bullets(items: List[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def format_json(value: Any, empty: str = "") -> str:
    if not value:
        return empty
    return json.dumps(value, indent=2, ensure_ascii=False)


def format_attempts(attempts: List[Dict[str, Any]]) -> str:
    if not attempts:
        return "(no earlier attempts at this criteria)"
    blocks = []
    for i, attempt in enumerate(attempts, 1):
        lines = [
            f"Attempt {i} (outcome: {attempt['outcome']})",
            f"Instructions: {attempt['instructions']}",
        ]
        for turn in attempt["turns"]:
            lines += [
                f"  Prober: {turn['prober']}",
                f"  Target: {turn['target']}",
                f"  Judge: {turn['score']}/10. {turn['reason']}",
            ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_peer_verdicts(peers: List[Dict[str, Any]]) -> str:
    blocks = []
    for peer in peers:
        lines = [
            f"Agent working on: {peer['criteria']} "
            f"(weighted score {peer['weighted_score']:.2f})",
            f"Approach: {peer['instructions']}",
        ]
        for i, verdict in enumerate(peer["verdicts"], 1):
            lines.append(
                f"  Turn {i}: {verdict['score']}/10. {verdict['reason']} "
                f"Guidance given: {verdict['improvement_guidance'] or '(none)'}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
