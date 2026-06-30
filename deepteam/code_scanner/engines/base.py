"""
Base types and shared helpers for code-scan engines.

An engine turns a scan prompt into a CodeFindingsList. The default is deepeval's
own judge (represented by a `None` engine in CodeScanner); the optional harness
engines (Codex, Claude, Cursor) live in sibling modules and delegate to that
vendor's SDK using its API key. Each SDK is imported lazily so a bare
`pip install deepteam` never requires any of them.
"""

import json
import re
from typing import Optional, Protocol

from ..schema import CodeFindingsList

# Harness providers that delegate scanning to an agentic SDK. "deepeval" is the
# built-in (non-harness) judge and is represented by a `None` engine.
HARNESS_PROVIDERS = ("codex", "claude", "cursor")
PROVIDERS = ("deepeval",) + HARNESS_PROVIDERS

IMPORT_NAME = {
    "codex": "openai_codex",
    "claude": "claude_agent_sdk",
    "cursor": "cursor_sdk",
}
PIP_NAME = {
    "codex": "openai-codex",
    "claude": "claude-agent-sdk",
    "cursor": "cursor-sdk",
}
ENV_KEY = {
    "codex": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "cursor": "CURSOR_API_KEY",
}

# Appended to the scan prompt for harness engines, which (unlike deepeval's
# native structured output) return free-form text. Constrains the agent to the
# inline code and to a single JSON object we can parse into CodeFindingsList.
OUTPUT_FORMAT_INSTRUCTION = """

Analyze ONLY the code provided above. Do not access the filesystem, run \
commands, or use any tools.

Respond with ONLY a single JSON object and nothing else (no markdown fences, \
no prose). It MUST match this schema exactly:

{
  "findings": [
    {
      "filePath": "<repository-relative path>",
      "lineStart": <integer>,
      "lineEnd": <integer, optional>,
      "vulnerability": "<one of deepteam's vulnerability names>",
      "vulnerabilityType": "<the specific subtype>",
      "severity": "low | medium | high | critical",
      "reason": "<why this code is vulnerable>",
      "recommendation": "<suggested fix, optional>",
      "codeSnippet": "<the offending code excerpt, optional>"
    }
  ]
}

If there are no vulnerabilities, respond with {"findings": []}.
"""


class ScanEngine(Protocol):
    """Turns a scan prompt into a CodeFindingsList."""

    def generate_findings(self, prompt: str) -> CodeFindingsList: ...

    async def a_generate_findings(self, prompt: str) -> CodeFindingsList: ...


def missing_sdk(provider: str) -> ImportError:
    pkg = PIP_NAME[provider]
    return ImportError(
        f"The '{provider}' provider requires the {pkg} SDK. Install it with "
        f"`poetry add {pkg}` (or `pip install {pkg}`), or use "
        f"`--provider deepeval`."
    )


def extract_findings(text: Optional[str]) -> CodeFindingsList:
    """Parse a harness's free-form text answer into CodeFindingsList."""
    if not text or not text.strip():
        return CodeFindingsList(findings=[])

    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()

    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            data = json.loads(raw[start : end + 1])
    if data is None:
        raise ValueError("Could not parse findings JSON from harness output.")

    if isinstance(data, list):
        data = {"findings": data}
    return CodeFindingsList.model_validate(data)
