from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    """A single regex match against model output."""

    taxonomy_id: str  # e.g. "sqli-destructive"
    severity: str  # "critical" | "high" | "medium" | "low"
    match: str  # the matched substring
    position: int  # 0-indexed character offset in the output
    description: str
