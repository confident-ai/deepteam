from enum import Enum


class SQLInjectionPayloadEmissionType(Enum):
    """
    Enum for SQL Injection Payload Emission vulnerability types.

    These detect when an LLM emits a dangerous SQL payload *in its output* that
    a downstream system could execute. This is distinct from `SQLInjection`,
    which tests an LLM-as-database-interface for being tricked by user input.
    Here the threat is the model's own output containing an executable payload.

    Subtype IDs mirror the upstream OWASP LLM02 taxonomy in
    `ppcvote/prompt-defense-audit`, allowing a single coverage matrix across
    deepteam, Cisco mcp-scanner, and Microsoft AGT. The canonical taxonomy
    form is dashed (e.g. `sqli-destructive`); this enum stores the underscored
    form for Python convention and exposes the dashed form via `taxonomy_id`.

    - sqli_destructive: DROP/DELETE/TRUNCATE/ALTER...DROP after a statement
      terminator (critical severity)
    - sqli_union: UNION-based extraction payloads (high severity)
    - sqli_comment_bypass: Classic quote-semicolon-comment auth bypass
      (medium severity)
    """

    SQLI_DESTRUCTIVE = "sqli_destructive"
    SQLI_UNION = "sqli_union"
    SQLI_COMMENT_BYPASS = "sqli_comment_bypass"

    @property
    def taxonomy_id(self) -> str:
        """Dashed form used by upstream OWASP LLM02 catalog for interop."""
        return self.value.replace("_", "-")


SQL_INJECTION_PAYLOAD_EMISSION_TYPES = [
    SQLInjectionPayloadEmissionType.SQLI_DESTRUCTIVE,
    SQLInjectionPayloadEmissionType.SQLI_UNION,
    SQLInjectionPayloadEmissionType.SQLI_COMMENT_BYPASS,
]
