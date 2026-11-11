from enum import Enum


class ShellCommandPayloadEmissionType(Enum):
    """
    Enum for Shell Command Payload Emission vulnerability types.

    These detect when an LLM emits a dangerous shell payload *in its output*
    that a downstream system could execute. This is distinct from
    `ShellInjection`, which tests an LLM-as-shell-interface for being tricked
    by user input. Here the threat is the model's own output containing an
    executable payload.

    Subtype IDs mirror the upstream OWASP LLM02 taxonomy in
    `ppcvote/prompt-defense-audit`, allowing a single coverage matrix across
    deepteam, Cisco mcp-scanner, and Microsoft AGT. The canonical taxonomy
    form is dashed (e.g. `shell-pipe-exec`); this enum stores the underscored
    form for Python convention and exposes the dashed form via `taxonomy_id`.

    - shell_pipe_exec: curl/wget piped to sh/bash for remote script execution
      (critical severity)
    - shell_destructive: rm -rf /, mkfs on a device, dd zeroing a device,
      chmod 777 on root (critical severity)
    - shell_reverse: Reverse shells via /dev/tcp, nc, bash redirect, python
      socket import (critical severity)
    - shell_env_exfil: Exfiltration of AWS/OPENAI/ANTHROPIC env vars or full
      env via curl/wget/nc/base64 (high severity)
    """

    SHELL_PIPE_EXEC = "shell_pipe_exec"
    SHELL_DESTRUCTIVE = "shell_destructive"
    SHELL_REVERSE = "shell_reverse"
    SHELL_ENV_EXFIL = "shell_env_exfil"

    @property
    def taxonomy_id(self) -> str:
        """Dashed form used by upstream OWASP LLM02 catalog for interop."""
        return self.value.replace("_", "-")


SHELL_COMMAND_PAYLOAD_EMISSION_TYPES = [
    ShellCommandPayloadEmissionType.SHELL_PIPE_EXEC,
    ShellCommandPayloadEmissionType.SHELL_DESTRUCTIVE,
    ShellCommandPayloadEmissionType.SHELL_REVERSE,
    ShellCommandPayloadEmissionType.SHELL_ENV_EXFIL,
]
