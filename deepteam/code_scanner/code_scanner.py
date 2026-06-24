import asyncio
import json
import logging
from typing import List, Optional

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model

from deepteam.attacks.attack_simulator.utils import generate, a_generate

from .constants import CODE_CONTEXT_LIMIT
from .schema import CodeChunk, CodeFinding, CodeFindingsList
from .template import CodeScanTemplate

logger = logging.getLogger(__name__)


class CodeScanner:
    """
    Statically scans source code for AI-security vulnerabilities.
    """

    def __init__(
        self,
        model: DeepEvalBaseLLM,
        template=CodeScanTemplate,
        vulnerabilities: Optional[List[str]] = None,
        guidance: Optional[str] = None,
        limit: int = CODE_CONTEXT_LIMIT,
        max_concurrent: int = 10,
    ):
        self.model, self.using_native_model = initialize_model(model)
        self.template = template
        self.vulnerabilities = vulnerabilities
        self.guidance = guidance
        self.limit = limit
        self.max_concurrent = max_concurrent

    def _make_batches(self, chunks: List[CodeChunk]) -> List[List[CodeChunk]]:
        """
        Pack chunks into batches whose serialized size stays under
        limit. A chunk that alone exceeds the limit still gets its own batch
        (splitting oversized files is the collector's job, not the scanner's).
        """
        batches: List[List[CodeChunk]] = []
        current: List[CodeChunk] = []
        current_size = 0

        for chunk in chunks:
            size = len(json.dumps(chunk.model_dump(exclude_none=True)))
            if current and current_size + size > self.limit:
                batches.append(current)
                current = []
                current_size = 0
            current.append(chunk)
            current_size += size

        if current:
            batches.append(current)
        return batches

    def _build_prompt(self, batch: List[CodeChunk]) -> str:
        batch_string = json.dumps(
            [chunk.model_dump(exclude_none=True) for chunk in batch], 
            indent=2
        )
        return self.template.generate_code_batch_evaluation(
            batch_data=batch_string,
            vulnerabilities=self.vulnerabilities,
            guidance=self.guidance,
        )

    def scan(self, chunks: List[CodeChunk]) -> List[CodeFinding]:
        findings: List[CodeFinding] = []
        for batch in self._make_batches(chunks):
            try:
                prompt = self._build_prompt(batch)
                res: CodeFindingsList = generate(
                    prompt, CodeFindingsList, self.model
                )
                findings.extend(res.findings)
            except Exception as e:
                logger.warning(
                    "code_scan batch failed (%d chunks): %s", len(batch), e
                )
        return self._dedupe(findings)

    async def a_scan(self, chunks: List[CodeChunk]) -> List[CodeFinding]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        batches = self._make_batches(chunks)

        results = await asyncio.gather(
            *(self._a_scan_batch(batch, semaphore) for batch in batches),
            return_exceptions=True,
        )

        findings: List[CodeFinding] = []
        for batch, res in zip(batches, results):
            if isinstance(res, Exception):
                logger.warning(
                    "code_scan batch failed (%d chunks): %s",
                    len(batch),
                    res,
                )
                continue
            findings.extend(res)
        return self._dedupe(findings)

    async def _a_scan_batch(
        self, batch: List[CodeChunk], semaphore: asyncio.Semaphore
    ) -> List[CodeFinding]:
        prompt = self._build_prompt(batch)
        async with semaphore:
            res: CodeFindingsList = await a_generate(
                prompt, CodeFindingsList, self.model
            )
        return res.findings

    @staticmethod
    def _dedupe(findings: List[CodeFinding]) -> List[CodeFinding]:
        """
        Collapse findings that share a file, start line and vulnerability
        (the same issue can surface across overlapping batches).
        """
        seen = set()
        unique: List[CodeFinding] = []
        for f in findings:
            key = (
                f.filePath,
                f.lineStart,
                f.vulnerability,
                f.vulnerabilityType,
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(f)
        return unique
