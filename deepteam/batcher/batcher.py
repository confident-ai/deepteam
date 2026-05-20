import json
from typing import Dict, List, Any

from deepeval.tracing.types import AgentSpan, LlmSpan, RetrieverSpan, ToolSpan, Trace, BaseSpan
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import trimAndLoadJson

from .schema import BatchFinding, BatchFindingsList, ExtractedSpan

class TraceBatchEvaluator:
    def __init__(
        self,
        model: DeepEvalBaseLLM,
        using_native_model: bool,
        template: Any,
        limit: int = 40000,
    ):
        self.model = model
        self.using_native_model = using_native_model
        self.template = template
        self.limit = limit
        
        # Internal State is now strictly typed
        self.current_batch: List[ExtractedSpan] = []
        self.current_batch_size: int = 0
        self.all_findings: Dict[str, List[BatchFinding]] = {}

    def process_trace(self, trace: Trace) -> Dict[str, List[BatchFinding]]:
        self._reset_state()
        
        for span in trace.root_spans:
            self._traverse_post_order(span)
            
        trace_node = self._extract_trace_root(trace)
        self._add_to_batch_and_check(trace_node)
        
        if self.current_batch:
            self._flush_batch()
            
        return self.all_findings

    async def a_process_trace(self, trace: Trace) -> Dict[str, List[BatchFinding]]:
        self._reset_state()
        
        for span in trace.root_spans:
            await self._a_traverse_post_order(span)
            
        trace_node = self._extract_trace_root(trace)
        await self._a_add_to_batch_and_check(trace_node)
        
        if self.current_batch:
            await self._a_flush_batch()
            
        return self.all_findings

    def _reset_state(self):
        self.current_batch = []
        self.current_batch_size = 0
        self.all_findings = {}

    # ---------------------------------------------------------
    # SYNCHRONOUS TRAVERSAL
    # ---------------------------------------------------------

    def _traverse_post_order(self, span: BaseSpan):
        for child in span.children:
            self._traverse_post_order(child)
            
        child_findings = self._get_child_findings(span.children)
        span_node = self._extract_span_with_findings(span, child_findings)
        
        self._add_to_batch_and_check(span_node)

    def _add_to_batch_and_check(self, node: ExtractedSpan):
        # Dump to dict strictly excluding None to save tokens
        node_dict = node.model_dump(exclude_none=True)
        node_str = json.dumps(node_dict)
        node_size = len(node_str)
        
        if self.current_batch_size + node_size > self.limit and self.current_batch:
            self._flush_batch()

        self.current_batch.append(node)
        self.current_batch_size += node_size

        if self.current_batch_size >= self.limit:
            self._flush_batch()

    def _flush_batch(self):
        if not self.current_batch:
            return
            
        batch_list = [node.model_dump(exclude_none=True) for node in self.current_batch]
        batch_string = json.dumps(batch_list, indent=2)
        prompt = self.template.generate_trace_batch_evaluation(batch_data=batch_string)

        if self.using_native_model:
            res, _ = self.model.generate(prompt, schema=BatchFindingsList)
            findings = res.findings
        else:
            try:
                res: BatchFindingsList = self.model.generate(prompt, schema=BatchFindingsList)
                findings = res.findings
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res)
                findings = [BatchFinding(**item) for item in data.get("findings", [])]

        self._store_findings(findings)
        self.current_batch = []
        self.current_batch_size = 0

    # ---------------------------------------------------------
    # ASYNCHRONOUS TRAVERSAL
    # ---------------------------------------------------------

    async def _a_traverse_post_order(self, span: BaseSpan):
        for child in span.children:
            await self._a_traverse_post_order(child)
            
        child_findings = self._get_child_findings(span.children)
        span_node = self._extract_span_with_findings(span, child_findings)
        
        await self._a_add_to_batch_and_check(span_node)

    async def _a_add_to_batch_and_check(self, node: ExtractedSpan):
        node_dict = node.model_dump(exclude_none=True)
        node_str = json.dumps(node_dict)
        node_size = len(node_str)
        
        if self.current_batch_size + node_size > self.limit and self.current_batch:
            await self._a_flush_batch()
            
        self.current_batch.append(node)
        self.current_batch_size += node_size

        if self.current_batch_size >= self.limit:
            await self._a_flush_batch()

    async def _a_flush_batch(self):
        if not self.current_batch:
            return
            
        batch_list = [node.model_dump(exclude_none=True) for node in self.current_batch]
        batch_string = json.dumps(batch_list, indent=2)
        prompt = self.template.generate_trace_batch_evaluation(batch_data=batch_string)

        if self.using_native_model:
            res, _ = await self.model.a_generate(prompt, schema=BatchFindingsList)
            findings = res.findings
        else:
            try:
                res: BatchFindingsList = await self.model.a_generate(prompt, schema=BatchFindingsList)
                findings = res.findings
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res)
                findings = [BatchFinding(**item) for item in data.get("findings", [])]

        self._store_findings(findings)
        self.current_batch = []
        self.current_batch_size = 0

    # ---------------------------------------------------------
    # UTILITIES & EXTRACTION
    # ---------------------------------------------------------

    def _store_findings(self, findings: List[BatchFinding]):
        for finding in findings:
            if finding.spanUuid not in self.all_findings:
                self.all_findings[finding.spanUuid] = []
            self.all_findings[finding.spanUuid].append(finding)

    def _get_child_findings(self, children: List[BaseSpan]) -> List[BatchFinding]:
        findings = []
        for child in children:
            if child.uuid in self.all_findings:
                findings.extend(self.all_findings[child.uuid])
        return findings

    def _extract_span_with_findings(self, span: BaseSpan, child_findings: List[BatchFinding]) -> ExtractedSpan:
        """Strips useless metadata, keeps I/O, tools, and attaches child findings."""
        
        extracted = ExtractedSpan(
            spanUuid=span.uuid,
            type=span.__class__.__name__,
            name=span.name,
            status=span.status.value if span.status else None,
            input=span.input,
            output=span.output,
            error=span.error,
            context=span.context,
            retrieval_context=span.retrieval_context,
            expected_output=span.expected_output,
            tools_called=span.tools_called,
            child_findings=child_findings if child_findings else None
        )
        
        # 2. Map Subclass-Specific Critical Fields
        if isinstance(span, LlmSpan):
            extracted.model = span.model
            
        elif isinstance(span, AgentSpan):
            extracted.available_tools = span.available_tools
            extracted.agent_handoffs = span.agent_handoffs
            
        elif isinstance(span, RetrieverSpan):
            extracted.embedder = span.embedder
            
        elif isinstance(span, ToolSpan):
            extracted.description = span.description
            
        return extracted

    def _extract_trace_root(self, trace: Trace) -> ExtractedSpan:
        """Extracts the top-level trace I/O and any root-level findings."""
        root_findings = self._get_child_findings(trace.root_spans)
        
        return ExtractedSpan(
            spanUuid=trace.uuid,
            type="TraceRoot",
            name=getattr(trace, "name", None),
            status=trace.status.value if trace.status else None,
            input=trace.input,
            output=trace.output,
            context=getattr(trace, "context", None),
            retrieval_context=getattr(trace, "retrieval_context", None),
            expected_output=getattr(trace, "expected_output", None),
            tools_called=[tc.model_dump(exclude_none=True) for tc in trace.tools_called] if getattr(trace, "tools_called", None) else None,
            child_findings=root_findings if root_findings else None
        )