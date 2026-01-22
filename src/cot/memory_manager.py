"""
Intelligent Memory Management for Chain-of-Thought Pipeline

Based on: "LLM-Driven Intelligent Memory Optimization Engine"

Features:
- Context summarization and compression
- Memory evolution (continuously improving context)
- Intelligent context window management
- Selective context passing
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from . import PipelineContext, StepResult, StepStatus
from ..utils.token_optimizer import TokenOptimizer, count_tokens_approximate


@dataclass
class MemorySnapshot:
    """Snapshot of pipeline memory at a point in time."""
    step_name: str
    timestamp: datetime
    compressed_context: str
    key_insights: List[str]
    token_count: int


class MemoryManager:
    """
    Manages and optimizes pipeline memory/context.
    
    Features:
    - Compress context to fit token budgets
    - Extract key insights from steps
    - Evolve memory over time
    - Selective context passing
    """
    
    def __init__(
        self,
        max_context_tokens: int = 2000,
        compression_ratio: float = 0.3
    ):
        """
        Initialize memory manager.
        
        Args:
            max_context_tokens: Maximum tokens for context
            compression_ratio: Target compression ratio (0.3 = 30% of original)
        """
        self.max_context_tokens = max_context_tokens
        self.compression_ratio = compression_ratio
        self.optimizer = TokenOptimizer(aggressive=True)
        self.memory_history: List[MemorySnapshot] = []
    
    def compress_context(
        self,
        context: PipelineContext,
        target_step: Optional[str] = None
    ) -> str:
        """
        Compress context to fit token budget.
        
        Args:
            context: Pipeline context
            target_step: Optional step name to focus on
            
        Returns:
            Compressed context string
        """
        # Build context summary
        parts = []
        
        # Add key insights from each step
        for step_name, result in context.step_results.items():
            if result.status != StepStatus.SUCCESS:
                continue
            
            # Extract key information
            insights = self._extract_insights(step_name, result)
            if insights:
                parts.append(f"{step_name}: {', '.join(insights[:2])}")
        
        # Add current text (compressed)
        if context.current_text:
            text_tokens = count_tokens_approximate(context.current_text)
            if text_tokens > self.max_context_tokens * 0.5:
                # Compress text
                compressed, _ = self.optimizer.compress_prompt(
                    context.current_text,
                    max_tokens=int(self.max_context_tokens * 0.5)
                )
                parts.append(f"Text: {compressed[:500]}...")
            else:
                parts.append(f"Text: {context.current_text[:500]}...")
        
        compressed = "\n".join(parts)
        
        # Ensure it fits budget
        tokens = count_tokens_approximate(compressed)
        if tokens > self.max_context_tokens:
            compressed, _ = self.optimizer.compress_prompt(
                compressed,
                max_tokens=self.max_context_tokens
            )
        
        return compressed
    
    def _extract_insights(self, step_name: str, result: StepResult) -> List[str]:
        """Extract key insights from a step result."""
        insights = []
        output = result.output
        
        # Step-specific insight extraction
        if "domain" in step_name:
            domain = output.get("primary_domain") or output.get("domain")
            if domain:
                insights.append(f"Domain: {domain}")
        
        if "language" in step_name:
            lang = output.get("language_name") or output.get("language_code")
            if lang:
                insights.append(f"Language: {lang}")
        
        if "sentiment" in step_name:
            sent = output.get("sentiment")
            if sent:
                insights.append(f"Sentiment: {sent}")
        
        if "summary" in step_name:
            summary = output.get("summary", "")
            if summary:
                # Extract first sentence
                first_sent = summary.split('.')[0]
                insights.append(f"Summary: {first_sent[:100]}")
        
        if "ner" in step_name or "entities" in step_name:
            entities = output.get("entities", [])
            if entities:
                entity_names = [str(e.get("text", e))[:20] for e in entities[:3]]
                insights.append(f"Entities: {', '.join(entity_names)}")
        
        # Add confidence if low
        if result.confidence < 0.7:
            insights.append(f"Low confidence: {result.confidence:.2f}")
        
        return insights
    
    def evolve_memory(
        self,
        context: PipelineContext,
        new_result: StepResult
    ) -> str:
        """
        Evolve memory by incorporating new result.
        
        Updates memory to reflect new information while maintaining
        important historical context.
        """
        # Add new result to context
        context.add_result(new_result)
        
        # Create memory snapshot
        snapshot = MemorySnapshot(
            step_name=new_result.step_name,
            timestamp=datetime.now(),
            compressed_context=self.compress_context(context),
            key_insights=self._extract_insights(new_result.step_name, new_result),
            token_count=count_tokens_approximate(self.compress_context(context))
        )
        
        self.memory_history.append(snapshot)
        
        # Return evolved context
        return snapshot.compressed_context
    
    def get_relevant_context(
        self,
        context: PipelineContext,
        current_step: str
    ) -> str:
        """
        Get relevant context for current step.
        
        Selectively includes only relevant previous steps.
        """
        # Determine which previous steps are relevant
        relevant_steps = self._get_relevant_steps(current_step)
        
        # Build context from relevant steps only
        parts = []
        for step_name in relevant_steps:
            result = context.step_results.get(step_name)
            if result and result.status == StepStatus.SUCCESS:
                insights = self._extract_insights(step_name, result)
                if insights:
                    parts.append(f"{step_name}: {', '.join(insights)}")
        
        # Add current text
        if context.current_text:
            parts.append(f"Current text: {context.current_text[:300]}")
        
        return "\n".join(parts)
    
    def _get_relevant_steps(self, current_step: str) -> List[str]:
        """Determine which previous steps are relevant for current step."""
        # Step dependencies
        dependencies = {
            "translation": ["text_cleaning", "language_detection"],
            "summary": ["text_cleaning", "translation"],
            "domain_detection": ["text_cleaning"],
            "sentiment": ["text_cleaning", "translation"],
            "ner": ["text_cleaning", "translation"],
            "events": ["text_cleaning", "translation", "ner"],
            "relevancy": ["text_cleaning", "domain_detection"],
        }
        
        return dependencies.get(current_step, ["text_cleaning"])
    
    def optimize_for_step(
        self,
        context: PipelineContext,
        step_name: str,
        max_tokens: int
    ) -> str:
        """
        Optimize context specifically for a step.
        
        Args:
            context: Full pipeline context
            step_name: Name of step to optimize for
            max_tokens: Maximum tokens allowed
            
        Returns:
            Optimized context string
        """
        # Get relevant context
        relevant = self.get_relevant_context(context, step_name)
        
        # Compress if needed
        tokens = count_tokens_approximate(relevant)
        if tokens > max_tokens:
            compressed, _ = self.optimizer.compress_prompt(
                relevant,
                max_tokens=max_tokens
            )
            return compressed
        
        return relevant

