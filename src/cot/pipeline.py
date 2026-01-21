"""
CoT Pipeline Runner - Orchestrates the chain-of-thought workflow.

Implements:
- Sequential step execution with context passing
- Parallel step execution where possible
- Feedback loops and retry logic
- Comprehensive JSON output
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from . import PipelineContext, StepResult, StepStatus
from .executor import StepExecutor
from .steps import (
    TextCleaningStep,
    DomainDetectionStep,
    LanguageDetectionStep,
    ValidationStep
)
from ..utils.groq_client import GroqClient
from ..config import config


@dataclass
class PipelineConfig:
    """Configuration for the CoT pipeline."""
    enable_validation: bool = True
    enable_domain_detection: bool = True
    enable_semantic_cleaning: bool = False  # LLM-powered semantic cleaning
    enable_ner: bool = False  # Named Entity Recognition with ERA-CoT
    enable_relationships: bool = True  # Extract entity relationships (with NER)
    enable_events: bool = False  # Temporal event extraction
    enable_sentiment: bool = False  # Sentiment and emotion analysis
    enable_summary: bool = False  # Text summarization
    summary_style: str = "bullets"  # bullets, paragraph, executive, headlines, tldr
    enable_translation: bool = False  # Translate to English
    enable_relevancy: bool = False  # Relevancy scoring
    enable_country: bool = False  # Country/Region identification
    relevancy_topics: List[str] = None  # Custom topics for relevancy
    use_self_consistency: bool = False
    max_retries_per_step: int = 2
    skip_on_failure: bool = True  # Continue even if a step fails


class CoTPipeline:
    """
    Chain-of-Thought Pipeline with modern function calling.
    
    Executes a sequence of steps, each with:
    - Structured input/output via function definitions
    - Chain-of-thought reasoning
    - Context passing between steps
    - Validation and feedback loops
    
    Usage:
        pipeline = CoTPipeline(api_key="your-key")
        result = pipeline.run("Your text here")
        print(json.dumps(result, indent=2))
    """
    
    def __init__(
        self,
        api_key: str = None,
        pipeline_config: PipelineConfig = None
    ):
        """
        Initialize the CoT pipeline.
        
        Args:
            api_key: Groq API key (optional, only needed for domain detection)
            pipeline_config: Configuration options
        """
        self.config = pipeline_config or PipelineConfig()
        
        # Initialize executor if API key available
        self.executor = None
        if api_key or config.groq.api_key:
            try:
                client = GroqClient(api_key=api_key)
                self.executor = StepExecutor(client)
            except Exception:
                pass  # Continue without LLM executor
        
        # Initialize steps
        self.steps = self._build_steps()
    
    def _build_steps(self) -> List:
        """Build the pipeline steps."""
        steps = []
        
        # Step 1: Text Cleaning (always runs, local)
        steps.append(TextCleaningStep())
        
        # Step 2: Language Detection (local) - Detect BEFORE translation/semantic cleaning
        steps.append(LanguageDetectionStep())
        
        # Step 1.5: Semantic Cleaning (optional, requires LLM)
        if self.config.enable_semantic_cleaning and self.executor:
            from ..cleaners import SemanticCleaningStep
            steps.append(SemanticCleaningStep(executor=self.executor))
        
        # Step 2: Translation to English (optional, requires LLM)
        # Should run early so other steps work on English text
        if self.config.enable_translation and self.executor:
            from ..processors import TranslationStep
            steps.append(TranslationStep(executor=self.executor))
        
        # Step 3: NER - Named Entity Recognition (optional, requires LLM)
        if self.config.enable_ner and self.executor:
            from ..processors import NERStep
            steps.append(NERStep(
                executor=self.executor,
                extract_relationships=self.config.enable_relationships,
                use_self_consistency=self.config.use_self_consistency
            ))
        
        # Step 4: Event Calendar Extraction (optional, requires LLM)
        if self.config.enable_events and self.executor:
            from ..processors import EventCalendarStep
            steps.append(EventCalendarStep(executor=self.executor))
        
        # Step 5: Sentiment Analysis (optional, requires LLM)
        if self.config.enable_sentiment and self.executor:
            from ..processors import SentimentStep
            steps.append(SentimentStep(executor=self.executor))
        
        # Step 6: Text Summarization (optional, requires LLM)
        if self.config.enable_summary and self.executor:
            from ..processors import SummaryStep
            steps.append(SummaryStep(
                executor=self.executor,
                style=self.config.summary_style
            ))
        
        # Step 7: Relevancy Analysis (optional, requires LLM)
        if self.config.enable_relevancy and self.executor:
            from ..processors import RelevancyStep
            steps.append(RelevancyStep(
                executor=self.executor,
                topics=self.config.relevancy_topics
            ))
        
        # Step 8: Domain Detection (requires LLM)
        if self.config.enable_domain_detection:
            steps.append(DomainDetectionStep(
                executor=self.executor,
                use_consistency=self.config.use_self_consistency
            ))
            
        # Step 9: Country Identification (requires LLM)
        if self.config.enable_country and self.executor:
            from ..processors import CountryStep
            steps.append(CountryStep(executor=self.executor))
        
        
        # Step 7: Validation (optional) - always use basic validation (no LLM needed)
        if self.config.enable_validation:
            steps.append(ValidationStep(executor=None))  # Use basic validation
        
        return steps
    
    def run(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run the full CoT pipeline.
        
        Args:
            text: Input text to process
            metadata: Optional metadata to include
            
        Returns:
            Structured JSON output with all step results
        """
        start_time = datetime.now(timezone.utc)
        
        # Initialize context
        context = PipelineContext(
            original_input=text,
            current_text=text,
            metadata=metadata or {}
        )
        
        # Execute each step
        for step in self.steps:
            try:
                result = step.execute(context)
                context.add_result(result)
                
                # Update current_text if cleaning step
                if step.name == "text_cleaning" and result.status == StepStatus.SUCCESS:
                    context.current_text = result.output.get("cleaned_text", text)
                
                # Handle failures
                if result.status == StepStatus.FAILED:
                    context.errors.append(f"Step {step.name} failed: {result.output.get('error', 'Unknown error')}")
                    if not self.config.skip_on_failure:
                        break
                        
            except Exception as e:
                context.errors.append(f"Step {step.name} exception: {str(e)}")
                context.add_result(StepResult(
                    step_name=step.name,
                    status=StepStatus.FAILED,
                    output={"error": str(e)},
                    reasoning="",
                    confidence=0.0
                ))
                if not self.config.skip_on_failure:
                    break
        
        # Build final output
        end_time = datetime.now(timezone.utc)
        return self._build_output(context, start_time, end_time)
    
    def _build_output(
        self,
        context: PipelineContext,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Build the final structured output."""
        output = {}
        
        # Add each step's output with numbered keys
        step_num = 1
        for name, result in context.step_results.items():
            key = f"{step_num}_{name}"
            output[key] = {
                "status": result.status.value,
                "output": result.output,
                "reasoning": result.reasoning,
                "confidence": result.confidence,
                "duration_ms": result.duration_ms
            }
            if result.validation_notes:
                output[key]["validation_notes"] = result.validation_notes
            step_num += 1
        
        # Add metadata
        output["metadata"] = {
            "pipeline_version": "2.0.0",
            "pipeline_type": "chain_of_thought",
            "model_used": config.groq.model,
            "processed_at": end_time.isoformat(),
            "total_duration_ms": int((end_time - start_time).total_seconds() * 1000),
            "steps_executed": len(context.step_results),
            "errors": context.errors if context.errors else None
        }
        
        # Add chain-of-thought summary
        output["chain_of_thought_summary"] = context.get_chain_summary()
        
        return output
    
    def run_step(
        self,
        step_name: str,
        text: str,
        previous_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run a single step of the pipeline.
        
        Useful for:
        - Testing individual steps
        - Re-running failed steps
        - Custom pipeline composition
        """
        # Find the step
        step = None
        for s in self.steps:
            if s.name == step_name:
                step = s
                break
        
        if not step:
            return {"error": f"Step '{step_name}' not found"}
        
        # Build context
        context = PipelineContext(
            original_input=text,
            current_text=text
        )
        
        # Add previous results if provided
        if previous_context:
            for name, data in previous_context.items():
                context.step_results[name] = StepResult(
                    step_name=name,
                    status=StepStatus.SUCCESS,
                    output=data.get("output", {}),
                    reasoning=data.get("reasoning", "")
                )
        
        # Execute
        result = step.execute(context)
        
        return {
            "step_name": step_name,
            "result": result.to_dict()
        }
    
    def to_json(self, result: Dict[str, Any], pretty: bool = True) -> str:
        """Convert result to JSON string."""
        if pretty:
            return json.dumps(result, indent=2, ensure_ascii=False, default=str)
        return json.dumps(result, ensure_ascii=False, default=str)


# ============== Convenience Functions ==============

def create_pipeline(
    api_key: str = None,
    enable_validation: bool = True,
    enable_domain: bool = True,
    use_consistency: bool = False
) -> CoTPipeline:
    """
    Create a configured CoT pipeline.
    
    Args:
        api_key: Groq API key
        enable_validation: Whether to run validation step
        enable_domain: Whether to run domain detection
        use_consistency: Use self-consistency for domain detection
    """
    config = PipelineConfig(
        enable_validation=enable_validation,
        enable_domain_detection=enable_domain,
        use_self_consistency=use_consistency
    )
    return CoTPipeline(api_key=api_key, pipeline_config=config)


def process_with_cot(
    text: str,
    api_key: str = None
) -> Dict[str, Any]:
    """
    Quick function to process text with CoT pipeline.
    
    Args:
        text: Input text
        api_key: Optional Groq API key
        
    Returns:
        Pipeline result dictionary
    """
    pipeline = create_pipeline(api_key=api_key)
    return pipeline.run(text)
