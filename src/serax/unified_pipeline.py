"""
Unified Multi-Task Pipeline - Supports both JSON and SERAX output formats.

Implements the full chain-of-thought workflow:
1. Text Cleaning
2. Named Entity Recognition
3. Domain Classification
4. Sentiment Analysis
5. Summarization
6. Event Extraction
7. Language Detection
8. Relevancy Scoring
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from ..cot import PipelineContext, StepResult, StepStatus
from ..cot.steps import TextCleaningStep, LanguageDetectionStep
from ..cot.executor import StepExecutor
from ..serax import (
    SeraxParser, SeraxFormatter, SeraxSchema, 
    MULTI_TASK_SCHEMA, DOMAIN_DETECTION_SCHEMA
)
from ..serax.executor import SeraxExecutor
from ..serax.prompts import SeraxPromptBuilder, MULTI_TASK_PROMPT
from ..utils.groq_client import GroqClient
from ..utils.text_normalizer import to_text
from ..config import config


class OutputFormat(Enum):
    """Output format options."""
    JSON = "json"
    SERAX = "serax"


@dataclass 
class PipelineOptions:
    """Configuration options for the unified pipeline."""
    output_format: OutputFormat = OutputFormat.SERAX
    enable_ner: bool = True
    enable_domain: bool = True
    enable_sentiment: bool = True
    enable_summary: bool = True
    enable_events: bool = True
    enable_language: bool = True
    enable_relevancy: bool = True
    enable_validation: bool = True
    use_self_consistency: bool = False
    max_text_length: int = 4000
    temperature: float = 0.1


class UnifiedPipeline:
    """
    Unified multi-task NLP pipeline with SERAX/JSON output.
    
    Features:
    - Full CoT reasoning chain
    - SERAX format for reliable extraction (default)
    - JSON fallback option
    - Modular task selection
    - Context passing between steps
    """
    
    def __init__(
        self,
        api_key: str = None,
        options: PipelineOptions = None
    ):
        """Initialize pipeline with configuration."""
        self.options = options or PipelineOptions()
        
        # Initialize clients
        self.groq_client = None
        self.serax_executor = None
        self.step_executor = None
        
        if api_key or config.groq.api_key:
            try:
                self.groq_client = GroqClient(api_key=api_key)
                self.serax_executor = SeraxExecutor(self.groq_client)
                self.step_executor = StepExecutor(self.groq_client)
            except Exception:
                pass
        
        # Initialize local steps
        self.text_cleaner = TextCleaningStep()
        self.language_detector = LanguageDetectionStep()
        
        # Prompt builder
        self.prompt_builder = SeraxPromptBuilder()
    
    def run(self, text: str) -> Dict[str, Any]:
        """
        Run the full pipeline on input text.
        
        Args:
            text: Raw input text
            
        Returns:
            Structured output with all task results
        """
        start_time = datetime.now(timezone.utc)
        
        # Initialize context
        normalized_input = to_text(text)
        context = PipelineContext(
            original_input=normalized_input,
            current_text=normalized_input
        )
        
        results = {}
        
        # ===== Step 1: Text Cleaning (Local) =====
        clean_result = self.text_cleaner.execute(context)
        context.add_result(clean_result)
        context.current_text = to_text(clean_result.output.get("cleaned_text", normalized_input))
        results["1_text_cleaning"] = self._format_step_result(clean_result)
        
        # ===== Step 2: Language Detection (Local) =====
        if self.options.enable_language:
            lang_result = self.language_detector.execute(context)
            context.add_result(lang_result)
            results["2_language_detection"] = self._format_step_result(lang_result)
        
        # ===== Step 3: Multi-Task LLM Extraction =====
        # Run NER, Domain, Sentiment, Summary, Events, Relevancy in one call
        if self.serax_executor and self._has_llm_tasks():
            tasks = self._get_enabled_tasks()
            
            if self.options.output_format == OutputFormat.SERAX:
                llm_result = self._run_serax_extraction(context, tasks)
            else:
                llm_result = self._run_json_extraction(context, tasks)
            
            # Unpack multi-task results into separate step outputs
            results.update(self._unpack_llm_results(llm_result, context))
        else:
            # Skip LLM tasks if no executor
            for i, task in enumerate(["ner", "domain", "sentiment", "summary", "events", "relevancy"], start=3):
                if getattr(self.options, f"enable_{task.replace('ner', 'ner')}", False):
                    results[f"{i}_{task}"] = {
                        "status": "skipped",
                        "reason": "No API key configured"
                    }
        
        # ===== Final: Validation =====
        if self.options.enable_validation:
            validation_result = self._validate_results(results, context)
            results["validation"] = validation_result
        
        # ===== Metadata =====
        end_time = datetime.now(timezone.utc)
        
        return {
            **results,
            "chain_of_thought_summary": context.get_chain_summary(),
            "metadata": {
                "pipeline_version": "3.0.0",
                "output_format": self.options.output_format.value,
                "model_used": config.groq.model,
                "processed_at": end_time.isoformat(),
                "total_duration_ms": int((end_time - start_time).total_seconds() * 1000),
                "tasks_enabled": self._get_enabled_tasks()
            }
        }
    
    def _has_llm_tasks(self) -> bool:
        """Check if any LLM-required tasks are enabled."""
        return any([
            self.options.enable_ner,
            self.options.enable_domain,
            self.options.enable_sentiment,
            self.options.enable_summary,
            self.options.enable_events,
            self.options.enable_relevancy
        ])
    
    def _get_enabled_tasks(self) -> List[str]:
        """Get list of enabled task names."""
        tasks = []
        if self.options.enable_ner:
            tasks.append("ner")
        if self.options.enable_domain:
            tasks.append("domain")
        if self.options.enable_sentiment:
            tasks.append("sentiment")
        if self.options.enable_summary:
            tasks.append("summary")
        if self.options.enable_events:
            tasks.append("events")
        if self.options.enable_relevancy:
            tasks.append("relevancy")
        return tasks
    
    def _run_serax_extraction(
        self,
        context: PipelineContext,
        tasks: List[str]
    ) -> Dict[str, Any]:
        """Run multi-task extraction using SERAX format."""
        context.current_text = to_text(context.current_text or context.original_input)
        result = self.serax_executor.execute_multi_task(
            text=context.current_text[:self.options.max_text_length],
            tasks=tasks,
            context={
                "cleaned_text_length": len(context.current_text),
                "language": context.step_results.get("language_detection", StepResult(
                    step_name="", status=StepStatus.SKIPPED, output={}
                )).output.get("language_code", "unknown")
            },
            temperature=self.options.temperature
        )
        
        return {
            "status": result.status,
            "data": result.data,
            "validation_errors": result.validation_errors,
            "duration_ms": result.duration_ms,
            "raw_response": result.raw_response
        }
    
    def _run_json_extraction(
        self,
        context: PipelineContext,
        tasks: List[str]
    ) -> Dict[str, Any]:
        """Run multi-task extraction using JSON format."""
        system_prompt = self._build_json_system_prompt(tasks)
        user_prompt = f"Analyze this text:\n\n{context.current_text[:self.options.max_text_length]}"
        
        try:
            result = self.groq_client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.options.temperature
            )
            
            return {
                "status": "success",
                "data": result,
                "validation_errors": [],
                "duration_ms": 0
            }
        except Exception as e:
            return {
                "status": "error",
                "data": {"error": str(e)},
                "validation_errors": [str(e)],
                "duration_ms": 0
            }
    
    def _build_json_system_prompt(self, tasks: List[str]) -> str:
        """Build JSON format system prompt."""
        return f"""You are an NLP expert. Analyze the text and return JSON with these fields:

{json.dumps({task: "..." for task in tasks}, indent=2)}

For each task:
- ner: {"persons": [], "orgs": [], "locations": [], "dates": []}
- domain: "technology" | "business" | "general"  
- sentiment: "positive" | "negative" | "neutral"
- summary: brief summary text
- events: ["date: event", ...]
- relevancy: 0.0 to 1.0

Return valid JSON only."""
    
    def _unpack_llm_results(
        self,
        llm_result: Dict[str, Any],
        context: PipelineContext
    ) -> Dict[str, Any]:
        """Unpack multi-task LLM results into separate step outputs."""
        results = {}
        data = llm_result.get("data", {})
        
        step_num = 3  # Start after text_cleaning and language_detection
        
        if self.options.enable_ner:
            results[f"{step_num}_ner"] = {
                "status": llm_result.get("status", "error"),
                "output": {
                    "persons": data.get("persons", data.get("entities", {}).get("persons", [])),
                    "orgs": data.get("orgs", data.get("entities", {}).get("orgs", [])),
                    "locations": data.get("locations", []),
                    "dates": data.get("dates", []),
                    "entities": data.get("entities", [])
                }
            }
            step_num += 1
        
        if self.options.enable_domain:
            results[f"{step_num}_domain"] = {
                "status": llm_result.get("status", "error"),
                "output": {
                    "domain": data.get("domain", "general"),
                    "confidence": data.get("domain_conf", data.get("confidence", 0.8)),
                    "subcategories": data.get("subcats", [])
                },
                "reasoning": data.get("reasoning", "")
            }
            step_num += 1
        
        if self.options.enable_sentiment:
            results[f"{step_num}_sentiment"] = {
                "status": llm_result.get("status", "error"),
                "output": {
                    "sentiment": data.get("sentiment", "neutral"),
                    "score": data.get("sent_score", 0.0),
                    "confidence": data.get("confidence", 0.8)
                }
            }
            step_num += 1
        
        if self.options.enable_summary:
            results[f"{step_num}_summary"] = {
                "status": llm_result.get("status", "error"),
                "output": {
                    "summary": data.get("summary", ""),
                    "key_points": data.get("key_points", [])
                }
            }
            step_num += 1
        
        if self.options.enable_events:
            results[f"{step_num}_events"] = {
                "status": llm_result.get("status", "error"),
                "output": {
                    "events": data.get("events", [])
                }
            }
            step_num += 1
        
        if self.options.enable_relevancy:
            results[f"{step_num}_relevancy"] = {
                "status": llm_result.get("status", "error"),
                "output": {
                    "score": data.get("relevancy", 0.5)
                }
            }
        
        return results
    
    def _format_step_result(self, result: StepResult) -> Dict[str, Any]:
        """Format a step result for output."""
        return {
            "status": result.status.value,
            "output": result.output,
            "reasoning": result.reasoning,
            "confidence": result.confidence,
            "duration_ms": result.duration_ms
        }
    
    def _validate_results(
        self,
        results: Dict[str, Any],
        context: PipelineContext
    ) -> Dict[str, Any]:
        """Validate all results for consistency."""
        issues = []
        
        # Check text cleaning worked
        clean = results.get("1_text_cleaning", {})
        if clean.get("status") == "success":
            reduction = clean.get("output", {}).get("reduction_percent", 0)
            if reduction > 90:
                issues.append("Text cleaning removed >90% of content")
        
        # Check domain and sentiment consistency
        domain = results.get("4_domain", results.get("3_domain", {}))
        if domain.get("status") == "success":
            if domain.get("output", {}).get("confidence", 1) < 0.5:
                issues.append("Low domain classification confidence")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "quality_score": max(0.5, 1.0 - len(issues) * 0.2)
        }
    
    def to_json(self, result: Dict[str, Any], pretty: bool = True) -> str:
        """Convert result to JSON string."""
        if pretty:
            return json.dumps(result, indent=2, ensure_ascii=False, default=str)
        return json.dumps(result, ensure_ascii=False, default=str)
    
    def to_serax(self, result: Dict[str, Any]) -> str:
        """Convert result to SERAX format string."""
        formatter = SeraxFormatter()
        
        # Flatten key results
        flat_data = {}
        for key, value in result.items():
            if isinstance(value, dict) and "output" in value:
                flat_data.update(value["output"])
        
        return formatter.format(flat_data)


# ============== Convenience Functions ==============

def run_multi_task(
    text: str,
    api_key: str = None,
    output_format: str = "serax"
) -> Dict[str, Any]:
    """
    Quick function for multi-task extraction.
    
    Args:
        text: Input text
        api_key: Optional Groq API key
        output_format: "serax" or "json"
    """
    fmt = OutputFormat.SERAX if output_format == "serax" else OutputFormat.JSON
    options = PipelineOptions(output_format=fmt)
    pipeline = UnifiedPipeline(api_key=api_key, options=options)
    return pipeline.run(text)
