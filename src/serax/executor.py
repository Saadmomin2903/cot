"""
SERAX Executor - Executes SERAX-formatted LLM calls with parsing and validation.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from . import SeraxParser, SeraxSchema, SeraxFormatter
from .prompts import SeraxPromptBuilder
from ..utils.groq_client import GroqClient
from ..cot import StepResult, StepStatus, PipelineContext


@dataclass
class SeraxResult:
    """Result from a SERAX extraction."""
    status: str  # success, error, partial
    data: Dict[str, Any]
    raw_response: str
    validation_errors: List[str]
    duration_ms: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "data": self.data,
            "validation_errors": self.validation_errors,
            "duration_ms": self.duration_ms
        }


class SeraxExecutor:
    """
    Executes LLM calls with SERAX format parsing.
    
    Features:
    - Automatic SERAX parsing of LLM responses
    - Schema validation
    - Retry logic for parsing failures
    - Fallback to JSON if SERAX fails
    """
    
    def __init__(self, groq_client: GroqClient = None):
        """Initialize with Groq client."""
        self.client = groq_client or GroqClient()
        self.prompt_builder = SeraxPromptBuilder()
        self.formatter = SeraxFormatter()
        self.max_retries = 2
    
    def execute(
        self,
        text: str,
        schema: SeraxSchema,
        task_description: str = None,
        context: Dict[str, Any] = None,
        temperature: float = 0.1
    ) -> SeraxResult:
        """
        Execute a SERAX extraction.
        
        Args:
            text: Input text to process
            schema: SERAX schema for expected output
            task_description: Description of the task
            context: Previous step context
            temperature: LLM temperature
            
        Returns:
            SeraxResult with parsed data
        """
        start_time = time.time()
        
        # Build prompts
        system_prompt = self.prompt_builder.build_system_prompt(
            task_description=task_description or schema.description,
            schema=schema,
            include_cot=True
        )
        
        user_prompt = self.prompt_builder.build_user_prompt(text, context)
        
        # Create parser
        parser = SeraxParser(schema)
        
        for attempt in range(self.max_retries):
            try:
                # Call LLM
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                response = self.client.chat(
                    messages=messages,
                    temperature=temperature + (attempt * 0.1)  # Increase temp on retry
                )
                
                # Parse SERAX response
                parsed = parser.parse(response)
                
                # Check for validation errors
                meta = parsed.get("_serax_meta", {})
                errors = meta.get("validation_errors", [])
                
                duration_ms = int((time.time() - start_time) * 1000)
                
                if not errors or attempt == self.max_retries - 1:
                    return SeraxResult(
                        status="success" if not errors else "partial",
                        data={k: v for k, v in parsed.items() if not k.startswith("_")},
                        raw_response=response,
                        validation_errors=errors,
                        duration_ms=duration_ms
                    )
                
                # Retry with correction prompt
                user_prompt = self._add_correction(user_prompt, errors)
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return SeraxResult(
                        status="error",
                        data={"error": str(e)},
                        raw_response="",
                        validation_errors=[str(e)],
                        duration_ms=int((time.time() - start_time) * 1000)
                    )
        
        # Should not reach here
        return SeraxResult(
            status="error",
            data={},
            raw_response="",
            validation_errors=["Max retries exceeded"],
            duration_ms=int((time.time() - start_time) * 1000)
        )
    
    def execute_multi_task(
        self,
        text: str,
        tasks: List[str] = None,
        context: Dict[str, Any] = None,
        temperature: float = 0.1
    ) -> SeraxResult:
        """
        Execute multi-task extraction in a single call.
        
        Args:
            text: Input text
            tasks: List of tasks (default: all)
            context: Previous context
            temperature: LLM temperature
        """
        from . import MULTI_TASK_SCHEMA
        
        start_time = time.time()
        
        system_prompt, user_prompt = self.prompt_builder.build_multi_task_prompt(
            text, tasks, context
        )
        
        parser = SeraxParser(MULTI_TASK_SCHEMA)
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.client.chat(messages=messages, temperature=temperature)
            parsed = parser.parse(response)
            
            meta = parsed.get("_serax_meta", {})
            errors = meta.get("validation_errors", [])
            
            return SeraxResult(
                status="success" if not errors else "partial",
                data={k: v for k, v in parsed.items() if not k.startswith("_")},
                raw_response=response,
                validation_errors=errors,
                duration_ms=int((time.time() - start_time) * 1000)
            )
            
        except Exception as e:
            return SeraxResult(
                status="error",
                data={"error": str(e)},
                raw_response="",
                validation_errors=[str(e)],
                duration_ms=int((time.time() - start_time) * 1000)
            )
    
    def _add_correction(self, prompt: str, errors: List[str]) -> str:
        """Add correction instructions for retry."""
        return (
            f"{prompt}\n\n"
            f"CORRECTION NEEDED - Your previous output had these issues:\n"
            f"{chr(10).join(f'- {e}' for e in errors)}\n\n"
            f"Please provide a corrected SERAX output."
        )


class SeraxPipelineStep:
    """
    A pipeline step that uses SERAX format for extraction.
    
    Can be used in the CoT pipeline or standalone.
    """
    
    def __init__(
        self,
        name: str,
        schema: SeraxSchema,
        task_description: str,
        executor: SeraxExecutor = None
    ):
        self.name = name
        self.schema = schema
        self.task_description = task_description
        self.executor = executor
    
    def execute(self, context: PipelineContext) -> StepResult:
        """Execute this step and return result."""
        if not self.executor:
            return StepResult(
                step_name=self.name,
                status=StepStatus.SKIPPED,
                output={"reason": "No executor configured"},
                reasoning="",
                confidence=0.0
            )
        
        text = context.current_text or context.original_input
        
        # Get previous step outputs for context
        prev_context = {}
        for step_name, result in context.step_results.items():
            if result.status == StepStatus.SUCCESS:
                prev_context[step_name] = result.output
        
        # Execute SERAX extraction
        serax_result = self.executor.execute(
            text=text,
            schema=self.schema,
            task_description=self.task_description,
            context=prev_context
        )
        
        status = StepStatus.SUCCESS if serax_result.status == "success" else StepStatus.FAILED
        if serax_result.status == "partial":
            status = StepStatus.NEEDS_VALIDATION
        
        return StepResult(
            step_name=self.name,
            status=status,
            output=serax_result.data,
            reasoning=serax_result.data.get("reasoning", ""),
            confidence=serax_result.data.get("confidence", 0.8),
            validation_notes="; ".join(serax_result.validation_errors) if serax_result.validation_errors else "",
            raw_response=serax_result.raw_response,
            duration_ms=serax_result.duration_ms
        )
