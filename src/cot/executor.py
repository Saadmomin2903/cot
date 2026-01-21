"""
CoT Step Executor - Handles function calling with Groq LLM.

Implements:
- Structured function calling with JSON output
- Chain-of-thought prompting
- Retry logic with exponential backoff
- Output validation
"""

import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..utils.groq_client import GroqClient
from . import (
    FunctionDefinition, StepResult, StepStatus, PipelineContext
)


class StepExecutor:
    """
    Executes pipeline steps using Groq LLM with function calling.
    
    Features:
    - Structured output via JSON mode
    - Chain-of-thought reasoning prompts
    - Automatic retry on failures
    - Output validation
    """
    
    def __init__(self, groq_client: GroqClient = None):
        """Initialize with Groq client."""
        self.client = groq_client or GroqClient()
        self.max_retries = 3
        self.retry_delay = 1.0
    
    def execute_with_function(
        self,
        step_name: str,
        system_prompt: str,
        user_prompt: str,
        function_def: FunctionDefinition,
        context: PipelineContext,
        temperature: float = 0.1
    ) -> StepResult:
        """
        Execute a step with function calling.
        
        Args:
            step_name: Name of the step
            system_prompt: System message with CoT instructions
            user_prompt: User message with the task
            function_def: Function definition for structured output
            context: Pipeline context for state
            temperature: LLM temperature
            
        Returns:
            StepResult with structured output
        """
        start_time = time.time()
        
        # Build the full prompt with CoT instructions
        full_system = self._build_cot_system_prompt(
            system_prompt, function_def, context
        )
        
        for attempt in range(self.max_retries):
            try:
                # Call LLM with JSON mode for structured output
                response = self.client.chat_json(
                    system_prompt=full_system,
                    user_prompt=user_prompt,
                    temperature=temperature
                )
                
                # Validate response has required fields
                is_valid, validation_notes = self._validate_response(
                    response, function_def
                )
                
                duration_ms = int((time.time() - start_time) * 1000)
                
                if is_valid:
                    return StepResult(
                        step_name=step_name,
                        status=StepStatus.SUCCESS,
                        output=response,
                        reasoning=response.get("reasoning", ""),
                        confidence=response.get("confidence", 1.0),
                        validation_notes=validation_notes,
                        duration_ms=duration_ms
                    )
                else:
                    # Retry if validation failed
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        user_prompt = self._add_correction_prompt(
                            user_prompt, validation_notes
                        )
                        continue
                    
                    return StepResult(
                        step_name=step_name,
                        status=StepStatus.NEEDS_VALIDATION,
                        output=response,
                        reasoning=response.get("reasoning", ""),
                        confidence=0.5,
                        validation_notes=validation_notes,
                        duration_ms=duration_ms
                    )
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                
                duration_ms = int((time.time() - start_time) * 1000)
                return StepResult(
                    step_name=step_name,
                    status=StepStatus.FAILED,
                    output={"error": str(e)},
                    reasoning="",
                    confidence=0.0,
                    validation_notes=f"Error after {attempt + 1} attempts: {str(e)}",
                    duration_ms=duration_ms
                )
        
        # Should not reach here
        return StepResult(
            step_name=step_name,
            status=StepStatus.FAILED,
            output={"error": "Max retries exceeded"},
            reasoning="",
            confidence=0.0,
            duration_ms=int((time.time() - start_time) * 1000)
        )
    
    def _build_cot_system_prompt(
        self,
        base_prompt: str,
        function_def: FunctionDefinition,
        context: PipelineContext
    ) -> str:
        """Build system prompt with CoT instructions and context."""
        
        # Get previous step summaries for context passing
        previous_context = context.get_chain_summary()
        
        # Build JSON schema instruction
        schema_instruction = self._build_schema_instruction(function_def)
        
        prompt_parts = [
            base_prompt,
            "",
            "## Chain-of-Thought Instructions",
            "Think step-by-step before providing your answer:",
            "1. First, understand what is being asked",
            "2. Then, analyze the input systematically",
            "3. Consider edge cases and potential issues",
            "4. Finally, provide your structured output",
            "",
        ]
        
        if previous_context:
            prompt_parts.extend([
                "## Context from Previous Steps",
                previous_context,
                "",
            ])
        
        prompt_parts.extend([
            "## Required Output Format",
            schema_instruction,
            "",
            "IMPORTANT: You MUST respond with valid JSON matching the schema above.",
            "Include your step-by-step reasoning in the 'reasoning' field if available.",
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_schema_instruction(self, function_def: FunctionDefinition) -> str:
        """Build schema instruction from function definition."""
        schema = {
            "type": "object",
            "properties": function_def.parameters,
            "required": function_def.required
        }
        return f"```json\n{json.dumps(schema, indent=2)}\n```"
    
    def _validate_response(
        self,
        response: Dict[str, Any],
        function_def: FunctionDefinition
    ) -> tuple[bool, str]:
        """Validate response against function definition."""
        issues = []
        
        # Check required fields
        for field in function_def.required:
            if field not in response:
                issues.append(f"Missing required field: {field}")
        
        # Type validation for known fields
        for field, spec in function_def.parameters.items():
            if field in response:
                value = response[field]
                expected_type = spec.get("type")
                
                # Check enum constraints
                if "enum" in spec and value not in spec["enum"]:
                    issues.append(
                        f"Field '{field}' has invalid value '{value}'. "
                        f"Expected one of: {spec['enum']}"
                    )
                
                # Check numeric ranges
                if expected_type == "number":
                    if not isinstance(value, (int, float)):
                        issues.append(f"Field '{field}' should be a number")
                    elif "confidence" in field.lower() and not (0 <= value <= 1):
                        issues.append(f"Field '{field}' should be between 0 and 1")
        
        if issues:
            return False, "; ".join(issues)
        return True, "Validation passed"
    
    def _add_correction_prompt(
        self,
        original_prompt: str,
        validation_notes: str
    ) -> str:
        """Add correction instructions for retry."""
        return (
            f"{original_prompt}\n\n"
            f"CORRECTION NEEDED: Your previous response had issues: {validation_notes}\n"
            f"Please provide a corrected response following the exact schema."
        )


class SelfConsistencyExecutor:
    """
    Executor that runs steps multiple times for self-consistency.
    
    Implements the self-consistency method where multiple
    chain-of-thought runs are compared for agreement.
    """
    
    def __init__(self, base_executor: StepExecutor, num_runs: int = 3):
        self.executor = base_executor
        self.num_runs = num_runs
    
    def execute_with_consistency(
        self,
        step_name: str,
        system_prompt: str,
        user_prompt: str,
        function_def: FunctionDefinition,
        context: PipelineContext,
        key_field: str = "primary_domain"
    ) -> StepResult:
        """
        Execute step multiple times and return majority result.
        
        Args:
            key_field: The field to check for consistency across runs
        """
        results = []
        
        for i in range(self.num_runs):
            # Vary temperature slightly for diversity
            temperature = 0.1 + (i * 0.1)
            result = self.executor.execute_with_function(
                step_name=f"{step_name}_run_{i}",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                function_def=function_def,
                context=context,
                temperature=temperature
            )
            if result.status == StepStatus.SUCCESS:
                results.append(result)
        
        if not results:
            return StepResult(
                step_name=step_name,
                status=StepStatus.FAILED,
                output={"error": "All consistency runs failed"},
                reasoning="",
                confidence=0.0
            )
        
        # Find majority vote for key field
        votes = {}
        for r in results:
            key_value = r.output.get(key_field, "unknown")
            votes[key_value] = votes.get(key_value, 0) + 1
        
        majority_value = max(votes, key=votes.get)
        agreement_ratio = votes[majority_value] / len(results)
        
        # Return the result that matches majority
        for r in results:
            if r.output.get(key_field) == majority_value:
                return StepResult(
                    step_name=step_name,
                    status=StepStatus.SUCCESS,
                    output=r.output,
                    reasoning=r.reasoning,
                    confidence=agreement_ratio,  # Confidence based on agreement
                    validation_notes=f"Self-consistency: {votes[majority_value]}/{len(results)} agreement"
                )
        
        return results[0]  # Fallback
