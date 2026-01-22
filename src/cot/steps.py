"""
Concrete Pipeline Steps - CoT implementations for each task.

Each step implements:
- Chain-of-thought prompt generation
- Function definition for structured output
- Validation logic
- Context-aware execution
"""

from typing import Dict, Any
import time

from . import (
    PipelineStep, PipelineContext, StepResult, StepStatus,
    FunctionDefinition, TEXT_CLEAN_FUNCTION, DOMAIN_DETECT_FUNCTION,
    LANGUAGE_DETECT_FUNCTION, VALIDATE_OUTPUT_FUNCTION
)
from .executor import StepExecutor, SelfConsistencyExecutor
from ..cleaners import GlobalCleaner, TempCleaner
from ..processors import LanguageDetector
from ..utils.text_normalizer import to_text


class TextCleaningStep(PipelineStep):
    """
    Step 1: Text Cleaning with Chain-of-Thought.
    
    Uses local cleaning functions for speed, with optional
    LLM-based intelligent cleaning for complex cases.
    """
    
    def __init__(self):
        super().__init__(
            name="text_cleaning",
            description="Clean and normalize text content"
        )
        self.global_cleaner = GlobalCleaner()
        self.temp_cleaner = TempCleaner()
    
    def get_function_definition(self) -> FunctionDefinition:
        return TEXT_CLEAN_FUNCTION
    
    def get_cot_prompt(self, context: PipelineContext) -> str:
        return f"""You are a text cleaning expert. Your task is to clean the following text.

Think step-by-step:
1. First, identify what types of noise are present (URLs, HTML, navigation, etc.)
2. Then, plan what cleaning operations are needed
3. Apply the cleaning operations
4. Verify the cleaned text preserves meaningful content

Original text length: {len(context.original_input)} characters

Clean the following text:
{context.original_input[:2000]}{'...' if len(context.original_input) > 2000 else ''}"""
    
    def execute(self, context: PipelineContext) -> StepResult:
        """Execute text cleaning using local functions (fast path)."""
        start_time = time.time()
        
        try:
            # Step 1: Global cleaning
            global_result = self.global_cleaner.clean(context.original_input)
            
            # Step 2: Temp cleaning
            temp_result = self.temp_cleaner.clean(global_result["text"])
            
            cleaned_text = temp_result["text"]
            original_length = len(context.original_input)
            cleaned_length = len(cleaned_text)
            
            # Update context
            context.current_text = cleaned_text
            
            # Determine what was removed
            removed_elements = []
            if "url" in str(global_result).lower() or original_length > cleaned_length:
                if "https" in context.original_input or "http" in context.original_input:
                    removed_elements.append("urls")
                if "<" in context.original_input and ">" in context.original_input:
                    removed_elements.append("html_tags")
                if "[" in context.original_input and "]" in context.original_input:
                    removed_elements.append("bracketed_content")
            if temp_result["stats"]["lines_removed"] > 0:
                removed_elements.append("short_lines")
                removed_elements.append("navigation")
            
            output = {
                "cleaned_text": cleaned_text,
                "removed_elements": removed_elements,
                "original_length": original_length,
                "cleaned_length": cleaned_length,
                "reduction_percent": round(
                    (1 - cleaned_length / original_length) * 100, 2
                ) if original_length > 0 else 0
            }
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            return StepResult(
                step_name=self.name,
                status=StepStatus.SUCCESS,
                output=output,
                reasoning=f"Applied global cleaning (removed {', '.join(removed_elements) or 'nothing'}), "
                         f"then temp cleaning (removed {temp_result['stats']['lines_removed']} lines). "
                         f"Reduced text by {output['reduction_percent']}%.",
                confidence=1.0,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            return StepResult(
                step_name=self.name,
                status=StepStatus.FAILED,
                output={"error": str(e)},
                reasoning="",
                confidence=0.0,
                duration_ms=int((time.time() - start_time) * 1000)
            )


class DomainDetectionStep(PipelineStep):
    """
    Step 2: Domain Detection with Chain-of-Thought.
    
    Uses LLM with structured function calling to classify
    content into Technology, Business, or General domains.
    """
    
    def __init__(self, executor: StepExecutor = None, use_consistency: bool = False):
        super().__init__(
            name="domain_detection",
            description="Classify content domain using LLM"
        )
        self.executor = executor
        self.use_consistency = use_consistency
        
        # Add validators
        self.add_validator(self._validate_domain_scores)
    
    def get_function_definition(self) -> FunctionDefinition:
        return DOMAIN_DETECT_FUNCTION
    
    def get_cot_prompt(self, context: PipelineContext) -> str:
        # Get cleaned text from previous step
        cleaned_text = to_text(context.current_text or context.original_input)
        
        # Truncate for LLM
        text_preview = cleaned_text[:3000] + ("..." if len(cleaned_text) > 3000 else "")
        
        return f"""Classify the following text into ONE of three domains.

## Domains
1. **technology** - Software, hardware, programming, AI/ML, IT, engineering, web development
2. **business** - Companies, products, services, e-commerce, finance, marketing, corporate
3. **general** - News, education, entertainment, lifestyle, health, sports, other

## Chain-of-Thought Process
Think step-by-step:
1. First, identify key topics and themes in the text
2. Look for domain-specific terminology
3. Consider the overall purpose/intent of the content
4. Make your classification with confidence level

## Text to Classify
{text_preview}"""
    
    def get_system_prompt(self) -> str:
        return """You are an expert content classifier. Your task is to accurately categorize text content into predefined domains.

Be decisive - choose the single most appropriate domain. Provide probability scores for all domains that sum to 1.0.

Your reasoning should explicitly mention:
- Key indicators that led to your classification
- Why you ruled out other domains
- Your confidence level and what could change it"""
    
    def execute(self, context: PipelineContext) -> StepResult:
        """Execute domain detection using LLM."""
        if not self.executor:
            return StepResult(
                step_name=self.name,
                status=StepStatus.SKIPPED,
                output={"reason": "No executor configured (requires API key)"},
                reasoning="",
                confidence=0.0
            )
        
        system_prompt = self.get_system_prompt()
        user_prompt = self.get_cot_prompt(context)
        
        if self.use_consistency:
            consistency_executor = SelfConsistencyExecutor(self.executor, num_runs=3)
            return consistency_executor.execute_with_consistency(
                step_name=self.name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                function_def=self.get_function_definition(),
                context=context,
                key_field="primary_domain"
            )
        else:
            return self.executor.execute_with_function(
                step_name=self.name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                function_def=self.get_function_definition(),
                context=context
            )
    
    def _validate_domain_scores(self, output: Dict[str, Any]) -> bool:
        """Validate that domain scores sum to approximately 1.0."""
        scores = output.get("domain_scores", {})
        if not scores:
            return True  # Optional field
        
        total = sum(scores.values())
        return 0.95 <= total <= 1.05  # Allow small floating point errors


class LanguageDetectionStep(PipelineStep):
    """
    Step 3: Language Detection with Chain-of-Thought.
    
    Uses local langdetect library with Unicode analysis
    for fast, accurate language detection.
    """
    
    def __init__(self):
        super().__init__(
            name="language_detection",
            description="Detect language and script type"
        )
        self.detector = LanguageDetector()
    
    def get_function_definition(self) -> FunctionDefinition:
        return LANGUAGE_DETECT_FUNCTION
    
    def get_cot_prompt(self, context: PipelineContext) -> str:
        return f"""Detect the language and script type of this text.

Text sample:
{context.current_text[:500]}"""
    
    def execute(self, context: PipelineContext) -> StepResult:
        """Execute language detection using local detector."""
        start_time = time.time()
        
        text = context.current_text or context.original_input
        
        try:
            result = self.detector.detect(text)
            
            output = {
                "language_code": result.get("language_code"),
                "language_name": result.get("language_name"),
                "script_type": result.get("script_type"),
                "confidence": result.get("confidence", 1.0),
                "detected_scripts": list(result.get("script_breakdown", {}).keys())
            }
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            return StepResult(
                step_name=self.name,
                status=StepStatus.SUCCESS,
                output=output,
                reasoning=f"Detected {output['language_name']} ({output['language_code']}) "
                         f"with {output['script_type']} script at {output['confidence']:.0%} confidence.",
                confidence=output["confidence"],
                duration_ms=duration_ms
            )
            
        except Exception as e:
            return StepResult(
                step_name=self.name,
                status=StepStatus.FAILED,
                output={"error": str(e)},
                reasoning="",
                confidence=0.0,
                duration_ms=int((time.time() - start_time) * 1000)
            )


class ValidationStep(PipelineStep):
    """
    Final Step: Validate and verify all previous outputs.
    
    Uses LLM to review the chain and check for inconsistencies.
    """
    
    def __init__(self, executor: StepExecutor = None):
        super().__init__(
            name="validation",
            description="Verify and validate pipeline outputs"
        )
        self.executor = executor
    
    def get_function_definition(self) -> FunctionDefinition:
        return VALIDATE_OUTPUT_FUNCTION
    
    def get_cot_prompt(self, context: PipelineContext) -> str:
        # Build summary of all previous outputs
        outputs_summary = []
        for name, result in context.step_results.items():
            if result.status == StepStatus.SUCCESS:
                outputs_summary.append(f"## {name}\n{result.reasoning}\nOutput: {result.output}")
        
        return f"""Review and validate the following pipeline outputs for consistency and correctness.

{chr(10).join(outputs_summary)}

Check for:
1. Logical consistency between steps
2. Any obvious errors or contradictions
3. Quality of the cleaning (was important content preserved?)
4. Correctness of domain and language classification"""
    
    def get_system_prompt(self) -> str:
        return """You are a quality assurance expert reviewing an NLP pipeline's outputs.

Your task is to verify that all outputs are consistent and correct. Be critical but fair.

Consider:
- Does the domain classification match the content type?
- Is the language detection consistent with the text?
- Was the text cleaning appropriate (not too aggressive)?"""
    
    def execute(self, context: PipelineContext) -> StepResult:
        """Execute validation check."""
        if not self.executor:
            # Do basic validation without LLM
            return self._basic_validation(context)
        
        return self.executor.execute_with_function(
            step_name=self.name,
            system_prompt=self.get_system_prompt(),
            user_prompt=self.get_cot_prompt(context),
            function_def=self.get_function_definition(),
            context=context
        )
    
    def _basic_validation(self, context: PipelineContext) -> StepResult:
        """Perform basic validation without LLM."""
        issues = []
        
        # Check if all expected steps ran
        expected_steps = ["text_cleaning", "language_detection"]
        for step in expected_steps:
            if step not in context.step_results:
                issues.append(f"Missing step: {step}")
            elif context.step_results[step].status != StepStatus.SUCCESS:
                issues.append(f"Step {step} did not succeed")
        
        # Check text cleaning produced output
        clean_result = context.get_step_output("text_cleaning")
        if clean_result:
            if clean_result.get("reduction_percent", 0) > 90:
                issues.append("Text cleaning removed >90% of content - may be too aggressive")
            if clean_result.get("cleaned_length", 0) < 10:
                issues.append("Cleaned text is very short - possible issue")
        
        is_valid = len(issues) == 0
        quality_score = 1.0 if is_valid else max(0.5, 1.0 - len(issues) * 0.2)
        
        return StepResult(
            step_name=self.name,
            status=StepStatus.SUCCESS if is_valid else StepStatus.NEEDS_VALIDATION,
            output={
                "is_valid": is_valid,
                "issues_found": issues,
                "corrections": {},
                "quality_score": quality_score,
                "reasoning": "Basic validation completed"
            },
            reasoning=f"Validation {'passed' if is_valid else 'found issues'}: {', '.join(issues) if issues else 'all checks passed'}",
            confidence=quality_score
        )
