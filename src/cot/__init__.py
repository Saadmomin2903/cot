"""
Chain-of-Thought (CoT) Pipeline with Function Calling.

Modern implementation using:
- Structured function definitions with JSON schemas
- Chain-of-thought reasoning at each step
- Validation and feedback loops
- Self-consistency checks
- Modular step execution with context passing
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import json
from abc import ABC, abstractmethod


class StepStatus(Enum):
    """Status of a pipeline step execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_VALIDATION = "needs_validation"


@dataclass
class StepResult:
    """Result from a single pipeline step."""
    step_name: str
    status: StepStatus
    output: Dict[str, Any]
    reasoning: str = ""
    confidence: float = 1.0
    validation_notes: str = ""
    raw_response: str = ""
    tokens_used: int = 0
    duration_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "status": self.status.value,
            "output": self.output,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "validation_notes": self.validation_notes,
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms
        }


@dataclass
class PipelineContext:
    """
    Shared context passed between pipeline steps.
    
    Acts as memory/state for the chain-of-thought workflow,
    carrying forward intermediate results and reasoning.
    """
    original_input: str
    current_text: str = ""
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def get_step_output(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Get output from a previous step."""
        result = self.step_results.get(step_name)
        return result.output if result else None
    
    def add_result(self, result: StepResult):
        """Add a step result to context."""
        self.step_results[result.step_name] = result
    
    def get_chain_summary(self) -> str:
        """Get a summary of all steps for context passing."""
        summaries = []
        for name, result in self.step_results.items():
            if result.status == StepStatus.SUCCESS:
                summaries.append(f"- {name}: {result.reasoning}")
        return "\n".join(summaries)


@dataclass
class FunctionDefinition:
    """
    Definition of a function that the LLM can call.
    
    Uses JSON schema for structured output validation.
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = field(default_factory=list)
    
    def to_tool_schema(self) -> Dict[str, Any]:
        """Convert to Groq/OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required
                }
            }
        }


class PipelineStep(ABC):
    """
    Abstract base class for a chain-of-thought pipeline step.
    
    Each step:
    1. Receives context from previous steps
    2. Performs chain-of-thought reasoning
    3. Optionally calls functions/tools
    4. Returns structured output
    5. Can validate its own output
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.validators: List[Callable[[Dict], bool]] = []
    
    @abstractmethod
    def get_function_definition(self) -> FunctionDefinition:
        """Define the function schema for this step's output."""
        pass
    
    @abstractmethod
    def get_cot_prompt(self, context: PipelineContext) -> str:
        """
        Generate the chain-of-thought prompt for this step.
        
        Should include:
        - Clear task description
        - Context from previous steps
        - Step-by-step instructions
        - Expected output format
        """
        pass
    
    @abstractmethod
    def execute(self, context: PipelineContext) -> StepResult:
        """Execute this step and return result."""
        pass
    
    def validate(self, output: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate the step's output.
        
        Returns (is_valid, validation_notes)
        """
        for validator in self.validators:
            try:
                if not validator(output):
                    return False, f"Validator {validator.__name__} failed"
            except Exception as e:
                return False, f"Validation error: {str(e)}"
        return True, "All validations passed"
    
    def add_validator(self, validator: Callable[[Dict], bool]):
        """Add a validation function."""
        self.validators.append(validator)


# ============== Function Definitions for Each Step ==============

# Text Cleaning Function
TEXT_CLEAN_FUNCTION = FunctionDefinition(
    name="clean_text",
    description="Clean and normalize text by removing URLs, HTML, expanding contractions, and removing junk content",
    parameters={
        "cleaned_text": {
            "type": "string",
            "description": "The cleaned, normalized text"
        },
        "removed_elements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of types of elements that were removed (e.g., 'urls', 'html_tags', 'navigation')"
        },
        "original_length": {
            "type": "integer",
            "description": "Character count of original text"
        },
        "cleaned_length": {
            "type": "integer",
            "description": "Character count of cleaned text"
        },
        "reduction_percent": {
            "type": "number",
            "description": "Percentage of text reduced"
        }
    },
    required=["cleaned_text", "removed_elements", "original_length", "cleaned_length", "reduction_percent"]
)

# Domain Detection Function
DOMAIN_DETECT_FUNCTION = FunctionDefinition(
    name="detect_domain",
    description="Classify the text into one of three domains: technology, business, or general",
    parameters={
        "primary_domain": {
            "type": "string",
            "enum": ["technology", "business", "general"],
            "description": "The primary domain classification"
        },
        "confidence": {
            "type": "number",
            "description": "Confidence score between 0 and 1"
        },
        "domain_scores": {
            "type": "object",
            "properties": {
                "technology": {"type": "number"},
                "business": {"type": "number"},
                "general": {"type": "number"}
            },
            "description": "Probability scores for each domain (should sum to 1)"
        },
        "sub_categories": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-3 relevant sub-categories"
        },
        "reasoning": {
            "type": "string",
            "description": "Step-by-step reasoning for the classification"
        }
    },
    required=["primary_domain", "confidence", "domain_scores", "sub_categories", "reasoning"]
)

# Language Detection Function
LANGUAGE_DETECT_FUNCTION = FunctionDefinition(
    name="detect_language",
    description="Detect the language and script type of the text",
    parameters={
        "language_code": {
            "type": "string",
            "description": "ISO 639-1 language code (e.g., 'en', 'fr', 'de')"
        },
        "language_name": {
            "type": "string",
            "description": "Full language name (e.g., 'English', 'French')"
        },
        "script_type": {
            "type": "string",
            "enum": ["roman", "non_roman", "mixed"],
            "description": "Whether the text uses Roman alphabet, non-Roman, or mixed"
        },
        "confidence": {
            "type": "number",
            "description": "Confidence score between 0 and 1"
        },
        "detected_scripts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of detected scripts (e.g., 'Latin', 'Cyrillic', 'Devanagari')"
        }
    },
    required=["language_code", "language_name", "script_type", "confidence"]
)

# Validation/Verification Function
VALIDATE_OUTPUT_FUNCTION = FunctionDefinition(
    name="validate_pipeline_output",
    description="Verify and validate the outputs from previous pipeline steps",
    parameters={
        "is_valid": {
            "type": "boolean",
            "description": "Whether all outputs are valid and consistent"
        },
        "issues_found": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of any issues or inconsistencies found"
        },
        "corrections": {
            "type": "object",
            "description": "Any corrections to apply to previous outputs"
        },
        "quality_score": {
            "type": "number",
            "description": "Overall quality score from 0 to 1"
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation of the validation process and findings"
        }
    },
    required=["is_valid", "issues_found", "quality_score", "reasoning"]
)
