"""
Semantic Text Cleaner - LLM-powered intelligent text cleaning.

Uses LLM to:
1. Identify and preserve semantically important content
2. Remove noise while keeping context
3. Handle edge cases that rule-based cleaners miss
4. Broaden abbreviated or truncated semantic words
"""

from typing import Dict, Any, List, Optional
import time
import re

from ..cot import PipelineStep, PipelineContext, StepResult, StepStatus, FunctionDefinition
from ..cot.executor import StepExecutor
from ..serax import SeraxSchema, FieldDefinition, FieldType, SeraxParser, SeraxFormatter, SeraxDelimiters


# Schema for semantic cleaning output
SEMANTIC_CLEAN_SCHEMA = SeraxSchema(
    name="semantic_cleaning",
    description="Intelligent text cleaning with semantic preservation",
    fields=[
        FieldDefinition(
            "cleaned_text", FieldType.STRING,
            description="Cleaned text with important semantic content preserved"
        ),
        FieldDefinition(
            "expanded_terms", FieldType.LIST,
            required=False,
            description="Terms that were expanded or clarified"
        ),
        FieldDefinition(
            "preserved_entities", FieldType.LIST,
            required=False,
            description="Important entities that were preserved"
        ),
        FieldDefinition(
            "removed_items", FieldType.LIST,
            required=False,
            description="Types of noise that were removed"
        ),
        FieldDefinition(
            "reasoning", FieldType.STRING,
            description="Explanation of cleaning decisions"
        ),
    ]
)


# Function definition for CoT executor
SEMANTIC_CLEAN_FUNCTION = FunctionDefinition(
    name="semantic_clean",
    description="Intelligently clean text while preserving semantic meaning",
    parameters={
        "cleaned_text": {
            "type": "string",
            "description": "The cleaned text with important content preserved"
        },
        "expanded_terms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Terms that were expanded (e.g., 'AI' -> 'Artificial Intelligence')"
        },
        "preserved_entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Important entities/names that were kept"
        },
        "removed_items": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Categories of noise removed (e.g., 'navigation', 'boilerplate')"
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of cleaning decisions"
        }
    },
    required=["cleaned_text", "reasoning"]
)


class SemanticCleaner:
    """
    LLM-powered semantic text cleaner.
    
    Unlike rule-based cleaners, this:
    - Understands context and meaning
    - Preserves important entities and concepts
    - Expands abbreviations and acronyms intelligently
    - Handles domain-specific terminology
    - Removes only true noise (boilerplate, navigation, ads)
    
    Example:
        cleaner = SemanticCleaner(executor)
        result = cleaner.clean(text)
        print(result["cleaned_text"])
        print(result["expanded_terms"])  # ["AI" -> "Artificial Intelligence", ...]
    """
    
    SYSTEM_PROMPT = """You are an expert text cleaner specializing in preserving semantic meaning while removing noise.

## Your Task
Clean the input text while:
1. PRESERVING all important semantic content (facts, entities, relationships)
2. EXPANDING common abbreviations and acronyms for clarity
3. REMOVING only true noise (navigation, boilerplate, ads, legal footers)
4. KEEPING technical terms and domain-specific language
5. MAINTAINING the original meaning and context

## What to PRESERVE
- Named entities (people, companies, products, locations)
- Dates, numbers, statistics
- Technical terms and jargon (they carry meaning!)
- Key verbs and action words
- Relationships and connections between entities

## What to REMOVE
- Cookie notices, privacy banners
- Navigation menus (Home, About, Contact)
- Social media buttons/text
- Repetitive boilerplate
- Empty placeholders
- Comment/reply prompts

## What to EXPAND (when helpful)
- Common abbreviations: AI → Artificial Intelligence, ML → Machine Learning
- Acronyms in context: CEO → Chief Executive Officer (first occurrence)
- Truncated words if meaning is clear

## Important Rules
- When in doubt, PRESERVE the content
- Don't remove short but meaningful content
- Keep numbers and statistics
- Maintain proper nouns exactly as written
- Don't over-expand obvious terms
- **DO NOT TRANSLATE**: Keep the text in its original language (unless it matches noise patterns)"""
    
    def __init__(self, executor: StepExecutor = None, use_serax: bool = True):
        """Initialize semantic cleaner.
        
        Args:
            executor: StepExecutor for LLM calls
            use_serax: Use SERAX format for output (more reliable)
        """
        self.executor = executor
        self.use_serax = use_serax
        self.parser = SeraxParser(SEMANTIC_CLEAN_SCHEMA) if use_serax else None
    
    def clean(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Clean text using LLM with semantic understanding.
        
        Args:
            text: Input text to clean
            context: Optional context (e.g., domain, source type)
            
        Returns:
            Dict with cleaned_text, expanded_terms, etc.
        """
        if not self.executor:
            # Fallback to basic result if no executor
            return {
                "cleaned_text": text,
                "expanded_terms": [],
                "preserved_entities": [],
                "removed_items": [],
                "reasoning": "No LLM executor available - using original text",
                "stats": {
                    "original_length": len(text),
                    "cleaned_length": len(text),
                    "reduction_percent": 0.0
                }
            }
        
        start_time = time.time()
        
        # Build user prompt
        user_prompt = self._build_prompt(text, context)
        
        if self.use_serax:
            return self._clean_with_serax(text, user_prompt, start_time)
        else:
            return self._clean_with_json(text, user_prompt, start_time)
    
    def _build_prompt(self, text: str, context: Dict[str, Any] = None) -> str:
        """Build the user prompt for semantic cleaning."""
        parts = ["Clean the following text semantically:"]
        
        if context:
            if context.get("domain"):
                parts.append(f"\nDomain: {context['domain']}")
            if context.get("source"):
                parts.append(f"Source: {context['source']}")
        
        # Truncate if too long
        max_len = 4000
        display_text = text[:max_len] + ("..." if len(text) > max_len else "")
        
        parts.append(f"\n## Input Text\n{display_text}")
        
        if self.use_serax:
            parts.append(f"\n## Output Format (SERAX)")
            parts.append(SEMANTIC_CLEAN_SCHEMA.to_prompt_format())
        
        return "\n".join(parts)
    
    def _clean_with_serax(
        self, 
        original_text: str, 
        user_prompt: str, 
        start_time: float
    ) -> Dict[str, Any]:
        """Clean using SERAX format output."""
        try:
            # Call LLM
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.executor.client.chat(
                messages=messages,
                temperature=0.1
            )
            
            # Parse SERAX response
            parsed = self.parser.parse(response)
            
            cleaned_text = parsed.get("cleaned_text", original_text)
            
            return {
                "cleaned_text": cleaned_text,
                "expanded_terms": parsed.get("expanded_terms", []),
                "preserved_entities": parsed.get("preserved_entities", []),
                "removed_items": parsed.get("removed_items", []),
                "reasoning": parsed.get("reasoning", ""),
                "stats": {
                    "original_length": len(original_text),
                    "cleaned_length": len(cleaned_text),
                    "reduction_percent": round(
                        (1 - len(cleaned_text) / len(original_text)) * 100, 2
                    ) if original_text else 0
                },
                "duration_ms": int((time.time() - start_time) * 1000)
            }
            
        except Exception as e:
            return {
                "cleaned_text": original_text,
                "error": str(e),
                "expanded_terms": [],
                "preserved_entities": [],
                "removed_items": [],
                "reasoning": f"LLM error: {str(e)}",
                "stats": {
                    "original_length": len(original_text),
                    "cleaned_length": len(original_text),
                    "reduction_percent": 0.0
                },
                "duration_ms": int((time.time() - start_time) * 1000)
            }
    
    def _clean_with_json(
        self, 
        original_text: str, 
        user_prompt: str, 
        start_time: float
    ) -> Dict[str, Any]:
        """Clean using JSON format output."""
        try:
            result = self.executor.client.chat_json(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1
            )
            
            cleaned_text = result.get("cleaned_text", original_text)
            
            return {
                "cleaned_text": cleaned_text,
                "expanded_terms": result.get("expanded_terms", []),
                "preserved_entities": result.get("preserved_entities", []),
                "removed_items": result.get("removed_items", []),
                "reasoning": result.get("reasoning", ""),
                "stats": {
                    "original_length": len(original_text),
                    "cleaned_length": len(cleaned_text),
                    "reduction_percent": round(
                        (1 - len(cleaned_text) / len(original_text)) * 100, 2
                    ) if original_text else 0
                },
                "duration_ms": int((time.time() - start_time) * 1000)
            }
            
        except Exception as e:
            return {
                "cleaned_text": original_text,
                "error": str(e),
                "duration_ms": int((time.time() - start_time) * 1000)
            }


class SemanticCleaningStep(PipelineStep):
    """
    Pipeline step for semantic cleaning.
    
    Integrates with the CoT pipeline to provide intelligent
    LLM-based text cleaning after rule-based cleaning.
    """
    
    def __init__(self, executor: StepExecutor = None, use_serax: bool = True):
        super().__init__(
            name="semantic_cleaning",
            description="LLM-powered intelligent text cleaning"
        )
        self.cleaner = SemanticCleaner(executor, use_serax)
        self.executor = executor
    
    def get_function_definition(self) -> FunctionDefinition:
        return SEMANTIC_CLEAN_FUNCTION
    
    def get_cot_prompt(self, context: PipelineContext) -> str:
        return SemanticCleaner.SYSTEM_PROMPT
    
    def execute(self, context: PipelineContext) -> StepResult:
        """Execute semantic cleaning."""
        start_time = time.time()
        
        # Get text from previous cleaning step if available
        text = context.current_text or context.original_input
        
        # Get domain context if available
        clean_context = {}
        domain_result = context.get_step_output("domain_detection")
        if domain_result:
            clean_context["domain"] = domain_result.get("primary_domain", "general")
        
        # Run semantic cleaning
        result = self.cleaner.clean(text, clean_context)
        
        if "error" in result:
            return StepResult(
                step_name=self.name,
                status=StepStatus.FAILED,
                output=result,
                reasoning=result.get("reasoning", ""),
                confidence=0.0,
                duration_ms=result.get("duration_ms", 0)
            )
        
        # Update context with cleaned text
        context.current_text = result["cleaned_text"]
        
        return StepResult(
            step_name=self.name,
            status=StepStatus.SUCCESS,
            output={
                "cleaned_text": result["cleaned_text"],
                "expanded_terms": result.get("expanded_terms", []),
                "preserved_entities": result.get("preserved_entities", []),
                "removed_items": result.get("removed_items", []),
                "original_length": result["stats"]["original_length"],
                "cleaned_length": result["stats"]["cleaned_length"],
                "reduction_percent": result["stats"]["reduction_percent"]
            },
            reasoning=result.get("reasoning", "Semantic cleaning completed"),
            confidence=0.9,
            duration_ms=result.get("duration_ms", int((time.time() - start_time) * 1000))
        )


# ============== Quick-use functions ==============

def semantic_clean(
    text: str,
    api_key: str = None,
    use_serax: bool = True
) -> Dict[str, Any]:
    """
    Quick function for semantic text cleaning.
    
    Args:
        text: Input text to clean
        api_key: Optional Groq API key
        use_serax: Use SERAX format (more reliable)
        
    Returns:
        Dict with cleaned_text and metadata
    """
    from ..utils.groq_client import GroqClient
    from ..cot.executor import StepExecutor
    
    try:
        client = GroqClient(api_key=api_key)
        executor = StepExecutor(client)
        cleaner = SemanticCleaner(executor, use_serax)
        return cleaner.clean(text)
    except Exception as e:
        return {
            "cleaned_text": text,
            "error": str(e),
            "stats": {
                "original_length": len(text),
                "cleaned_length": len(text),
                "reduction_percent": 0.0
            }
        }
