"""
Translation Module

Translates text from any detected language to English using LLM.

Features:
- Auto-detect source language
- Preserve formatting and structure
- Handle technical terms and proper nouns
- Chunk-based translation for long texts
- Quality scoring for translations

Based on research for LLM-based translation at scale.
"""

import re
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from ..cot import PipelineStep, PipelineContext, StepResult, StepStatus, FunctionDefinition
from ..cot.executor import StepExecutor


@dataclass
class TranslationResult:
    """Result of translation."""
    original_text: str
    translated_text: str
    source_language: str = "unknown"
    target_language: str = "English"
    confidence: float = 0.85
    preserved_terms: List[str] = field(default_factory=list)
    reasoning: str = ""
    was_already_english: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "translated_text": self.translated_text,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "confidence": self.confidence,
            "preserved_terms": self.preserved_terms if self.preserved_terms else None,
            "was_already_english": self.was_already_english,
            "reasoning": self.reasoning
        }


# Function definition for CoT executor
TRANSLATE_FUNCTION = FunctionDefinition(
    name="translate_to_english",
    description="Translate text from any language to English",
    parameters={
        "translated_text": {
            "type": "string",
            "description": "The translated English text"
        },
        "source_language": {
            "type": "string",
            "description": "Detected source language"
        },
        "preserved_terms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Technical terms and proper nouns preserved as-is"
        },
        "reasoning": {
            "type": "string",
            "description": "Translation approach and decisions"
        }
    },
    required=["translated_text", "source_language", "reasoning"]
)


class TextTranslator:
    """
    Translation System for converting any language to English.
    
    Features:
    - Auto-detect source language
    - Preserve proper nouns and technical terms
    - Handle formatting (lists, paragraphs)
    - Chunk-based processing for long texts
    - Quality confidence scoring
    
    Usage:
        translator = TextTranslator(executor)
        result = translator.translate("Bonjour le monde!")
        print(result.translated_text)  # "Hello world!"
    """
    
    SYSTEM_PROMPT = """You are an expert translator who translates text to English accurately and naturally.

## Your Goal
Translate the input text to English while:
- Preserving the original meaning and tone
- Keeping proper nouns unchanged (names, places, brands)
- Maintaining technical terminology where appropriate
- Preserving formatting (lists, paragraphs, structure)

## Important Rules
1. Translate naturally, not word-for-word
2. Keep proper nouns in their original form
3. Preserve numbers, dates, and measurements
4. Maintain the document structure
5. Technical terms can stay in original or be translated with note

## Quality Standards
- Fluent, natural English
- Accurate meaning preservation
- Consistent terminology
- Cultural context adaptation where needed

## Chain-of-Thought Process
1. Identify the source language
2. Analyze the text structure and content
3. Identify proper nouns and technical terms to preserve
4. Translate segment by segment
5. Review for fluency and accuracy"""

    # Chunk size for long text translation
    MAX_CHUNK_SIZE = 3000  # characters

    def __init__(
        self,
        executor: StepExecutor = None,
        preserve_formatting: bool = True
    ):
        """
        Initialize Text Translator.
        
        Args:
            executor: StepExecutor for LLM calls
            preserve_formatting: Keep original text structure
        """
        self.executor = executor
        self.preserve_formatting = preserve_formatting
    
    def translate(
        self,
        text: str,
        source_language: str = None
    ) -> TranslationResult:
        """
        Translate text to English.
        
        Args:
            text: Input text in any language
            source_language: Optional known source language
            
        Returns:
            TranslationResult with translated text
        """
        start_time = time.time()
        
        # Check if already English (simple heuristic)
        if self._is_likely_english(text):
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language="English",
                was_already_english=True,
                confidence=1.0,
                reasoning="Text is already in English, no translation needed"
            )
        
        if not self.executor:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language="unknown",
                confidence=0.0,
                reasoning="No LLM executor available for translation"
            )
        
        # Handle long texts with chunking
        if len(text) > self.MAX_CHUNK_SIZE:
            return self._translate_chunks(text, source_language)
        
        return self._translate_text(text, source_language)
    
    def _translate_text(
        self,
        text: str,
        source_language: str = None
    ) -> TranslationResult:
        """Translate text using LLM."""
        lang_hint = f"The source language is {source_language}. " if source_language else ""
        
        user_prompt = f"""Translate the following text to English.

{lang_hint}

## Text to Translate
{text}

## Output Format
SOURCE_LANGUAGE: [detected language]
PRESERVED_TERMS: [comma-separated list of proper nouns/terms kept as-is]
TRANSLATION:
[Your English translation here]

REASONING:
[Brief explanation of translation decisions]

Translate now:"""

        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.executor.client.chat(
                messages=messages,
                temperature=0.3
            )
            
            return self._parse_translation_response(text, response)
            
        except Exception as e:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language="unknown",
                confidence=0.0,
                reasoning=f"Translation failed: {str(e)}"
            )
    
    def _translate_chunks(
        self,
        text: str,
        source_language: str = None
    ) -> TranslationResult:
        """Translate long text in chunks."""
        chunks = self._split_into_chunks(text)
        translated_chunks = []
        detected_language = source_language
        all_preserved = []
        
        for chunk in chunks:
            result = self._translate_text(chunk, detected_language)
            translated_chunks.append(result.translated_text)
            
            if not detected_language:
                detected_language = result.source_language
            
            if result.preserved_terms:
                all_preserved.extend(result.preserved_terms)
        
        combined = "\n\n".join(translated_chunks)
        
        return TranslationResult(
            original_text=text,
            translated_text=combined,
            source_language=detected_language or "unknown",
            confidence=0.85,
            preserved_terms=list(set(all_preserved)),
            reasoning=f"Translated in {len(chunks)} chunks"
        )
    
    def _parse_translation_response(
        self,
        original: str,
        response: str
    ) -> TranslationResult:
        """Parse LLM response into TranslationResult."""
        # Extract fields
        source_lang = self._extract_field(response, "SOURCE_LANGUAGE", "unknown")
        preserved_str = self._extract_field(response, "PRESERVED_TERMS", "")
        translation = self._extract_section(response, "TRANSLATION")
        reasoning = self._extract_field(response, "REASONING", "")
        
        # Parse preserved terms
        preserved = [t.strip() for t in preserved_str.split(",")] if preserved_str else []
        preserved = [t for t in preserved if t and t.lower() not in ['none', 'n/a', '']]
        
        # Use full response if parsing failed
        if not translation:
            translation = response
        
        return TranslationResult(
            original_text=original,
            translated_text=translation.strip(),
            source_language=source_lang,
            confidence=0.85,
            preserved_terms=preserved,
            reasoning=reasoning
        )
    
    def _extract_field(self, text: str, field: str, default: str = "") -> str:
        """Extract a field value."""
        pattern = rf'{field}:\s*(.+?)(?:\n|$)'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else default
    
    def _extract_section(self, text: str, section: str) -> str:
        """Extract a section from response."""
        pattern = rf'{section}:\s*\n?(.*?)(?=\n[A-Z_]+:|$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks for translation."""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < self.MAX_CHUNK_SIZE:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _is_likely_english(self, text: str) -> bool:
        """Heuristic check if text is likely English."""
        # Common English words
        english_words = [
            'the', 'is', 'are', 'was', 'were', 'have', 'has', 'been',
            'will', 'would', 'could', 'should', 'can', 'may', 'might',
            'this', 'that', 'these', 'those', 'with', 'from', 'about',
            'which', 'when', 'where', 'what', 'who', 'how', 'why'
        ]
        
        words = text.lower().split()
        if len(words) < 5:
            return False
        
        # Count English word matches
        english_count = sum(1 for w in words[:50] if w in english_words)
        
        # If >15% are common English words, likely English
        return english_count / min(len(words), 50) > 0.15


class TranslationStep(PipelineStep):
    """
    Translation Pipeline Step.
    
    Translates text to English for further processing.
    """
    
    def __init__(self, executor: StepExecutor = None):
        super().__init__(
            name="translation",
            description="Translate text to English"
        )
        self.translator = TextTranslator(executor=executor)
    
    def get_function_definition(self) -> FunctionDefinition:
        return TRANSLATE_FUNCTION
    
    def get_cot_prompt(self, context: PipelineContext) -> str:
        return self.translator.SYSTEM_PROMPT
    
    def execute(self, context: PipelineContext) -> StepResult:
        """Execute translation."""
        start_time = time.time()
        
        text = context.current_text or context.original_input
        result = self.translator.translate(text)
        
        # Update context with translated text
        if not result.was_already_english:
            context.current_text = result.translated_text
        
        return StepResult(
            step_name=self.name,
            status=StepStatus.SUCCESS,
            output=result.to_dict(),
            reasoning=result.reasoning,
            confidence=result.confidence,
            duration_ms=int((time.time() - start_time) * 1000)
        )
