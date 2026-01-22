"""
Text Summarization Module

Advanced summarization using LLMs with multiple strategies:
- Extractive: Select key sentences from source
- Abstractive: Generate new summary text
- Hybrid: Combine extractive + abstractive
- Hierarchical: Map-reduce for long documents
- Query-focused: Summarize based on specific questions

Features:
- Configurable length (bullet points, sentences, paragraphs)
- Hallucination detection and mitigation
- Multi-document summarization
- Key points extraction
- Chain-of-thought reasoning

Based on research best practices for production-grade summarization.
"""

import re
import time
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..cot import PipelineStep, PipelineContext, StepResult, StepStatus, FunctionDefinition
from ..cot.executor import StepExecutor
from ..utils.text_normalizer import to_text


class SummaryStyle(Enum):
    """Summary output styles."""
    BULLETS = "bullets"  # Bullet point list
    PARAGRAPH = "paragraph"  # Flowing paragraph
    EXECUTIVE = "executive"  # Executive summary format
    HEADLINES = "headlines"  # Key headlines only
    TLDR = "tldr"  # Very brief TL;DR


class SummaryStrategy(Enum):
    """Summarization strategies."""
    EXTRACTIVE = "extractive"  # Select key sentences
    ABSTRACTIVE = "abstractive"  # Generate new text
    HYBRID = "hybrid"  # Extract then abstract
    HIERARCHICAL = "hierarchical"  # Map-reduce for long docs


@dataclass
class SummaryResult:
    """Result of summarization."""
    summary: str
    strategy_used: str = "abstractive"
    style: str = "bullets"
    key_points: List[str] = field(default_factory=list)
    word_count: int = 0
    compression_ratio: float = 0.0
    confidence: float = 0.85
    reasoning: str = ""
    source_sentences: List[str] = field(default_factory=list)  # For extractive
    hallucination_check: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "strategy": self.strategy_used,
            "style": self.style,
            "key_points": self.key_points if self.key_points else None,
            "word_count": self.word_count,
            "compression_ratio": round(self.compression_ratio, 2),
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "hallucination_check": self.hallucination_check if self.hallucination_check else None
        }


# Function definition for CoT executor
SUMMARIZE_FUNCTION = FunctionDefinition(
    name="summarize_text",
    description="Generate a summary of the input text",
    parameters={
        "summary": {
            "type": "string",
            "description": "The generated summary"
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of key points extracted"
        },
        "reasoning": {
            "type": "string",
            "description": "Chain of thought reasoning for summarization"
        }
    },
    required=["summary", "key_points", "reasoning"]
)


class TextSummarizer:
    """
    Advanced Text Summarization System.
    
    Strategies:
    - EXTRACTIVE: Identifies and extracts key sentences
    - ABSTRACTIVE: Generates new summary using LLM
    - HYBRID: Extracts key info, then paraphrases
    - HIERARCHICAL: Map-reduce for long documents
    
    Styles:
    - BULLETS: Bullet point list
    - PARAGRAPH: Flowing narrative
    - EXECUTIVE: Formal executive summary
    - HEADLINES: Key headlines only
    - TLDR: Ultra-brief summary
    
    Usage:
        summarizer = TextSummarizer(executor)
        result = summarizer.summarize(
            "Long text...",
            strategy="abstractive",
            style="bullets",
            max_length=150
        )
        print(result.summary)
    """
    
    SYSTEM_PROMPT = """You are an expert summarizer who creates clear, accurate, and concise summaries.

## Your Goal
Create summaries that capture the essential information while being:
- Accurate: Only include information present in the source
- Concise: Remove redundancy while preserving meaning
- Clear: Easy to understand at a glance
- Complete: Cover all major points

## Important Rules
1. NEVER add information not in the source (no hallucination)
2. Preserve key facts, numbers, names, and dates
3. Maintain the original meaning and tone
4. Use active voice for clarity
5. Order points by importance

## Chain-of-Thought Process
1. Read the entire text carefully
2. Identify the main topic and purpose
3. Extract key facts, arguments, and conclusions
4. Determine what can be omitted
5. Synthesize into a coherent summary

## Hallucination Prevention
- Only summarize what is explicitly stated
- Do not infer or extrapolate
- When uncertain, be conservative
- Flag any assumptions made"""

    # Chunk size for hierarchical processing
    CHUNK_SIZE = 2000  # tokens (roughly 1500 words)
    MAX_SINGLE_PASS = 4000  # tokens for single-pass summary

    def __init__(
        self,
        executor: StepExecutor = None,
        default_strategy: str = "abstractive",
        default_style: str = "bullets",
        detect_hallucinations: bool = True
    ):
        """
        Initialize Text Summarizer.
        
        Args:
            executor: StepExecutor for LLM calls
            default_strategy: Default summarization strategy
            default_style: Default output style
            detect_hallucinations: Enable hallucination checking
        """
        self.executor = executor
        self.default_strategy = default_strategy
        self.default_style = default_style
        self.detect_hallucinations = detect_hallucinations
    
    def summarize(
        self,
        text: str,
        strategy: str = None,
        style: str = None,
        max_length: int = None,
        query: str = None,
        num_points: int = 5
    ) -> SummaryResult:
        """
        Summarize text using specified strategy.
        
        Args:
            text: Input text to summarize
            strategy: extractive, abstractive, hybrid, hierarchical
            style: bullets, paragraph, executive, headlines, tldr
            max_length: Maximum summary length in words
            query: Optional query for focused summarization
            num_points: Number of key points for bullet style
            
        Returns:
            SummaryResult with summary and metadata
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy
        style = style or self.default_style
        
        # Estimate token count
        word_count = len(text.split())
        token_estimate = int(word_count * 1.3)
        
        # Choose processing approach based on length
        if token_estimate > self.MAX_SINGLE_PASS and strategy != "extractive":
            result = self._hierarchical_summarize(text, style, max_length, query, num_points)
        elif strategy == "extractive":
            result = self._extractive_summarize(text, num_points)
        elif strategy == "hybrid":
            result = self._hybrid_summarize(text, style, max_length, num_points)
        else:
            result = self._abstractive_summarize(text, style, max_length, query, num_points)
        
        # Calculate compression ratio
        summary_words = len(result.summary.split())
        result.word_count = summary_words
        result.compression_ratio = summary_words / word_count if word_count > 0 else 0
        result.strategy_used = strategy
        result.style = style
        
        # Hallucination check for abstractive summaries
        if self.detect_hallucinations and strategy in ["abstractive", "hybrid"]:
            result.hallucination_check = self._check_hallucinations(text, result.summary)
        
        return result
    
    def _abstractive_summarize(
        self,
        text: str,
        style: str,
        max_length: int,
        query: str,
        num_points: int
    ) -> SummaryResult:
        """Generate abstractive summary using LLM."""
        if not self.executor:
            return self._fallback_extractive(text, num_points)
        
        # Build style-specific prompt
        style_instructions = self._get_style_instructions(style, max_length, num_points)
        query_instruction = f"\n\nFocus on answering: {query}" if query else ""
        
        user_prompt = f"""Summarize the following text.

## Style Requirements
{style_instructions}
{query_instruction}

## Text to Summarize
{text[:8000]}

## Output Format
SUMMARY:
[Your summary here]

KEY_POINTS:
1. [First key point]
2. [Second key point]
...

REASONING:
[Brief explanation of how you approached the summarization]

Generate the summary now:"""

        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.executor.client.chat(
                messages=messages,
                temperature=0.3
            )
            
            return self._parse_summary_response(response)
            
        except Exception as e:
            return self._fallback_extractive(text, num_points)
    
    def _extractive_summarize(self, text: str, num_points: int) -> SummaryResult:
        """Extract key sentences from text."""
        sentences = self._split_sentences(text)
        
        if not sentences:
            return SummaryResult(summary="", key_points=[], reasoning="No sentences found")
        
        # Score sentences by importance
        scored = self._score_sentences(sentences)
        
        # Select top sentences
        top_sentences = sorted(scored, key=lambda x: x[1], reverse=True)[:num_points]
        
        # Re-order by original position for coherence
        top_sentences.sort(key=lambda x: sentences.index(x[0]))
        
        selected = [s[0] for s in top_sentences]
        summary = " ".join(selected)
        
        return SummaryResult(
            summary=summary,
            key_points=selected,
            source_sentences=selected,
            reasoning=f"Extracted top {len(selected)} sentences by importance scoring",
            confidence=0.8
        )
    
    def _hybrid_summarize(
        self,
        text: str,
        style: str,
        max_length: int,
        num_points: int
    ) -> SummaryResult:
        """Extract key info, then paraphrase into summary."""
        # First: extract key sentences
        extractive = self._extractive_summarize(text, num_points * 2)
        
        if not self.executor:
            return extractive
        
        # Then: abstractively summarize the extracted content
        extracted_text = "\n".join(extractive.key_points)
        
        style_instructions = self._get_style_instructions(style, max_length, num_points)
        
        user_prompt = f"""Rewrite the following key points into a coherent {style} summary.

## Key Points Extracted
{extracted_text}

## Style Requirements
{style_instructions}

## Output
SUMMARY:
[Your rewritten summary]

KEY_POINTS:
[Refined key points]

REASONING:
[How you synthesized the information]

Generate:"""

        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.executor.client.chat(
                messages=messages,
                temperature=0.3
            )
            
            result = self._parse_summary_response(response)
            result.source_sentences = extractive.source_sentences
            return result
            
        except Exception:
            return extractive
    
    def _hierarchical_summarize(
        self,
        text: str,
        style: str,
        max_length: int,
        query: str,
        num_points: int
    ) -> SummaryResult:
        """Map-reduce summarization for long documents."""
        # Split into chunks
        chunks = self._split_into_chunks(text)
        
        if len(chunks) <= 1:
            return self._abstractive_summarize(text, style, max_length, query, num_points)
        
        # Map: Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            chunk_result = self._abstractive_summarize(
                chunk,
                style="paragraph",
                max_length=max_length // len(chunks) if max_length else 100,
                query=query,
                num_points=3
            )
            chunk_summaries.append(chunk_result.summary)
        
        # Reduce: Combine chunk summaries
        combined = "\n\n".join([f"Section {i+1}: {s}" for i, s in enumerate(chunk_summaries)])
        
        final_result = self._abstractive_summarize(
            combined,
            style=style,
            max_length=max_length,
            query=query,
            num_points=num_points
        )
        
        final_result.reasoning = f"Hierarchical: {len(chunks)} chunks → merged → final summary"
        return final_result
    
    def _get_style_instructions(self, style: str, max_length: int, num_points: int) -> str:
        """Get style-specific instructions."""
        length_note = f"\nMaximum length: {max_length} words" if max_length else ""
        
        style_map = {
            "bullets": f"""- Use bullet points (•)
- Provide exactly {num_points} key points
- Each point should be one clear sentence
- Start each with action verb or key fact{length_note}""",
            
            "paragraph": f"""- Write in flowing paragraph form
- Use clear topic sentences
- Maintain logical flow between ideas
- Keep it concise but complete{length_note}""",
            
            "executive": f"""- Start with one-sentence overview
- Follow with key findings/conclusions
- Include relevant metrics/numbers
- End with implications or recommendations
- Use formal, professional tone{length_note}""",
            
            "headlines": f"""- Provide {num_points} headline-style key points
- Each headline: 5-10 words maximum
- Focus on most newsworthy aspects
- Use active, impactful language{length_note}""",
            
            "tldr": """- Ultra-brief summary (1-2 sentences maximum)
- Capture only the absolute essential point
- Be direct and punchy"""
        }
        
        return style_map.get(style, style_map["bullets"])
    
    def _parse_summary_response(self, response: str) -> SummaryResult:
        """Parse LLM response into SummaryResult."""
        # Extract sections
        summary = self._extract_section(response, "SUMMARY")
        key_points_section = self._extract_section(response, "KEY_POINTS")
        reasoning = self._extract_section(response, "REASONING")
        
        # Parse key points
        key_points = []
        if key_points_section:
            lines = key_points_section.strip().split("\n")
            for line in lines:
                # Remove numbering and bullets
                cleaned = re.sub(r'^[\d\.\-\•\*]+\s*', '', line.strip())
                if cleaned:
                    key_points.append(cleaned)
        
        return SummaryResult(
            summary=summary or response[:500],
            key_points=key_points,
            reasoning=reasoning or "Summary generated",
            confidence=0.85
        )
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a section from response."""
        pattern = rf'{section_name}:\s*\n?(.*?)(?=\n[A-Z_]+:|$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
    
    def _score_sentences(self, sentences: List[str]) -> List[Tuple[str, float]]:
        """Score sentences by importance."""
        scored = []
        
        # Position score: first and last sentences often important
        total = len(sentences)
        
        for i, sentence in enumerate(sentences):
            score = 0.0
            
            # Position score
            if i == 0:
                score += 0.3  # First sentence bonus
            elif i < 3:
                score += 0.2  # Early sentences
            elif i >= total - 2:
                score += 0.15  # Conclusion sentences
            
            # Length score (favor medium-length sentences)
            words = len(sentence.split())
            if 15 <= words <= 40:
                score += 0.2
            elif 10 <= words <= 50:
                score += 0.1
            
            # Keyword indicators
            importance_signals = [
                'important', 'key', 'main', 'significant', 'critical',
                'conclusion', 'result', 'finding', 'therefore', 'thus',
                'in summary', 'overall', 'notably', 'primarily'
            ]
            for signal in importance_signals:
                if signal in sentence.lower():
                    score += 0.15
                    break
            
            # Numeric content (often factual)
            if re.search(r'\d+', sentence):
                score += 0.1
            
            # Named entities indicator (capitalized words)
            caps = len(re.findall(r'\b[A-Z][a-z]+\b', sentence))
            score += min(0.1, caps * 0.02)
            
            scored.append((sentence, score))
        
        return scored
    
    def _split_into_chunks(self, text: str, chunk_size: int = None) -> List[str]:
        """Split text into chunks for hierarchical processing."""
        chunk_size = chunk_size or self.CHUNK_SIZE
        words = text.split()
        
        # Estimate ~1.3 tokens per word
        words_per_chunk = int(chunk_size / 1.3)
        
        chunks = []
        for i in range(0, len(words), words_per_chunk):
            chunk = " ".join(words[i:i + words_per_chunk])
            chunks.append(chunk)
        
        return chunks
    
    def _check_hallucinations(self, source: str, summary: str) -> Dict[str, Any]:
        """Check summary for potential hallucinations."""
        source_lower = source.lower()
        
        # Extract claims from summary
        claims = self._split_sentences(summary)
        
        flagged = []
        for claim in claims:
            # Check for numbers/statistics
            numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', claim)
            for num in numbers:
                if num not in source:
                    flagged.append(f"Number '{num}' not found in source")
            
            # Check for proper nouns
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim)
            for noun in proper_nouns:
                if noun.lower() not in source_lower and len(noun) > 3:
                    # Allow common words
                    if noun.lower() not in ['the', 'this', 'that', 'these', 'those']:
                        flagged.append(f"Entity '{noun}' not found in source")
        
        return {
            "checked": True,
            "potential_issues": flagged[:5] if flagged else None,
            "is_clean": len(flagged) == 0,
            "confidence": 0.9 if len(flagged) == 0 else max(0.5, 0.9 - len(flagged) * 0.1)
        }
    
    def _fallback_extractive(self, text: str, num_points: int) -> SummaryResult:
        """Fallback to extractive when LLM unavailable."""
        return self._extractive_summarize(text, num_points)


class SummaryStep(PipelineStep):
    """
    Text Summarization Pipeline Step.
    
    Integrates with the CoT pipeline for text summarization
    with configurable strategy and style.
    """
    
    def __init__(
        self,
        executor: StepExecutor = None,
        strategy: str = "abstractive",
        style: str = "bullets",
        max_length: int = None,
        num_points: int = 5
    ):
        super().__init__(
            name="summary",
            description="Text summarization with key points extraction"
        )
        self.summarizer = TextSummarizer(executor=executor)
        self.strategy = strategy
        self.style = style
        self.max_length = max_length
        self.num_points = num_points
    
    def get_function_definition(self) -> FunctionDefinition:
        return SUMMARIZE_FUNCTION
    
    def get_cot_prompt(self, context: PipelineContext) -> str:
        return self.summarizer.SYSTEM_PROMPT
    
    def execute(self, context: PipelineContext) -> StepResult:
        """Execute summarization."""
        start_time = time.time()
        
        text = to_text(context.current_text or context.original_input)
        result = self.summarizer.summarize(
            text,
            strategy=self.strategy,
            style=self.style,
            max_length=self.max_length,
            num_points=self.num_points
        )
        
        return StepResult(
            step_name=self.name,
            status=StepStatus.SUCCESS,
            output=result.to_dict(),
            reasoning=result.reasoning,
            confidence=result.confidence,
            duration_ms=int((time.time() - start_time) * 1000)
        )


# ============== Quick-use functions ==============

def summarize_text(
    text: str,
    strategy: str = "abstractive",
    style: str = "bullets",
    max_length: int = None,
    api_key: str = None
) -> Dict[str, Any]:
    """
    Quick function for text summarization.
    
    Args:
        text: Input text
        strategy: extractive, abstractive, hybrid, hierarchical
        style: bullets, paragraph, executive, headlines, tldr
        max_length: Maximum summary length in words
        api_key: Optional Groq API key
        
    Returns:
        Dict with summary and metadata
    """
    from ..utils.groq_client import GroqClient
    from ..cot.executor import StepExecutor
    
    try:
        client = GroqClient(api_key=api_key)
        executor = StepExecutor(client)
        summarizer = TextSummarizer(executor=executor)
        return summarizer.summarize(
            text,
            strategy=strategy,
            style=style,
            max_length=max_length
        ).to_dict()
    except Exception as e:
        return {
            "summary": "",
            "error": str(e)
        }
