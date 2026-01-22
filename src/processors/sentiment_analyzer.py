"""
Sentiment Analysis Module

Advanced sentiment analysis using:
- LLM-based analysis for nuanced understanding
- Aspect-based sentiment (per entity/topic)
- Multi-class: positive, negative, neutral, mixed
- Special category: anti-national detection
- Emotion detection: joy, anger, fear, sadness, etc.
- Confidence scoring

Based on research best practices:
- Chain-of-thought reasoning for complex cases
- Few-shot examples for consistency
- Aspect extraction before sentiment assignment
"""

import re
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..cot import PipelineStep, PipelineContext, StepResult, StepStatus, FunctionDefinition
from ..cot.executor import StepExecutor


class SentimentClass(Enum):
    """Sentiment classifications."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    ANTI_NATIONAL = "anti_national"  # Special category for concerning content


class EmotionClass(Enum):
    """Emotion classifications (fine-grained)."""
    JOY = "joy"
    ANGER = "anger"
    FEAR = "fear"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    overall_sentiment: str  # positive, negative, neutral, mixed
    confidence: float
    scores: Dict[str, float]  # Score for each sentiment class
    primary_emotion: str = "neutral"
    emotions: Dict[str, float] = field(default_factory=dict)
    aspects: List[Dict[str, Any]] = field(default_factory=list)  # Aspect-based sentiments
    is_concerning: bool = False  # Anti-national or harmful content flag
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentiment": self.overall_sentiment,
            "confidence": self.confidence,
            "scores": self.scores,
            "emotion": self.primary_emotion,
            "emotions": self.emotions if self.emotions else None,
            "aspects": self.aspects if self.aspects else None,
            "is_concerning": self.is_concerning,
            "reasoning": self.reasoning
        }


# Function definition for CoT executor
SENTIMENT_FUNCTION = FunctionDefinition(
    name="analyze_sentiment",
    description="Analyze text sentiment with aspect-based details and anti-national detection",
    parameters={
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral", "mixed", "anti_national"],
            "description": "Overall sentiment"
        },
        "confidence": {
            "type": "number",
            "description": "Confidence score 0-1"
        },
        "scores": {
            "type": "object",
            "properties": {
                "positive": {"type": "number", "description": "Positive sentiment score (0-1)"},
                "negative": {"type": "number", "description": "Negative sentiment score (0-1)"},
                "neutral": {"type": "number", "description": "Neutral sentiment score (0-1)"},
                "anti_national": {"type": "number", "description": "Anti-national sentiment score (0-1)"}
            },
            "description": "Scores for each sentiment category (should sum to ~1.0)"
        },
        "emotion": {
            "type": "string",
            "enum": ["joy", "anger", "fear", "sadness", "surprise", "disgust", "trust", "anticipation", "neutral"],
            "description": "Primary emotion"
        },
        "aspects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {"type": "string"},
                    "sentiment": {"type": "string"},
                    "confidence": {"type": "number"}
                }
            },
            "description": "Sentiment per aspect/entity"
        },
        "reasoning": {
            "type": "string",
            "description": "Chain of thought reasoning"
        }
    },
    required=["sentiment", "confidence", "scores", "reasoning"]
)


class SentimentAnalyzer:
    """
    Advanced Sentiment Analysis System.
    
    Features:
    - Overall sentiment classification (pos/neg/neutral/mixed/anti-national)
    - Aspect-based sentiment analysis
    - Emotion detection (8 primary emotions)
    - Anti-national/concerning content detection
    - Chain-of-thought reasoning
    - Confidence scoring
    
    Based on research best practices:
    - Uses LLM for nuanced understanding
    - Provides aspect-level granularity
    - Detects concerning content patterns
    
    Usage:
        analyzer = SentimentAnalyzer(executor)
        result = analyzer.analyze("I love the UI but the pricing is terrible")
        print(result.overall_sentiment)  # "mixed"
        print(result.aspects)  # [{"aspect": "UI", "sentiment": "positive"}, ...]
    """
    
    SYSTEM_PROMPT = """You are an expert sentiment analyst specializing in nuanced text understanding and anti-national content detection.

## Your Task
Analyze the sentiment of the input text comprehensively, including detection of anti-national content.

## Sentiment Classes
- positive: Overall positive tone, approval, satisfaction, happiness
- negative: Overall negative tone, disapproval, dissatisfaction, unhappiness
- neutral: Factual, objective, no clear emotional tone
- mixed: Contains both positive and negative sentiments
- anti_national: Content acting against national interests, promoting separatism, incitement, or hatred toward nation/people

## Emotion Classes (fine-grained)
- joy: happiness, contentment, pleasure
- anger: frustration, irritation, rage
- fear: worry, anxiety, concern
- sadness: disappointment, grief, melancholy
- surprise: shock, amazement, unexpectedness
- disgust: revulsion, contempt, aversion
- trust: confidence, faith, reliability
- anticipation: expectation, hope, excitement

## Special Detection: Anti-National Content
Detect and score anti-national content that shows:
- Hatred toward a nation or its people
- Incitement of violence against national institutions
- Promotion of terrorism, separatism, or sedition
- Blasphemy or hate speech against national symbols
- Deliberate misinformation against national interest
- Calls to destroy or overthrow constitutional institutions
- Anti-national slogans or chants

IMPORTANT: Anti-national is a SEPARATE score category. Provide scores for:
- positive (0.0-1.0)
- negative (0.0-1.0)
- neutral (0.0-1.0)
- anti_national (0.0-1.0)

Scores should sum to approximately 1.0. If anti-national content is detected, assign a score (0.0-1.0) to anti_national.

## Chain-of-Thought Analysis
1. Read the text carefully
2. Identify overall tone
3. Detect emotions expressed
4. Find aspect-specific sentiments
5. Check for anti-national patterns (separately)
6. Assign scores to all four categories (positive, negative, neutral, anti_national)
7. Assign confidence based on clarity

## Output Format
SENTIMENT: [positive/negative/neutral/mixed/anti_national]
CONFIDENCE: [0.0-1.0]
SCORES: positive=[0.0-1.0], negative=[0.0-1.0], neutral=[0.0-1.0], anti_national=[0.0-1.0]
EMOTION: [primary emotion]
ASPECTS: [aspect1:sentiment, aspect2:sentiment, ...]
REASONING: [your analysis]"""

    # Words indicating different sentiments
    POSITIVE_INDICATORS = [
        'love', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'happy', 'pleased', 'satisfied', 'proud', 'excited', 'brilliant'
    ]
    
    NEGATIVE_INDICATORS = [
        'hate', 'terrible', 'awful', 'horrible', 'bad', 'poor',
        'disappointed', 'frustrated', 'angry', 'sad', 'worst', 'failed'
    ]
    
    CONCERNING_PATTERNS = [
        r'\b(?:death\s+to|destroy|kill\s+all)\b',
        r'\b(?:hate|destroy)\s+(?:the\s+)?(?:country|nation|government)\b',
        r'\b(?:terrorist|terrorism|separatist)\b',
    ]

    def __init__(
        self,
        executor: StepExecutor = None,
        detect_aspects: bool = True,
        detect_emotions: bool = True,
        detect_concerning: bool = True
    ):
        """
        Initialize Sentiment Analyzer.
        
        Args:
            executor: StepExecutor for LLM calls
            detect_aspects: Enable aspect-based sentiment
            detect_emotions: Enable emotion detection
            detect_concerning: Enable anti-national content detection
        """
        self.executor = executor
        self.detect_aspects = detect_aspects
        self.detect_emotions = detect_emotions
        self.detect_concerning = detect_concerning
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            SentimentResult with detailed analysis
        """
        start_time = time.time()
        
        if not self.executor:
            return self._fallback_analysis(text)
        
        # Use LLM for comprehensive analysis
        return self._analyze_with_llm(text)
    
    def _analyze_with_llm(self, text: str) -> SentimentResult:
        """Analyze using LLM with chain-of-thought and structured output."""
        user_prompt = f"""Analyze the sentiment of the following text:

## Text
{text[:4000]}

## Analysis Requirements
1. Overall sentiment (positive/negative/neutral/mixed/anti_national)
2. Confidence level (0.0 to 1.0)
3. Scores for ALL categories:
   - positive: 0.0-1.0
   - negative: 0.0-1.0
   - neutral: 0.0-1.0
   - anti_national: 0.0-1.0 (score if anti-national content detected)
4. Primary emotion detected
5. Aspect-based sentiments (if multiple topics/entities)
6. Provide reasoning for your analysis

IMPORTANT: Always provide scores for all four categories. If anti-national content is detected, assign a score to anti_national (0.0 if none detected).

Analyze now:"""

        try:
            # Try using structured function calling if available
            from ..cot import PipelineContext
            
            context = PipelineContext(original_input=text, current_text=text)
            
            # Use execute_with_function for structured output
            result = self.executor.execute_with_function(
                step_name="sentiment_analysis",
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                function_def=SENTIMENT_FUNCTION,
                context=context,
                temperature=0.2
            )
            
            if result.status == StepStatus.SUCCESS:
                output = result.output
                # Extract scores from structured output
                scores = output.get("scores", {})
                # Ensure all four scores are present
                if "anti_national" not in scores:
                    scores["anti_national"] = 0.0
                if "positive" not in scores:
                    scores["positive"] = 0.0
                if "negative" not in scores:
                    scores["negative"] = 0.0
                if "neutral" not in scores:
                    scores["neutral"] = 0.0
                
                # Normalize scores to sum to 1.0
                total = sum(scores.values())
                if total > 0:
                    scores = {k: v / total for k, v in scores.items()}
                
                return SentimentResult(
                    overall_sentiment=output.get("sentiment", "neutral"),
                    confidence=output.get("confidence", 0.5),
                    scores=scores,
                    primary_emotion=output.get("emotion", "neutral"),
                    aspects=output.get("aspects", []),
                    is_concerning=scores.get("anti_national", 0.0) > 0.3,
                    reasoning=output.get("reasoning", "")
                )
            else:
                # Fallback to text parsing if structured output failed
                return self._fallback_analysis(text)
            
        except Exception as e:
            # Fallback to manual parsing if function calling not available
            try:
                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
                
                response = self.executor.client.chat(
                    messages=messages,
                    temperature=0.2
                )
                
                return self._parse_result(response)
            except:
                return self._fallback_analysis(text)
    
    def _parse_result(self, response: str) -> SentimentResult:
        """Parse LLM response into SentimentResult."""
        # Extract fields
        sentiment = self._extract_field(response, 'SENTIMENT', 'neutral').lower()
        confidence_str = self._extract_field(response, 'CONFIDENCE', '0.5')
        scores_str = self._extract_field(response, 'SCORES', '')
        emotion = self._extract_field(response, 'EMOTION', 'neutral').lower()
        aspects_str = self._extract_field(response, 'ASPECTS', '')
        reasoning = self._extract_field(response, 'REASONING', '')
        
        # Parse confidence
        try:
            confidence = float(re.search(r'[\d.]+', confidence_str).group())
            confidence = min(1.0, max(0.0, confidence))
        except:
            confidence = 0.5
        
        # Parse scores from SCORES field
        scores = self._parse_scores(scores_str, sentiment, confidence)
        
        # Parse aspects
        aspects = []
        if aspects_str:
            aspect_matches = re.findall(r'([^:,]+):\s*(\w+)(?:\s*\(([0-9.]+)\))?', aspects_str)
            for match in aspect_matches:
                aspect_name = match[0].strip()
                aspect_sent = match[1].strip().lower()
                aspect_conf = float(match[2]) if match[2] else 0.8
                aspects.append({
                    "aspect": aspect_name,
                    "sentiment": aspect_sent,
                    "confidence": aspect_conf
                })
        
        # Check if concerning based on anti_national score
        is_concerning = scores.get('anti_national', 0.0) > 0.3
        
        return SentimentResult(
            overall_sentiment=sentiment,
            confidence=confidence,
            scores=scores,
            primary_emotion=emotion,
            aspects=aspects,
            is_concerning=is_concerning,
            reasoning=reasoning
        )
    
    def _parse_scores(self, scores_str: str, sentiment: str, confidence: float) -> Dict[str, float]:
        """Parse scores from SCORES field or calculate from sentiment."""
        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0, "anti_national": 0.0}
        
        # Try to parse scores from SCORES field
        if scores_str:
            # Pattern: positive=0.5, negative=0.3, neutral=0.2, anti_national=0.0
            score_patterns = {
                'positive': r'positive\s*=\s*([\d.]+)',
                'negative': r'negative\s*=\s*([\d.]+)',
                'neutral': r'neutral\s*=\s*([\d.]+)',
                'anti_national': r'anti_national\s*=\s*([\d.]+)'
            }
            
            parsed = False
            for key, pattern in score_patterns.items():
                match = re.search(pattern, scores_str, re.IGNORECASE)
                if match:
                    try:
                        scores[key] = float(match.group(1))
                        parsed = True
                    except:
                        pass
            
            # If we parsed scores, normalize them
            if parsed:
                total = sum(scores.values())
                if total > 0:
                    # Normalize to sum to 1.0
                    for key in scores:
                        scores[key] = scores[key] / total
                return scores
        
        # Fallback: calculate scores from sentiment
        return self._calculate_scores(sentiment, confidence)
    
    def _extract_field(self, text: str, field_name: str, default: str = '') -> str:
        """Extract a field value from response."""
        pattern = rf'{field_name}:\s*(.+?)(?:\n|$)'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else default
    
    def _calculate_scores(self, sentiment: str, confidence: float) -> Dict[str, float]:
        """Calculate sentiment scores distribution including anti-national."""
        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0, "anti_national": 0.0}
        
        if sentiment == "positive":
            scores["positive"] = confidence
            scores["neutral"] = (1 - confidence) * 0.7
            scores["negative"] = (1 - confidence) * 0.3
            scores["anti_national"] = 0.0
        elif sentiment == "negative":
            scores["negative"] = confidence
            scores["neutral"] = (1 - confidence) * 0.7
            scores["positive"] = (1 - confidence) * 0.3
            scores["anti_national"] = 0.0
        elif sentiment == "neutral":
            scores["neutral"] = confidence
            scores["positive"] = (1 - confidence) * 0.5
            scores["negative"] = (1 - confidence) * 0.5
            scores["anti_national"] = 0.0
        elif sentiment == "mixed":
            scores["positive"] = 0.4
            scores["negative"] = 0.4
            scores["neutral"] = 0.2
            scores["anti_national"] = 0.0
        elif sentiment == "anti_national":
            # Anti-national gets primary score, rest distributed
            scores["anti_national"] = confidence
            scores["negative"] = (1 - confidence) * 0.6  # Anti-national is also negative
            scores["neutral"] = (1 - confidence) * 0.3
            scores["positive"] = (1 - confidence) * 0.1
        else:
            # Default fallback
            scores["neutral"] = confidence
            scores["positive"] = (1 - confidence) * 0.5
            scores["negative"] = (1 - confidence) * 0.5
            scores["anti_national"] = 0.0
        
        # Ensure scores sum to approximately 1.0
        total = sum(scores.values())
        if total > 0:
            for key in scores:
                scores[key] = scores[key] / total
        
        return scores
    
    def _fallback_analysis(self, text: str) -> SentimentResult:
        """Fallback analysis using lexicon-based approach."""
        text_lower = text.lower()
        
        # Count indicators
        pos_count = sum(1 for word in self.POSITIVE_INDICATORS if word in text_lower)
        neg_count = sum(1 for word in self.NEGATIVE_INDICATORS if word in text_lower)
        
        # Check concerning patterns (anti-national detection)
        is_concerning = any(re.search(pattern, text_lower) for pattern in self.CONCERNING_PATTERNS)
        
        # Determine sentiment
        if is_concerning:
            sentiment = "anti_national"
            confidence = min(0.9, 0.6 + 0.1)  # Higher confidence for detected patterns
        elif pos_count > neg_count * 1.5:
            sentiment = "positive"
            confidence = min(0.9, 0.5 + pos_count * 0.1)
        elif neg_count > pos_count * 1.5:
            sentiment = "negative"
            confidence = min(0.9, 0.5 + neg_count * 0.1)
        elif pos_count > 0 and neg_count > 0:
            sentiment = "mixed"
            confidence = 0.6
        else:
            sentiment = "neutral"
            confidence = 0.7
        
        scores = self._calculate_scores(sentiment, confidence)
        
        return SentimentResult(
            overall_sentiment=sentiment,
            confidence=confidence,
            scores=scores,
            primary_emotion="neutral",
            is_concerning=is_concerning,
            reasoning="Fallback lexicon-based analysis (no LLM available)"
        )


class SentimentStep(PipelineStep):
    """
    Sentiment Analysis Pipeline Step.
    
    Integrates with the CoT pipeline for comprehensive
    sentiment analysis with aspect and emotion detection.
    """
    
    def __init__(
        self,
        executor: StepExecutor = None,
        detect_aspects: bool = True,
        detect_emotions: bool = True
    ):
        super().__init__(
            name="sentiment",
            description="Sentiment analysis with emotion and aspect detection"
        )
        self.analyzer = SentimentAnalyzer(
            executor=executor,
            detect_aspects=detect_aspects,
            detect_emotions=detect_emotions
        )
    
    def get_function_definition(self) -> FunctionDefinition:
        return SENTIMENT_FUNCTION
    
    def get_cot_prompt(self, context: PipelineContext) -> str:
        return self.analyzer.SYSTEM_PROMPT
    
    def execute(self, context: PipelineContext) -> StepResult:
        """Execute sentiment analysis."""
        start_time = time.time()
        
        text = context.current_text or context.original_input
        result = self.analyzer.analyze(text)
        
        return StepResult(
            step_name=self.name,
            status=StepStatus.SUCCESS,
            output=result.to_dict(),
            reasoning=result.reasoning,
            confidence=result.confidence,
            duration_ms=int((time.time() - start_time) * 1000)
        )


# ============== Quick-use functions ==============

def analyze_sentiment(
    text: str,
    api_key: str = None,
    detect_aspects: bool = True
) -> Dict[str, Any]:
    """
    Quick function for sentiment analysis.
    
    Args:
        text: Input text
        api_key: Optional Groq API key
        detect_aspects: Extract aspect-level sentiments
        
    Returns:
        Dict with sentiment analysis results
    """
    from ..utils.groq_client import GroqClient
    from ..cot.executor import StepExecutor
    
    try:
        client = GroqClient(api_key=api_key)
        executor = StepExecutor(client)
        analyzer = SentimentAnalyzer(
            executor=executor,
            detect_aspects=detect_aspects
        )
        return analyzer.analyze(text).to_dict()
    except Exception as e:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "error": str(e)
        }
