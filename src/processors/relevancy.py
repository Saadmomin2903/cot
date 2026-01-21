"""
Relevancy Analysis Module

Analyzes text relevancy to topics, domains, and concepts.

Features:
- Topic relevancy scoring
- Domain matching
- Keyword relevancy
- Semantic relevancy using LLM
- Multi-topic scoring
- Context relevancy assessment

Determines what the text is about and how relevant it is to specific topics.
"""

import re
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from ..cot import PipelineStep, PipelineContext, StepResult, StepStatus, FunctionDefinition
from ..cot.executor import StepExecutor


@dataclass
class RelevancyScore:
    """Score for a single topic/category."""
    topic: str
    score: float  # 0.0 to 1.0
    confidence: float = 0.8
    keywords_matched: List[str] = field(default_factory=list)
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 2),
            "keywords": self.keywords_matched if self.keywords_matched else None,
            "reasoning": self.reasoning if self.reasoning else None
        }


@dataclass
class RelevancyResult:
    """Complete relevancy analysis result."""
    primary_topic: str
    primary_score: float
    topic_scores: List[RelevancyScore] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    is_relevant_to: List[str] = field(default_factory=list)  # Topics above threshold
    overall_specificity: float = 0.5  # How specific vs general
    confidence: float = 0.85
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_topic": self.primary_topic,
            "primary_score": round(self.primary_score, 3),
            "topic_scores": [t.to_dict() for t in self.topic_scores],
            "keywords": self.keywords if self.keywords else None,
            "is_relevant_to": self.is_relevant_to if self.is_relevant_to else None,
            "specificity": round(self.overall_specificity, 2),
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


# Default topic categories for relevancy analysis
DEFAULT_TOPICS = [
    "Technology",
    "Business",
    "Finance",
    "Science",
    "Politics",
    "Sports",
    "Entertainment",
    "Health",
    "Education",
    "Environment",
    "Legal",
    "Travel",
    "Food",
    "Art & Culture",
    "General"
]


# Function definition for CoT executor
RELEVANCY_FUNCTION = FunctionDefinition(
    name="analyze_relevancy",
    description="Analyze text relevancy to topics and concepts",
    parameters={
        "primary_topic": {
            "type": "string",
            "description": "The most relevant topic"
        },
        "primary_score": {
            "type": "number",
            "description": "Relevancy score 0-1 for primary topic"
        },
        "topic_scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "score": {"type": "number"},
                    "keywords": {"type": "array", "items": {"type": "string"}}
                }
            },
            "description": "Scores for each topic"
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key terms from the text"
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation of relevancy analysis"
        }
    },
    required=["primary_topic", "primary_score", "reasoning"]
)


class RelevancyAnalyzer:
    """
    Relevancy Analysis System.
    
    Determines what topics/categories text is relevant to
    with confidence scores for each.
    
    Features:
    - Multi-topic relevancy scoring
    - Keyword extraction and matching
    - Specificity assessment
    - Custom topic lists support
    - LLM-powered semantic analysis
    
    Usage:
        analyzer = RelevancyAnalyzer(executor)
        result = analyzer.analyze("Apple announced new iPhone...")
        print(result.primary_topic)  # "Technology"
        print(result.topic_scores)  # [{topic: "Technology", score: 0.95}, ...]
    """
    
    SYSTEM_PROMPT = """You are an expert at analyzing text relevancy and categorization.

## Your Goal
Determine what topics, domains, and concepts the text is most relevant to.

## Analysis Process
1. Read and understand the text content
2. Identify key themes, subjects, and terminology
3. Match against topic categories
4. Score relevancy for each topic (0.0 to 1.0)
5. Extract keywords that indicate topic relevancy

## Scoring Guidelines
- 0.9-1.0: Highly relevant, primary focus of text
- 0.7-0.8: Strongly relevant, significant content
- 0.5-0.6: Moderately relevant, mentioned topics
- 0.3-0.4: Tangentially relevant
- 0.1-0.2: Barely relevant, minor mentions
- 0.0: Not relevant at all

## Key Indicators
- Domain-specific terminology
- Named entities (people, companies, places)
- Topic-specific concepts
- Context and purpose of text
- Industry or field references

## Output Quality
- Be specific about why topics are relevant
- Extract keywords that justify scores
- Identify overlapping relevancies
- Note if text is general or highly specific"""

    # Topic keyword mappings for fallback
    TOPIC_KEYWORDS = {
        "Technology": [
            "software", "hardware", "ai", "artificial intelligence", "machine learning",
            "computer", "digital", "app", "algorithm", "data", "cloud", "internet",
            "smartphone", "device", "tech", "innovation", "coding", "programming"
        ],
        "Business": [
            "company", "corporation", "ceo", "revenue", "profit", "market", "strategy",
            "enterprise", "startup", "investment", "stakeholder", "management", "growth"
        ],
        "Finance": [
            "money", "bank", "stock", "investment", "trading", "currency", "loan",
            "interest", "portfolio", "dividend", "equity", "capital", "financial"
        ],
        "Science": [
            "research", "study", "experiment", "hypothesis", "discovery", "laboratory",
            "scientist", "physics", "chemistry", "biology", "theory", "evidence"
        ],
        "Politics": [
            "government", "election", "policy", "politician", "vote", "democracy",
            "congress", "parliament", "legislation", "political", "party", "campaign"
        ],
        "Sports": [
            "game", "team", "player", "score", "championship", "athlete", "tournament",
            "coach", "league", "match", "competition", "win", "training"
        ],
        "Entertainment": [
            "movie", "music", "show", "celebrity", "film", "actor", "singer",
            "concert", "album", "series", "streaming", "hollywood", "award"
        ],
        "Health": [
            "medical", "doctor", "patient", "treatment", "disease", "hospital",
            "medicine", "health", "symptom", "diagnosis", "therapy", "vaccine"
        ],
        "Education": [
            "school", "university", "student", "teacher", "learning", "course",
            "education", "degree", "exam", "curriculum", "academic", "study"
        ],
        "Environment": [
            "climate", "environment", "pollution", "sustainable", "carbon", "ecosystem",
            "conservation", "energy", "renewable", "green", "nature", "recycling"
        ],
        "Legal": [
            "law", "court", "legal", "lawyer", "judge", "lawsuit", "regulation",
            "rights", "contract", "compliance", "verdict", "attorney", "justice"
        ]
    }

    def __init__(
        self,
        executor: StepExecutor = None,
        topics: List[str] = None,
        relevancy_threshold: float = 0.4
    ):
        """
        Initialize Relevancy Analyzer.
        
        Args:
            executor: StepExecutor for LLM calls
            topics: Custom list of topics to analyze
            relevancy_threshold: Minimum score to be considered relevant
        """
        self.executor = executor
        self.topics = topics or DEFAULT_TOPICS
        self.relevancy_threshold = relevancy_threshold
    
    def analyze(
        self,
        text: str,
        custom_topics: List[str] = None
    ) -> RelevancyResult:
        """
        Analyze text relevancy to topics.
        
        Args:
            text: Input text to analyze
            custom_topics: Optional custom topics to score against
            
        Returns:
            RelevancyResult with topic scores
        """
        topics = custom_topics or self.topics
        
        if not self.executor:
            return self._fallback_analysis(text, topics)
        
        return self._llm_analysis(text, topics)
    
    def _llm_analysis(
        self,
        text: str,
        topics: List[str]
    ) -> RelevancyResult:
        """Analyze using LLM."""
        topics_str = ", ".join(topics)
        
        user_prompt = f"""Analyze the relevancy of the following text to different topics.

## Topics to Score
{topics_str}

## Text to Analyze
{text[:4000]}

## Output Format
PRIMARY_TOPIC: [most relevant topic]
PRIMARY_SCORE: [0.0 to 1.0]

TOPIC_SCORES:
- Topic1: score (keywords: word1, word2)
- Topic2: score (keywords: word1, word2)
...

KEYWORDS: [key terms from text, comma separated]

SPECIFICITY: [0.0 to 1.0 - how specific vs general the text is]

REASONING:
[Explain why topics received their scores]

Analyze now:"""

        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.executor.client.chat(
                messages=messages,
                temperature=0.2
            )
            
            return self._parse_relevancy_response(response, topics)
            
        except Exception as e:
            return self._fallback_analysis(text, topics)
    
    def _parse_relevancy_response(
        self,
        response: str,
        topics: List[str]
    ) -> RelevancyResult:
        """Parse LLM response into RelevancyResult."""
        # Extract primary topic and score
        primary_topic = self._extract_field(response, "PRIMARY_TOPIC", "General")
        primary_score_str = self._extract_field(response, "PRIMARY_SCORE", "0.5")
        keywords_str = self._extract_field(response, "KEYWORDS", "")
        specificity_str = self._extract_field(response, "SPECIFICITY", "0.5")
        reasoning = self._extract_section(response, "REASONING")
        
        # Parse primary score
        try:
            primary_score = float(re.search(r'[\d.]+', primary_score_str).group())
            primary_score = min(1.0, max(0.0, primary_score))
        except:
            primary_score = 0.5
        
        # Parse specificity
        try:
            specificity = float(re.search(r'[\d.]+', specificity_str).group())
            specificity = min(1.0, max(0.0, specificity))
        except:
            specificity = 0.5
        
        # Parse keywords
        keywords = [k.strip() for k in keywords_str.split(",")] if keywords_str else []
        keywords = [k for k in keywords if k and len(k) > 1]
        
        # Parse topic scores
        topic_scores = self._parse_topic_scores(response, topics)
        
        # Get topics above threshold
        is_relevant_to = [
            ts.topic for ts in topic_scores 
            if ts.score >= self.relevancy_threshold
        ]
        
        return RelevancyResult(
            primary_topic=primary_topic,
            primary_score=primary_score,
            topic_scores=topic_scores,
            keywords=keywords[:15],
            is_relevant_to=is_relevant_to,
            overall_specificity=specificity,
            confidence=0.85,
            reasoning=reasoning
        )
    
    def _parse_topic_scores(
        self,
        response: str,
        topics: List[str]
    ) -> List[RelevancyScore]:
        """Parse topic scores from response."""
        scores = []
        
        # Look for "Topic: score" patterns
        for topic in topics:
            pattern = rf'{re.escape(topic)}[:\s]+([0-9.]+)'
            match = re.search(pattern, response, re.IGNORECASE)
            
            if match:
                try:
                    score = float(match.group(1))
                    score = min(1.0, max(0.0, score))
                except:
                    score = 0.0
                
                # Try to find keywords for this topic
                kw_pattern = rf'{re.escape(topic)}.*?keywords?[:\s]+([^\n\)]+)'
                kw_match = re.search(kw_pattern, response, re.IGNORECASE)
                keywords = []
                if kw_match:
                    keywords = [k.strip() for k in kw_match.group(1).split(",")]
                    keywords = [k for k in keywords if k and len(k) > 1][:5]
                
                scores.append(RelevancyScore(
                    topic=topic,
                    score=score,
                    keywords_matched=keywords
                ))
        
        # Sort by score descending
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores
    
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
    
    def _fallback_analysis(
        self,
        text: str,
        topics: List[str]
    ) -> RelevancyResult:
        """Keyword-based fallback analysis."""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        scores = []
        for topic in topics:
            keywords = self.TOPIC_KEYWORDS.get(topic, [])
            matched = [k for k in keywords if k in text_lower]
            
            if matched:
                # Score based on keyword matches
                score = min(1.0, len(matched) * 0.15)
            else:
                score = 0.0
            
            scores.append(RelevancyScore(
                topic=topic,
                score=score,
                confidence=0.6,
                keywords_matched=matched[:5]
            ))
        
        # Sort by score
        scores.sort(key=lambda x: x.score, reverse=True)
        
        primary = scores[0] if scores else RelevancyScore(topic="General", score=0.5)
        
        is_relevant_to = [s.topic for s in scores if s.score >= self.relevancy_threshold]
        
        return RelevancyResult(
            primary_topic=primary.topic,
            primary_score=primary.score,
            topic_scores=scores,
            keywords=[],
            is_relevant_to=is_relevant_to,
            overall_specificity=0.5,
            confidence=0.6,
            reasoning="Keyword-based analysis (fallback mode)"
        )


class RelevancyStep(PipelineStep):
    """
    Relevancy Analysis Pipeline Step.
    
    Determines what topics/domains the text is relevant to.
    """
    
    def __init__(
        self,
        executor: StepExecutor = None,
        topics: List[str] = None
    ):
        super().__init__(
            name="relevancy",
            description="Analyze text relevancy to topics"
        )
        self.analyzer = RelevancyAnalyzer(
            executor=executor,
            topics=topics
        )
    
    def get_function_definition(self) -> FunctionDefinition:
        return RELEVANCY_FUNCTION
    
    def get_cot_prompt(self, context: PipelineContext) -> str:
        return self.analyzer.SYSTEM_PROMPT
    
    def execute(self, context: PipelineContext) -> StepResult:
        """Execute relevancy analysis."""
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

def analyze_relevancy(
    text: str,
    topics: List[str] = None,
    api_key: str = None
) -> Dict[str, Any]:
    """
    Quick function for relevancy analysis.
    
    Args:
        text: Input text
        topics: Optional custom topics
        api_key: Optional Groq API key
        
    Returns:
        Dict with relevancy scores
    """
    from ..utils.groq_client import GroqClient
    from ..cot.executor import StepExecutor
    
    try:
        client = GroqClient(api_key=api_key)
        executor = StepExecutor(client)
        analyzer = RelevancyAnalyzer(executor=executor, topics=topics)
        return analyzer.analyze(text).to_dict()
    except Exception as e:
        return {
            "primary_topic": "General",
            "primary_score": 0.0,
            "error": str(e)
        }
