"""
Domain Detector - 3-domain content classification using Groq LLM.

Classifies text into:
- Technology: Software, hardware, programming, engineering, IT
- Business: Companies, products, services, e-commerce, finance
- General: News, education, entertainment, lifestyle, other
"""

from typing import Dict, Any

from ..utils.groq_client import GroqClient


class DomainDetector:
    """
    Detects the domain/category of text content using Groq LLM.
    
    Uses Llama 3.3 70B to classify text into one of three domains
    with confidence scores.
    """
    
    SYSTEM_PROMPT = """You are a content classification expert. Your task is to analyze text and classify it into exactly ONE of three domains.

DOMAINS:
1. technology - Content about software, hardware, programming, engineering, IT, web development, AI/ML, data science, cybersecurity, gadgets, tech news
2. business - Content about companies, products, services, e-commerce, finance, marketing, corporate news, startups, investments, economics
3. general - Content about news, education, entertainment, lifestyle, health, sports, travel, food, culture, politics, or anything else

RULES:
- Choose the SINGLE most appropriate domain
- Provide confidence scores for ALL domains (must sum to 1.0)
- Include 2-3 relevant sub-categories
- Be decisive - avoid equal scores

Respond in JSON format:
{
    "primary_domain": "technology|business|general",
    "confidence": 0.0-1.0,
    "all_domains": {
        "technology": 0.0-1.0,
        "business": 0.0-1.0,
        "general": 0.0-1.0
    },
    "sub_categories": ["category1", "category2"],
    "reasoning": "Brief explanation of classification"
}"""

    def __init__(self, groq_client: GroqClient = None):
        """
        Initialize domain detector.
        
        Args:
            groq_client: Optional GroqClient instance (creates one if not provided)
        """
        self.client = groq_client or GroqClient()
    
    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect the domain of the given text.
        
        Args:
            text: Cleaned text to classify
            
        Returns:
            Dictionary with domain classification results
        """
        if not text or len(text.strip()) < 10:
            return self._empty_result("Text too short for classification")
        
        # Truncate very long text to save tokens
        max_chars = 4000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        try:
            result = self.client.chat_json(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=f"Classify the following text:\n\n{text}"
            )
            
            # Validate and normalize result
            return self._normalize_result(result)
            
        except Exception as e:
            return self._error_result(str(e))
    
    def _normalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and validate the LLM response."""
        valid_domains = ["technology", "business", "general"]
        
        primary_domain = result.get("primary_domain", "general").lower()
        if primary_domain not in valid_domains:
            primary_domain = "general"
        
        # Ensure all_domains exists and has all keys
        all_domains = result.get("all_domains", {})
        for domain in valid_domains:
            if domain not in all_domains:
                all_domains[domain] = 0.0
        
        # Normalize scores to sum to 1.0
        total = sum(all_domains.values())
        if total > 0:
            all_domains = {k: round(v / total, 3) for k, v in all_domains.items()}
        
        return {
            "status": "success",
            "primary_domain": primary_domain,
            "confidence": round(result.get("confidence", all_domains.get(primary_domain, 0.5)), 3),
            "all_domains": all_domains,
            "sub_categories": result.get("sub_categories", []),
            "reasoning": result.get("reasoning", "")
        }
    
    def _empty_result(self, reason: str) -> Dict[str, Any]:
        """Return result for empty/short input."""
        return {
            "status": "skipped",
            "primary_domain": None,
            "confidence": 0.0,
            "all_domains": {
                "technology": 0.0,
                "business": 0.0,
                "general": 0.0
            },
            "sub_categories": [],
            "reasoning": reason
        }
    
    def _error_result(self, error: str) -> Dict[str, Any]:
        """Return result for errors."""
        return {
            "status": "error",
            "primary_domain": None,
            "confidence": 0.0,
            "all_domains": {
                "technology": 0.0,
                "business": 0.0,
                "general": 0.0
            },
            "sub_categories": [],
            "error": error
        }
