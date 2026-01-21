"""
Chain of Thought Pipeline - Main orchestrator.

Chains all processing modules together:
1. Text Cleaning (Global + Temp)
2. Domain Identification
3. Language Detection

Outputs structured JSON with results from each step.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional
import json

from .config import config, validate_config
from .cleaners import GlobalCleaner, TempCleaner
from .processors import DomainDetector, LanguageDetector
from .utils import GroqClient, MarkdownConverter


class ChainOfThoughtPipeline:
    """
    Main pipeline orchestrator that chains all processing modules.
    
    Each step produces output that feeds into the next,
    with all results collected in a structured JSON output.
    """
    
    def __init__(self, groq_api_key: str = None):
        """
        Initialize the pipeline with all modules.
        
        Args:
            groq_api_key: Optional API key (defaults to env variable)
        """
        # Initialize cleaners
        self.global_cleaner = GlobalCleaner()
        self.temp_cleaner = TempCleaner()
        self.markdown_converter = MarkdownConverter()
        
        # Initialize Groq client (lazy - only when needed)
        self._groq_client = None
        self._groq_api_key = groq_api_key
        
        # Initialize processors (lazy)
        self._domain_detector = None
        self._language_detector = None
    
    @property
    def groq_client(self) -> GroqClient:
        """Lazy initialization of Groq client."""
        if self._groq_client is None:
            validate_config()
            self._groq_client = GroqClient(api_key=self._groq_api_key)
        return self._groq_client
    
    @property
    def domain_detector(self) -> DomainDetector:
        """Lazy initialization of domain detector."""
        if self._domain_detector is None:
            self._domain_detector = DomainDetector(self.groq_client)
        return self._domain_detector
    
    @property
    def language_detector(self) -> LanguageDetector:
        """Lazy initialization of language detector."""
        if self._language_detector is None:
            self._language_detector = LanguageDetector()
        return self._language_detector
    
    def process(
        self,
        raw_text: str,
        is_html: bool = False,
        skip_domain: bool = False,
        skip_language: bool = False
    ) -> Dict[str, Any]:
        """
        Process text through the full pipeline.
        
        Args:
            raw_text: Raw input text (HTML or plain text)
            is_html: If True, treat input as HTML
            skip_domain: If True, skip domain detection (saves API call)
            skip_language: If True, skip language detection
            
        Returns:
            Structured JSON dict with all processing results
        """
        start_time = datetime.now(timezone.utc)
        
        # Initialize result structure
        result = {
            "1_text_clean": {},
            "2_domain_ident": {},
            "3_lang_detect": {},
            "metadata": {}
        }
        
        # ===== Step 1: Text Cleaning =====
        result["1_text_clean"] = self._process_cleaning(raw_text, is_html)
        
        # Get cleaned text for subsequent steps
        cleaned_text = result["1_text_clean"].get("final_text", "")
        
        # ===== Step 2: Domain Identification =====
        if skip_domain:
            result["2_domain_ident"] = {"status": "skipped", "reason": "skip_domain=True"}
        else:
            result["2_domain_ident"] = self._process_domain(cleaned_text)
        
        # ===== Step 3: Language Detection =====
        if skip_language:
            result["3_lang_detect"] = {"status": "skipped", "reason": "skip_language=True"}
        else:
            result["3_lang_detect"] = self._process_language(cleaned_text)
        
        # ===== Metadata =====
        end_time = datetime.now(timezone.utc)
        result["metadata"] = {
            "pipeline_version": config.version,
            "model_used": config.groq.model,
            "processed_at": end_time.isoformat(),
            "processing_time_ms": int((end_time - start_time).total_seconds() * 1000)
        }
        
        return result
    
    def _process_cleaning(self, raw_text: str, is_html: bool) -> Dict[str, Any]:
        """Run cleaning steps."""
        original_length = len(raw_text)
        
        # Step 1a: Global cleaning
        global_result = self.global_cleaner.clean(raw_text)
        
        # Step 1b: Temp cleaning
        temp_result = self.temp_cleaner.clean(global_result["text"])
        
        # Step 1c: Markdown conversion (if enabled)
        if config.enable_markdown_conversion:
            md_result = self.markdown_converter.convert(temp_result["text"], is_html=is_html)
            markdown_text = md_result["text"]
        else:
            markdown_text = temp_result["text"]
        
        return {
            "status": "success",
            "global_cleaned": global_result["text"],
            "global_stats": global_result["stats"],
            "temp_cleaned": temp_result["text"],
            "temp_stats": temp_result["stats"],
            "markdown": markdown_text,
            "final_text": markdown_text,
            "overall_stats": {
                "original_length": original_length,
                "final_length": len(markdown_text),
                "total_reduction_percent": round(
                    (1 - len(markdown_text) / original_length) * 100, 2
                ) if original_length > 0 else 0
            }
        }
    
    def _process_domain(self, text: str) -> Dict[str, Any]:
        """Run domain detection."""
        try:
            return self.domain_detector.detect(text)
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _process_language(self, text: str) -> Dict[str, Any]:
        """Run language detection."""
        try:
            return self.language_detector.detect(text)
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def process_batch(
        self,
        texts: list,
        is_html: bool = False,
        skip_domain: bool = False,
        skip_language: bool = False
    ) -> list:
        """
        Process multiple texts through the pipeline.
        
        Args:
            texts: List of raw text strings
            is_html: If True, treat all inputs as HTML
            skip_domain: If True, skip domain detection
            skip_language: If True, skip language detection
            
        Returns:
            List of result dictionaries
        """
        return [
            self.process(text, is_html, skip_domain, skip_language)
            for text in texts
        ]
    
    def to_json(self, result: Dict[str, Any], pretty: bool = True) -> str:
        """
        Convert result to JSON string.
        
        Args:
            result: Pipeline result dictionary
            pretty: If True, format with indentation
            
        Returns:
            JSON string
        """
        if pretty:
            return json.dumps(result, indent=2, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)


# Convenience function for quick processing
def process_text(
    text: str,
    is_html: bool = False,
    groq_api_key: str = None
) -> Dict[str, Any]:
    """
    Quick function to process text through the pipeline.
    
    Args:
        text: Raw input text
        is_html: If True, treat as HTML
        groq_api_key: Optional API key
        
    Returns:
        Pipeline result dictionary
    """
    pipeline = ChainOfThoughtPipeline(groq_api_key=groq_api_key)
    return pipeline.process(text, is_html=is_html)
