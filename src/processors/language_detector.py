"""
Language Detector - Language and script type detection.

Detects:
- Primary language (ISO 639-1 code)
- Script type (roman, non_roman, mixed)
- Confidence score
"""

import re
import unicodedata
from typing import Dict, Any, Tuple

try:
    from langdetect import detect, detect_langs
    from langdetect.lang_detect_exception import LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


class LanguageDetector:
    """
    Detects language and script type of text.
    
    Uses langdetect library for language detection and
    Unicode analysis for script type classification.
    """
    
    # ISO 639-1 language names
    LANGUAGE_NAMES = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
        "zh-cn": "Chinese (Simplified)",
        "zh-tw": "Chinese (Traditional)",
        "ar": "Arabic",
        "hi": "Hindi",
        "bn": "Bengali",
        "pa": "Punjabi",
        "ta": "Tamil",
        "te": "Telugu",
        "mr": "Marathi",
        "gu": "Gujarati",
        "kn": "Kannada",
        "ml": "Malayalam",
        "th": "Thai",
        "vi": "Vietnamese",
        "id": "Indonesian",
        "ms": "Malay",
        "tl": "Filipino",
        "nl": "Dutch",
        "pl": "Polish",
        "uk": "Ukrainian",
        "tr": "Turkish",
        "he": "Hebrew",
        "fa": "Persian",
        "ur": "Urdu",
        "el": "Greek",
        "sv": "Swedish",
        "no": "Norwegian",
        "da": "Danish",
        "fi": "Finnish",
        "cs": "Czech",
        "ro": "Romanian",
        "hu": "Hungarian",
        "sk": "Slovak",
        "bg": "Bulgarian",
        "hr": "Croatian",
        "sr": "Serbian",
        "sl": "Slovenian",
        "lt": "Lithuanian",
        "lv": "Latvian",
        "et": "Estonian",
    }
    
    # Scripts that use Roman alphabet
    ROMAN_SCRIPTS = {
        "LATIN",
    }
    
    # Scripts that are non-Roman
    NON_ROMAN_SCRIPTS = {
        "CYRILLIC", "GREEK", "ARABIC", "HEBREW",
        "DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI",
        "TAMIL", "TELUGU", "KANNADA", "MALAYALAM",
        "THAI", "LAO", "TIBETAN", "MYANMAR",
        "GEORGIAN", "ARMENIAN", "ETHIOPIC",
        "CJK", "HIRAGANA", "KATAKANA", "HANGUL",
        "SINHALA", "KHMER"
    }
    
    def __init__(self):
        """Initialize language detector."""
        if not HAS_LANGDETECT:
            raise ImportError(
                "langdetect library required. Install with: pip install langdetect"
            )
    
    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect language and script type of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with language detection results
        """
        if not text or len(text.strip()) < 10:
            return self._empty_result("Text too short for detection")
        
        try:
            # Detect language
            lang_code, lang_confidence = self._detect_language(text)
            
            # Detect script type
            script_type, script_breakdown = self._detect_script(text)
            
            return {
                "status": "success",
                "language_code": lang_code,
                "language_name": self.LANGUAGE_NAMES.get(lang_code, "Unknown"),
                "script_type": script_type,
                "confidence": lang_confidence,
                "script_breakdown": script_breakdown
            }
            
        except Exception as e:
            return self._error_result(str(e))
    
    def _detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect primary language using langdetect.
        
        Returns:
            Tuple of (language_code, confidence)
        """
        try:
            # Get all detected languages with probabilities
            langs = detect_langs(text)
            
            if langs:
                primary = langs[0]
                return (primary.lang, round(primary.prob, 3))
            else:
                return ("en", 0.5)  # Default fallback
                
        except LangDetectException:
            return ("en", 0.0)
    
    def _detect_script(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Detect script type by analyzing Unicode characters.
        
        Returns:
            Tuple of (script_type, breakdown_dict)
        """
        roman_count = 0
        non_roman_count = 0
        other_count = 0
        
        for char in text:
            if char.isspace() or not char.isalpha():
                continue
            
            try:
                script = unicodedata.name(char, "").split()[0]
                
                if script in self.ROMAN_SCRIPTS or script == "LATIN":
                    roman_count += 1
                elif any(s in unicodedata.name(char, "") for s in self.NON_ROMAN_SCRIPTS):
                    non_roman_count += 1
                elif self._is_cjk(char):
                    non_roman_count += 1
                else:
                    # Check if it's a letter but not clearly categorized
                    if char.isalpha():
                        # Default Latin-like characters to Roman
                        if ord(char) < 0x0250:  # Basic Latin + Latin Extended
                            roman_count += 1
                        else:
                            non_roman_count += 1
                    else:
                        other_count += 1
            except (ValueError, KeyError):
                other_count += 1
        
        total = roman_count + non_roman_count
        if total == 0:
            return ("unknown", {"roman": 0.0, "non_roman": 0.0})
        
        roman_ratio = roman_count / total
        non_roman_ratio = non_roman_count / total
        
        breakdown = {
            "roman": round(roman_ratio, 3),
            "non_roman": round(non_roman_ratio, 3)
        }
        
        # Determine script type
        if roman_ratio >= 0.9:
            script_type = "roman"
        elif non_roman_ratio >= 0.9:
            script_type = "non_roman"
        else:
            script_type = "mixed"
        
        return (script_type, breakdown)
    
    def _is_cjk(self, char: str) -> bool:
        """Check if character is CJK (Chinese, Japanese, Korean)."""
        cp = ord(char)
        return (
            (0x4E00 <= cp <= 0x9FFF) or   # CJK Unified Ideographs
            (0x3400 <= cp <= 0x4DBF) or   # CJK Unified Ideographs Extension A
            (0x3040 <= cp <= 0x309F) or   # Hiragana
            (0x30A0 <= cp <= 0x30FF) or   # Katakana
            (0xAC00 <= cp <= 0xD7AF)      # Hangul Syllables
        )
    
    def _empty_result(self, reason: str) -> Dict[str, Any]:
        """Return result for empty/short input."""
        return {
            "status": "skipped",
            "language_code": None,
            "language_name": None,
            "script_type": None,
            "confidence": 0.0,
            "script_breakdown": {},
            "reasoning": reason
        }
    
    def _error_result(self, error: str) -> Dict[str, Any]:
        """Return result for errors."""
        return {
            "status": "error",
            "language_code": None,
            "language_name": None,
            "script_type": None,
            "confidence": 0.0,
            "script_breakdown": {},
            "error": error
        }
