"""
Global Cleaner - Standard text normalization that should always happen.

Handles:
- URL removal
- HTML tag removal
- Contraction expansion
- Bracketed content removal
- Unicode normalization
- Special character handling
"""

import re
import unicodedata
from typing import Dict, Any
import contractions


class GlobalCleaner:
    """
    Performs global text cleaning operations.
    
    These are standard normalizations that should be applied to all text
    regardless of the specific use case.
    """
    
    def __init__(self, config=None):
        """Initialize with optional configuration."""
        from ..config import config as default_config
        self.config = config or default_config.cleaning
        
        # Compile regex patterns for performance
        self._url_pattern = re.compile(self.config.url_pattern, re.IGNORECASE)
        self._html_pattern = re.compile(self.config.html_tag_pattern, re.DOTALL)
        self._bracketed_pattern = re.compile(self.config.bracketed_pattern)
        self._multi_space_pattern = re.compile(r'[ \t]+')
        self._multi_newline_pattern = re.compile(r'\n{3,}')
    
    def clean(self, text: str) -> Dict[str, Any]:
        """
        Apply all global cleaning operations.
        
        Args:
            text: Raw input text
            
        Returns:
            Dictionary with cleaned text and statistics
        """
        original_length = len(text)
        
        # Step 1: Remove HTML tags first (they may contain URLs)
        text = self._remove_html_tags(text)
        
        # Step 2: Remove URLs
        text = self._remove_urls(text)
        
        # Step 3: Remove bracketed content (often navigation links)
        text = self._remove_bracketed_content(text)
        
        # Step 4: Unicode normalization
        text = self._normalize_unicode(text)
        
        # Step 5: Expand contractions
        text = self._expand_contractions(text)
        
        # Step 6: Normalize whitespace
        text = self._normalize_whitespace(text)
        
        cleaned_length = len(text)
        
        return {
            "text": text,
            "stats": {
                "original_length": original_length,
                "cleaned_length": cleaned_length,
                "chars_removed": original_length - cleaned_length,
                "reduction_percent": round(
                    (1 - cleaned_length / original_length) * 100, 2
                ) if original_length > 0 else 0
            }
        }
    
    def _remove_urls(self, text: str) -> str:
        """Remove HTTP/HTTPS URLs and www links."""
        return self._url_pattern.sub('', text)
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML/XML tags while preserving content."""
        # First, handle common block elements by adding newlines
        text = re.sub(r'</(p|div|br|li|h[1-6])>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        
        # Then remove all remaining tags
        return self._html_pattern.sub('', text)
    
    def _remove_bracketed_content(self, text: str) -> str:
        """Remove content in square brackets (often links or references)."""
        return self._bracketed_pattern.sub('', text)
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters using NFKC normalization.
        
        This:
        - Converts compatibility characters to canonical form
        - Normalizes fancy quotes to standard quotes
        - Handles ligatures and other special characters
        """
        # NFKC normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Additional common replacements
        replacements = {
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2026': '...',  # Ellipsis
            '\u00a0': ' ',  # Non-breaking space
            '\u200b': '',   # Zero-width space
            '\ufeff': '',   # BOM
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _expand_contractions(self, text: str) -> str:
        """
        Expand English contractions to full form.
        
        Examples:
        - "I'm" -> "I am"
        - "won't" -> "will not"
        - "they've" -> "they have"
        """
        try:
            return contractions.fix(text)
        except Exception:
            # If contractions library fails, return original
            return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace:
        - Collapse multiple spaces/tabs to single space
        - Collapse multiple newlines to max 2
        - Strip leading/trailing whitespace from lines
        """
        # Collapse multiple spaces/tabs to single space
        text = self._multi_space_pattern.sub(' ', text)
        
        # Collapse multiple newlines to max 2
        text = self._multi_newline_pattern.sub('\n\n', text)
        
        # Strip each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Final strip
        return text.strip()
