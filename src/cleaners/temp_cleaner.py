"""
Temp Cleaner - Context-specific cleaning operations.

Handles:
- Short/low-information line removal
- Null string removal
- Navigation text removal
- Whitespace collapse
- Header/footer stripping
"""

import re
from typing import Dict, Any, List


class TempCleaner:
    """
    Performs temporary/context-specific text cleaning operations.
    
    These cleanings are more aggressive and context-dependent,
    aimed at removing web-specific junk content.
    """
    
    def __init__(self, config=None):
        """Initialize with optional configuration."""
        from ..config import config as default_config
        self.config = config or default_config.cleaning
        
        # Compile navigation patterns
        self._nav_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.config.nav_patterns
        ]
        
        # Pattern for short lines (low information content)
        self._short_line_pattern = re.compile(
            r'^[ \t]*(\S{1,' + str(self.config.min_line_length) + r'})?[ \t]*$',
            re.MULTILINE
        )
        
        # Pattern for lines that are just punctuation/symbols
        self._symbol_line_pattern = re.compile(
            r'^[\s\-_=+*#@!?.,:;|/\\<>(){}[\]]+$',
            re.MULTILINE
        )
        
        # Pattern for repeated characters (like "======" or "------")
        self._repeated_char_pattern = re.compile(
            r'^(.)\1{4,}$',
            re.MULTILINE
        )
    
    def clean(self, text: str) -> Dict[str, Any]:
        """
        Apply all temp cleaning operations.
        
        Args:
            text: Text after global cleaning
            
        Returns:
            Dictionary with cleaned text and statistics
        """
        original_length = len(text)
        original_lines = text.count('\n') + 1
        
        # Step 1: Remove null strings
        text = self._remove_null_strings(text)
        
        # Step 2: Remove navigation patterns
        text = self._remove_navigation(text)
        
        # Step 3: Remove short/low-information lines
        text = self._remove_short_lines(text)
        
        # Step 4: Remove symbol-only lines
        text = self._remove_symbol_lines(text)
        
        # Step 5: Remove repeated character lines
        text = self._remove_repeated_chars(text)
        
        # Step 6: Final whitespace cleanup
        text = self._final_cleanup(text)
        
        cleaned_length = len(text)
        cleaned_lines = text.count('\n') + 1 if text else 0
        
        return {
            "text": text,
            "stats": {
                "original_length": original_length,
                "cleaned_length": cleaned_length,
                "original_lines": original_lines,
                "cleaned_lines": cleaned_lines,
                "lines_removed": original_lines - cleaned_lines,
                "reduction_percent": round(
                    (1 - cleaned_length / original_length) * 100, 2
                ) if original_length > 0 else 0
            }
        }
    
    def _remove_null_strings(self, text: str) -> str:
        """Remove common null/undefined strings."""
        for null_str in self.config.null_strings:
            # Only remove if it's a standalone word (not part of another word)
            pattern = re.compile(r'\b' + re.escape(null_str) + r'\b', re.IGNORECASE)
            text = pattern.sub('', text)
        return text
    
    def _remove_navigation(self, text: str) -> str:
        """Remove common navigation patterns."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            is_nav = False
            for pattern in self._nav_patterns:
                if pattern.match(line.strip()):
                    is_nav = True
                    break
            if not is_nav:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _remove_short_lines(self, text: str) -> str:
        """Remove lines that are too short to contain useful information."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Keep line if it has enough content
            # Also keep empty lines (for paragraph structure)
            if len(stripped) == 0 or len(stripped) >= self.config.min_line_length:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _remove_symbol_lines(self, text: str) -> str:
        """Remove lines that contain only symbols/punctuation."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if not self._symbol_line_pattern.match(line):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _remove_repeated_chars(self, text: str) -> str:
        """Remove lines with repeated characters (like separators)."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not self._repeated_char_pattern.match(stripped):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _final_cleanup(self, text: str) -> str:
        """Final whitespace normalization."""
        # Remove multiple consecutive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove blank lines at start and end
        return text.strip()
