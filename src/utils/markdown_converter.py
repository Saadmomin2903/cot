"""
Markdown Converter - HTML to Markdown conversion.

Converts cleaned HTML/text to well-structured Markdown
for better LLM processing.
"""

from typing import Dict, Any
import re

try:
    from markdownify import markdownify as md
    HAS_MARKDOWNIFY = True
except ImportError:
    HAS_MARKDOWNIFY = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


class MarkdownConverter:
    """
    Converts text/HTML to clean Markdown format.
    
    Features:
    - Preserves semantic structure (headers, lists, links)
    - Removes non-content elements
    - Cleans up excess whitespace
    """
    
    def __init__(self):
        """Initialize converter."""
        self._header_pattern = re.compile(r'^(#{1,6})\s*$', re.MULTILINE)
        self._empty_link_pattern = re.compile(r'\[([^\]]*)\]\(\s*\)')
        self._multi_newline_pattern = re.compile(r'\n{3,}')
    
    def convert(self, text: str, is_html: bool = False) -> Dict[str, Any]:
        """
        Convert text to Markdown format.
        
        Args:
            text: Input text (HTML or plain text)
            is_html: If True, treat input as HTML
            
        Returns:
            Dictionary with markdown text and stats
        """
        original_length = len(text)
        
        if is_html and HAS_MARKDOWNIFY:
            # Convert HTML to Markdown
            markdown = self._convert_html(text)
        else:
            # Clean and structure plain text as Markdown
            markdown = self._structure_text(text)
        
        # Post-process markdown
        markdown = self._clean_markdown(markdown)
        
        return {
            "text": markdown,
            "stats": {
                "original_length": original_length,
                "markdown_length": len(markdown),
                "is_html_source": is_html
            }
        }
    
    def _convert_html(self, html: str) -> str:
        """Convert HTML to Markdown using markdownify."""
        if not HAS_MARKDOWNIFY:
            return html
        
        # Configure markdownify options
        markdown = md(
            html,
            heading_style="ATX",  # Use # style headers
            bullets="-",  # Use - for lists
            strip=['script', 'style', 'nav', 'footer', 'header', 'aside'],
            convert=['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'a', 'strong', 'em', 'code', 'pre', 'blockquote']
        )
        
        return markdown
    
    def _structure_text(self, text: str) -> str:
        """
        Structure plain text as Markdown.
        
        Attempts to identify and format:
        - Potential headers (short lines followed by content)
        - Lists (lines starting with -, *, or numbers)
        - Paragraphs
        """
        lines = text.split('\n')
        result = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if not stripped:
                result.append('')
                continue
            
            # Check if this looks like a list item
            if re.match(r'^[-*•]\s+', stripped):
                result.append(f"- {stripped[2:].strip()}")
            elif re.match(r'^\d+[.)]\s+', stripped):
                # Numbered list
                result.append(stripped)
            # Check if this could be a header (short line, followed by content)
            elif len(stripped) < 60 and not stripped.endswith(('.', ',', ';', ':')):
                # Check if next non-empty line exists and is longer
                next_content = self._get_next_content(lines, i + 1)
                if next_content and len(next_content) > len(stripped):
                    result.append(f"## {stripped}")
                else:
                    result.append(stripped)
            else:
                result.append(stripped)
        
        return '\n'.join(result)
    
    def _get_next_content(self, lines: list, start: int) -> str:
        """Get the next non-empty line."""
        for i in range(start, min(start + 3, len(lines))):
            if lines[i].strip():
                return lines[i].strip()
        return ""
    
    def _clean_markdown(self, markdown: str) -> str:
        """Clean up generated Markdown."""
        # Remove empty headers
        markdown = self._header_pattern.sub('', markdown)
        
        # Remove empty links
        markdown = self._empty_link_pattern.sub(r'\1', markdown)
        
        # Collapse multiple newlines
        markdown = self._multi_newline_pattern.sub('\n\n', markdown)
        
        # Strip and return
        return markdown.strip()
