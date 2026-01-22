"""
Tests for the Chain of Thought Pipeline.
"""

import pytest
from src.cleaners import GlobalCleaner, TempCleaner
from src.utils import MarkdownConverter
from src.processors import LanguageDetector


class TestGlobalCleaner:
    """Tests for GlobalCleaner."""
    
    def setup_method(self):
        self.cleaner = GlobalCleaner()
    
    def test_remove_urls(self):
        text = "Check out https://example.com for more info"
        result = self.cleaner.clean(text)
        assert "https://example.com" not in result["text"]
        assert "Check out" in result["text"]
    
    def test_remove_html_tags(self):
        text = "<p>Hello <strong>World</strong></p>"
        result = self.cleaner.clean(text)
        assert "<p>" not in result["text"]
        assert "<strong>" not in result["text"]
        assert "Hello" in result["text"]
        assert "World" in result["text"]
    
    def test_expand_contractions(self):
        text = "I'm learning and it's awesome"
        result = self.cleaner.clean(text)
        assert "I am" in result["text"]
        assert "it is" in result["text"]
    
    def test_normalize_unicode(self):
        text = 'Smart quotes: "hello"'
        result = self.cleaner.clean(text)
        assert '"hello"' in result["text"]
    
    def test_stats_calculation(self):
        text = "Some text https://url.com with extra stuff"
        result = self.cleaner.clean(text)
        assert result["stats"]["original_length"] > result["stats"]["cleaned_length"]
        assert result["stats"]["reduction_percent"] > 0


class TestTempCleaner:
    """Tests for TempCleaner."""
    
    def setup_method(self):
        self.cleaner = TempCleaner()
    
    def test_remove_null_strings(self):
        text = "Value is null or undefined"
        result = self.cleaner.clean(text)
        assert "null" not in result["text"]
        assert "undefined" not in result["text"]
    
    def test_remove_short_lines(self):
        text = "This is a good line\nHi\nAnother good line here"
        result = self.cleaner.clean(text)
        assert "Hi" not in result["text"]
        assert "This is a good line" in result["text"]
    
    def test_remove_navigation(self):
        text = "Home\nActual content here"
        result = self.cleaner.clean(text)
        assert "Actual content here" in result["text"]
    
    def test_remove_symbol_lines(self):
        text = "Real content\n========\nMore content"
        result = self.cleaner.clean(text)
        assert "========" not in result["text"]
        assert "Real content" in result["text"]


class TestMarkdownConverter:
    """Tests for MarkdownConverter."""
    
    def setup_method(self):
        self.converter = MarkdownConverter()
    
    def test_structure_plain_text(self):
        text = "Title\nThis is the content under the title"
        result = self.converter.convert(text)
        assert result["text"]
        assert result["stats"]["original_length"] > 0
    
    def test_clean_empty_links(self):
        text = "[link title]() more text"
        result = self.converter.convert(text)
        assert "]()" not in result["text"]


class TestLanguageDetector:
    """Tests for LanguageDetector."""
    
    def setup_method(self):
        self.detector = LanguageDetector()
    
    def test_detect_english(self):
        text = "This is a sample English text for testing the language detection."
        result = self.detector.detect(text)
        assert result["status"] == "success"
        assert result["language_code"] == "en"
        assert result["script_type"] == "roman"
    
    def test_detect_script_roman(self):
        text = "Hello World in English"
        result = self.detector.detect(text)
        assert result["script_type"] == "roman"
    
    def test_short_text(self):
        text = "Hi"
        result = self.detector.detect(text)
        assert result["status"] == "skipped"


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_cleaning_pipeline(self):
        """Test cleaning without LLM (no API key needed)."""
        from src.cleaners import GlobalCleaner, TempCleaner
        
        raw_text = """
        <div>
            Check out https://example.com for more!
            
            I'm building an AI system and it's working great.
            
            Home | About | Contact
            
            ===============
            
            This is the main content that should be preserved.
            It contains useful information about technology.
            
            null
            
            Copyright 2024
        </div>
        """
        
        # Run global cleaning
        global_cleaner = GlobalCleaner()
        global_result = global_cleaner.clean(raw_text)
        
        # Run temp cleaning
        temp_cleaner = TempCleaner()
        temp_result = temp_cleaner.clean(global_result["text"])
        
        final_text = temp_result["text"]
        
        # Verify cleaning worked
        assert "https://example.com" not in final_text  # URL removed
        assert "<div>" not in final_text  # HTML removed
        assert "I am" in final_text  # Contraction expanded
        assert "======" not in final_text  # Separator removed
        assert "This is the main content" in final_text  # Content preserved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
