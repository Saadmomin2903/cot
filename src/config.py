"""
Configuration management for Chain of Thought Pipeline.

Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class GroqConfig:
    """Groq LLM API configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    temperature: float = 0.1  # Low temperature for consistent classification
    max_tokens: int = 1024


@dataclass
class CleaningConfig:
    """Text cleaning configuration."""
    # Minimum line length to keep (shorter lines are removed in temp cleaning)
    min_line_length: int = field(
        default_factory=lambda: int(os.getenv("MIN_LINE_LENGTH", "10"))
    )
    
    # Patterns to remove in global cleaning
    url_pattern: str = r'https?://\S+|www\.\S+'
    html_tag_pattern: str = r'<[^>]+>'
    bracketed_pattern: str = r'\[[^\[\]]*?\]'
    
    # Strings to remove
    null_strings: List[str] = field(default_factory=lambda: [
        "null", "undefined", "NaN", "None", "N/A", "n/a"
    ])
    
    # Navigation patterns to remove
    nav_patterns: List[str] = field(default_factory=lambda: [
        r'^(Home|Menu|Search|Login|Sign Up|Subscribe|Contact|About)$',
        r'^(Skip to content|Skip navigation)$',
        r'^\s*(©|Copyright)\s*\d{4}',
        r'^Cookie Policy$',
        r'^Privacy Policy$',
        r'^Terms of Service$',
    ])


@dataclass
class DomainConfig:
    """Domain detection configuration."""
    domains: List[str] = field(default_factory=lambda: [
        "technology",
        "business", 
        "general"
    ])
    
    domain_descriptions: dict = field(default_factory=lambda: {
        "technology": "Software, hardware, programming, engineering, IT, web development, AI/ML",
        "business": "Companies, products, services, e-commerce, finance, marketing, corporate",
        "general": "News, education, entertainment, lifestyle, health, sports, other"
    })


@dataclass
class PipelineConfig:
    """Main pipeline configuration combining all settings."""
    groq: GroqConfig = field(default_factory=GroqConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    
    # Feature flags
    enable_markdown_conversion: bool = field(
        default_factory=lambda: os.getenv("ENABLE_MARKDOWN_CONVERSION", "true").lower() == "true"
    )
    
    # Pipeline version
    version: str = "1.0.0"


# Global config instance
config = PipelineConfig()


def validate_config() -> bool:
    """Validate that required configuration is present."""
    if not config.groq.api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Please set it in .env file or environment variables.\n"
            "Get your API key at: https://console.groq.com/keys"
        )
    return True
