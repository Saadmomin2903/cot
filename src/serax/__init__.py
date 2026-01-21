"""
SERAX (Structured Extraction with Rare-character Anchors for eXtraction) Format

A robust format for extracting structured data from LLM outputs using rare Unicode
delimiters that won't appear in natural text or LLM-generated content.

SERAX Delimiters:
    ⟐  (U+27D0) - SERAX block start
    ⊶  (U+22B6) - Field name start
    ⊷  (U+22B7) - Field name end / value start  
    ⊸  (U+22B8) - Field separator
    ⊹  (U+22B9) - SERAX block end

Example:
    ⟐⊶entities⊷[Person: John, Org: Tesla]⊸⊶domain⊷technology⊸⊶sentiment⊷positive⊹

Benefits:
    - Rare Unicode characters prevent parsing failures with complex content
    - Works with multilingual text, code blocks, quotes, brackets
    - Battle-tested for production NLP pipelines
    - Semantic type validation built-in
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Type
from enum import Enum


# SERAX Delimiter Constants
class SeraxDelimiters:
    """SERAX format delimiter constants using rare Unicode characters."""
    BLOCK_START = "⟐"      # U+27D0 - White diamond with centred dot
    FIELD_START = "⊶"      # U+22B6 - Original of
    FIELD_END = "⊷"        # U+22B7 - Image of
    FIELD_SEP = "⊸"        # U+22B8 - Multimap
    BLOCK_END = "⊹"        # U+22B9 - Hermitian conjugate matrix
    
    # Alternative delimiters (backup set)
    ALT_BLOCK_START = "⧫"  # U+29EB
    ALT_FIELD_START = "⦃"  # U+2983
    ALT_FIELD_END = "⦄"    # U+2984
    ALT_FIELD_SEP = "⦁"    # U+2981
    ALT_BLOCK_END = "⧫"    # U+29EB


class FieldType(Enum):
    """Supported field types for semantic validation."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    DATE = "date"
    ENUM = "enum"


@dataclass
class FieldDefinition:
    """Definition of a SERAX field with type validation."""
    name: str
    field_type: FieldType
    required: bool = True
    default: Any = None
    enum_values: List[str] = field(default_factory=list)
    description: str = ""
    
    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate a value against this field definition."""
        if value is None:
            if self.required and self.default is None:
                return False, f"Required field '{self.name}' is missing"
            return True, ""
        
        try:
            if self.field_type == FieldType.STRING:
                return isinstance(value, str), f"Expected string, got {type(value).__name__}"
            
            elif self.field_type == FieldType.INTEGER:
                int(value)
                return True, ""
            
            elif self.field_type == FieldType.FLOAT:
                float(value)
                return True, ""
            
            elif self.field_type == FieldType.BOOLEAN:
                if isinstance(value, bool):
                    return True, ""
                if str(value).lower() in ("true", "false", "yes", "no", "1", "0"):
                    return True, ""
                return False, f"Expected boolean, got {value}"
            
            elif self.field_type == FieldType.LIST:
                # Lists can be comma-separated or bracket-enclosed
                return True, ""
            
            elif self.field_type == FieldType.ENUM:
                if str(value).lower() in [v.lower() for v in self.enum_values]:
                    return True, ""
                return False, f"Value '{value}' not in enum {self.enum_values}"
            
            return True, ""
            
        except (ValueError, TypeError) as e:
            return False, str(e)


@dataclass
class SeraxSchema:
    """Schema definition for SERAX structured output."""
    name: str
    fields: List[FieldDefinition]
    description: str = ""
    
    def to_prompt_format(self) -> str:
        """Generate the SERAX format instruction for prompts."""
        parts = [SeraxDelimiters.BLOCK_START]
        
        for i, field_def in enumerate(self.fields):
            parts.append(f"{SeraxDelimiters.FIELD_START}{field_def.name}{SeraxDelimiters.FIELD_END}")
            
            # Add placeholder based on type
            if field_def.field_type == FieldType.LIST:
                parts.append("[...]")
            elif field_def.field_type == FieldType.ENUM:
                parts.append(f"[{'/'.join(field_def.enum_values)}]")
            elif field_def.field_type == FieldType.FLOAT:
                parts.append("[0.0-1.0]")
            elif field_def.field_type == FieldType.INTEGER:
                parts.append("[number]")
            elif field_def.field_type == FieldType.BOOLEAN:
                parts.append("[true/false]")
            else:
                parts.append("[text]")
            
            if i < len(self.fields) - 1:
                parts.append(SeraxDelimiters.FIELD_SEP)
        
        parts.append(SeraxDelimiters.BLOCK_END)
        return "".join(parts)
    
    def get_field(self, name: str) -> Optional[FieldDefinition]:
        """Get field definition by name."""
        for f in self.fields:
            if f.name == name:
                return f
        return None


class SeraxParser:
    """
    Parser for SERAX format structured outputs.
    
    Extracts structured data from LLM outputs using rare Unicode delimiters.
    """
    
    def __init__(self, schema: SeraxSchema = None):
        """Initialize parser with optional schema for validation."""
        self.schema = schema
        self.d = SeraxDelimiters
        
        # Build regex pattern for parsing
        # Pattern: ⟐ ... content ... ⊹
        self._block_pattern = re.compile(
            f'{re.escape(self.d.BLOCK_START)}(.*?){re.escape(self.d.BLOCK_END)}',
            re.DOTALL
        )
        
        # Pattern for individual fields: ⊶name⊷value
        self._field_pattern = re.compile(
            f'{re.escape(self.d.FIELD_START)}([^{re.escape(self.d.FIELD_END)}]+){re.escape(self.d.FIELD_END)}([^{re.escape(self.d.FIELD_SEP)}{re.escape(self.d.BLOCK_END)}]*)',
            re.DOTALL
        )
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse SERAX formatted text and extract fields.
        
        Args:
            text: Text containing SERAX block
            
        Returns:
            Dictionary of extracted field values
        """
        result = {}
        errors = []
        
        # Find SERAX block
        block_match = self._block_pattern.search(text)
        if not block_match:
            # Try to find partial match or undelimited content
            return self._parse_fallback(text)
        
        content = block_match.group(1)
        
        # Split by field separator and parse each field
        # Handle both ⊸⊶ (sep + new field) and just fields
        field_matches = self._field_pattern.findall(content)
        
        for field_name, field_value in field_matches:
            field_name = field_name.strip()
            field_value = field_value.strip()
            
            # Parse value based on schema or auto-detect
            parsed_value = self._parse_value(field_name, field_value)
            result[field_name] = parsed_value
        
        # Validate against schema if provided
        if self.schema:
            result, validation_errors = self._validate_result(result)
            errors.extend(validation_errors)
        
        # Add metadata
        result["_serax_meta"] = {
            "parsed_fields": len(result) - 1,  # Exclude meta
            "validation_errors": errors,
            "raw_block": content[:200] if len(content) > 200 else content
        }
        
        return result
    
    def _parse_value(self, field_name: str, raw_value: str) -> Any:
        """Parse a field value, applying type conversion if schema exists."""
        if not raw_value:
            return None
        
        # Get field definition from schema
        field_def = self.schema.get_field(field_name) if self.schema else None
        
        if field_def:
            return self._convert_type(raw_value, field_def.field_type, field_def.enum_values)
        
        # Auto-detect type
        return self._auto_parse(raw_value)
    
    def _convert_type(self, value: str, field_type: FieldType, enum_values: List[str] = None) -> Any:
        """Convert string value to specified type."""
        value = value.strip()
        
        if field_type == FieldType.INTEGER:
            try:
                # Handle potential floats that should be ints
                return int(float(value))
            except ValueError:
                return value
        
        elif field_type == FieldType.FLOAT:
            try:
                return round(float(value), 4)
            except ValueError:
                return value
        
        elif field_type == FieldType.BOOLEAN:
            return value.lower() in ("true", "yes", "1")
        
        elif field_type == FieldType.LIST:
            return self._parse_list(value)
        
        elif field_type == FieldType.ENUM:
            # Normalize to match enum values
            for ev in (enum_values or []):
                if value.lower() == ev.lower():
                    return ev
            return value
        
        return value
    
    def _parse_list(self, value: str) -> List[str]:
        """Parse a list value from various formats."""
        # Remove brackets if present
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        if value.startswith("(") and value.endswith(")"):
            value = value[1:-1]
        
        # Split by comma or semicolon
        items = re.split(r'[,;]', value)
        return [item.strip() for item in items if item.strip()]
    
    def _auto_parse(self, value: str) -> Any:
        """Auto-detect and parse value type."""
        value = value.strip()
        
        # Boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        # Integer
        try:
            if "." not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            float_val = float(value)
            if 0 <= float_val <= 1 and "." in value:
                return round(float_val, 4)
            return float_val
        except ValueError:
            pass
        
        # List (bracket or comma-separated)
        if value.startswith("[") or "," in value:
            return self._parse_list(value)
        
        return value
    
    def _validate_result(self, result: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        """Validate parsed result against schema."""
        errors = []
        validated = {}
        
        for field_def in self.schema.fields:
            value = result.get(field_def.name)
            
            if value is None:
                if field_def.required:
                    if field_def.default is not None:
                        validated[field_def.name] = field_def.default
                    else:
                        errors.append(f"Missing required field: {field_def.name}")
                continue
            
            is_valid, error = field_def.validate(value)
            if not is_valid:
                errors.append(f"Field '{field_def.name}': {error}")
            
            validated[field_def.name] = value
        
        # Include any extra fields not in schema
        for key, value in result.items():
            if key not in validated:
                validated[key] = value
        
        return validated, errors
    
    def _parse_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback parsing when proper SERAX block not found."""
        # Try to extract any field patterns
        result = {}
        field_matches = self._field_pattern.findall(text)
        
        for field_name, field_value in field_matches:
            result[field_name.strip()] = self._auto_parse(field_value.strip())
        
        if not result:
            result["_raw_text"] = text[:500]
        
        result["_serax_meta"] = {
            "parsed_fields": len(result) - 1,
            "fallback_mode": True,
            "validation_errors": ["No valid SERAX block found"]
        }
        
        return result


class SeraxFormatter:
    """
    Formatter for creating SERAX output strings.
    
    Used to format structured data into SERAX format for LLM training
    or for consistent output formatting.
    """
    
    def __init__(self):
        self.d = SeraxDelimiters
    
    def format(self, data: Dict[str, Any]) -> str:
        """
        Format a dictionary into SERAX format.
        
        Args:
            data: Dictionary of field names to values
            
        Returns:
            SERAX formatted string
        """
        parts = [self.d.BLOCK_START]
        
        items = [(k, v) for k, v in data.items() if not k.startswith("_")]
        
        for i, (key, value) in enumerate(items):
            parts.append(f"{self.d.FIELD_START}{key}{self.d.FIELD_END}")
            parts.append(self._format_value(value))
            
            if i < len(items) - 1:
                parts.append(self.d.FIELD_SEP)
        
        parts.append(self.d.BLOCK_END)
        return "".join(parts)
    
    def _format_value(self, value: Any) -> str:
        """Format a value for SERAX output."""
        if value is None:
            return ""
        
        if isinstance(value, bool):
            return str(value).lower()
        
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(str(v) for v in value) + "]"
        
        if isinstance(value, dict):
            pairs = [f"{k}: {v}" for k, v in value.items()]
            return "{" + ", ".join(pairs) + "}"
        
        if isinstance(value, float):
            return f"{value:.4f}"
        
        return str(value)


# ============== Pre-defined Schemas for Pipeline ==============

# Schema for text cleaning output
TEXT_CLEANING_SCHEMA = SeraxSchema(
    name="text_cleaning",
    description="Text cleaning and normalization output",
    fields=[
        FieldDefinition("cleaned_text", FieldType.STRING, description="Cleaned text content"),
        FieldDefinition("removed", FieldType.LIST, description="Types of elements removed"),
        FieldDefinition("original_len", FieldType.INTEGER, description="Original character count"),
        FieldDefinition("cleaned_len", FieldType.INTEGER, description="Cleaned character count"),
        FieldDefinition("reduction", FieldType.FLOAT, description="Reduction percentage 0-1"),
    ]
)

# Schema for domain detection output
DOMAIN_DETECTION_SCHEMA = SeraxSchema(
    name="domain_detection",
    description="Domain classification output",
    fields=[
        FieldDefinition(
            "domain", FieldType.ENUM, 
            enum_values=["technology", "business", "general"],
            description="Primary domain classification"
        ),
        FieldDefinition("confidence", FieldType.FLOAT, description="Confidence score 0-1"),
        FieldDefinition("subcats", FieldType.LIST, description="Sub-categories"),
        FieldDefinition("reasoning", FieldType.STRING, required=False, description="Classification reasoning"),
    ]
)

# Schema for language detection output
LANGUAGE_DETECTION_SCHEMA = SeraxSchema(
    name="language_detection",
    description="Language and script detection output",
    fields=[
        FieldDefinition("lang", FieldType.STRING, description="ISO 639-1 language code"),
        FieldDefinition("lang_name", FieldType.STRING, description="Full language name"),
        FieldDefinition(
            "script", FieldType.ENUM,
            enum_values=["roman", "non_roman", "mixed"],
            description="Script type"
        ),
        FieldDefinition("confidence", FieldType.FLOAT, description="Detection confidence 0-1"),
    ]
)

# Schema for NER output
NER_SCHEMA = SeraxSchema(
    name="named_entity_recognition",
    description="Named entity extraction output",
    fields=[
        FieldDefinition("persons", FieldType.LIST, required=False, description="Person entities"),
        FieldDefinition("orgs", FieldType.LIST, required=False, description="Organization entities"),
        FieldDefinition("locations", FieldType.LIST, required=False, description="Location entities"),
        FieldDefinition("dates", FieldType.LIST, required=False, description="Date entities"),
        FieldDefinition("other", FieldType.LIST, required=False, description="Other entities"),
    ]
)

# Schema for sentiment analysis output
SENTIMENT_SCHEMA = SeraxSchema(
    name="sentiment_analysis",
    description="Sentiment analysis output",
    fields=[
        FieldDefinition(
            "sentiment", FieldType.ENUM,
            enum_values=["positive", "negative", "neutral", "mixed"],
            description="Overall sentiment"
        ),
        FieldDefinition("score", FieldType.FLOAT, description="Sentiment score -1 to 1"),
        FieldDefinition("confidence", FieldType.FLOAT, description="Confidence score 0-1"),
        FieldDefinition("aspects", FieldType.LIST, required=False, description="Aspect sentiments"),
    ]
)

# Schema for summarization output
SUMMARIZATION_SCHEMA = SeraxSchema(
    name="summarization",
    description="Text summarization output",
    fields=[
        FieldDefinition("summary", FieldType.STRING, description="Summarized text"),
        FieldDefinition("key_points", FieldType.LIST, required=False, description="Key points"),
        FieldDefinition("compression", FieldType.FLOAT, description="Compression ratio"),
    ]
)

# Schema for relevancy scoring output
RELEVANCY_SCHEMA = SeraxSchema(
    name="relevancy_scoring",
    description="Relevancy scoring output",
    fields=[
        FieldDefinition("score", FieldType.FLOAT, description="Relevancy score 0-1"),
        FieldDefinition("category", FieldType.STRING, required=False, description="Relevancy category"),
        FieldDefinition("reasoning", FieldType.STRING, required=False, description="Scoring reasoning"),
    ]
)

# Combined schema for full multi-task output
MULTI_TASK_SCHEMA = SeraxSchema(
    name="multi_task_extraction",
    description="Combined multi-task NLP extraction output",
    fields=[
        # Entities
        FieldDefinition("entities", FieldType.LIST, description="Named entities [Type: Name]"),
        # Domain
        FieldDefinition("domain", FieldType.ENUM, enum_values=["technology", "business", "general"]),
        FieldDefinition("domain_conf", FieldType.FLOAT, description="Domain confidence"),
        # Sentiment
        FieldDefinition("sentiment", FieldType.ENUM, enum_values=["positive", "negative", "neutral", "mixed"]),
        FieldDefinition("sent_score", FieldType.FLOAT, description="Sentiment score -1 to 1"),
        # Summary
        FieldDefinition("summary", FieldType.STRING, description="Brief summary"),
        # Events
        FieldDefinition("events", FieldType.LIST, required=False, description="Key events with dates"),
        # Language
        FieldDefinition("lang", FieldType.STRING, description="ISO language code"),
        FieldDefinition("script", FieldType.ENUM, enum_values=["roman", "non_roman", "mixed"]),
        # Relevancy
        FieldDefinition("relevancy", FieldType.FLOAT, description="Relevancy score 0-1"),
    ]
)
