"""
SERAX Prompt Builder - Generates LLM prompts with SERAX format instructions.

Creates prompts that instruct LLMs to output in SERAX format
for reliable structured extraction.
"""

from typing import Dict, Any, List, Optional
from . import (
    SeraxSchema, SeraxDelimiters, FieldType,
    MULTI_TASK_SCHEMA, DOMAIN_DETECTION_SCHEMA, SENTIMENT_SCHEMA,
    NER_SCHEMA, SUMMARIZATION_SCHEMA, LANGUAGE_DETECTION_SCHEMA
)


class SeraxPromptBuilder:
    """
    Builds prompts that instruct LLMs to output in SERAX format.
    
    Features:
    - Automatic format instruction generation
    - Chain-of-thought reasoning integration
    - Multi-task prompt composition
    """
    
    def __init__(self):
        self.d = SeraxDelimiters
    
    def build_system_prompt(
        self,
        task_description: str,
        schema: SeraxSchema,
        include_cot: bool = True,
        examples: List[Dict[str, Any]] = None
    ) -> str:
        """
        Build a system prompt with SERAX format instructions.
        
        Args:
            task_description: Description of the task
            schema: SERAX schema for expected output
            include_cot: Include chain-of-thought instructions
            examples: Optional few-shot examples
        """
        parts = [
            f"# Task: {task_description}",
            "",
            "## Output Format",
            "You MUST output your response in SERAX format using these special delimiters:",
            f"- Start block: {self.d.BLOCK_START}",
            f"- Field start: {self.d.FIELD_START}",
            f"- Field end/value start: {self.d.FIELD_END}",
            f"- Field separator: {self.d.FIELD_SEP}",
            f"- End block: {self.d.BLOCK_END}",
            "",
            "## Expected Format",
            "```",
            schema.to_prompt_format(),
            "```",
            "",
            "## Field Definitions",
        ]
        
        for field in schema.fields:
            req = "required" if field.required else "optional"
            type_str = field.field_type.value
            if field.field_type == FieldType.ENUM:
                type_str = f"one of: {', '.join(field.enum_values)}"
            parts.append(f"- **{field.name}** ({req}, {type_str}): {field.description}")
        
        if include_cot:
            parts.extend([
                "",
                "## Chain-of-Thought Process",
                "Before outputting SERAX, think step-by-step:",
                "1. Read and understand the input carefully",
                "2. Identify all relevant information for each field",
                "3. Validate your findings",
                "4. Format output in SERAX with accurate values",
            ])
        
        if examples:
            parts.extend([
                "",
                "## Examples",
            ])
            for i, example in enumerate(examples, 1):
                parts.append(f"\n### Example {i}")
                if "input" in example:
                    parts.append(f"Input: {example['input'][:200]}...")
                if "output" in example:
                    parts.append(f"Output: {example['output']}")
        
        parts.extend([
            "",
            "## Important Rules",
            "1. ALWAYS use the exact SERAX delimiters shown above",
            "2. Include ALL required fields",
            "3. Keep values concise but accurate",
            "4. For lists, use [item1, item2, item3] format",
            "5. For confidence scores, use decimal 0.0-1.0",
        ])
        
        return "\n".join(parts)
    
    def build_user_prompt(
        self,
        text: str,
        context: Dict[str, Any] = None,
        max_text_length: int = 4000
    ) -> str:
        """
        Build a user prompt with the input text.
        
        Args:
            text: Input text to process
            context: Optional context from previous steps
            max_text_length: Maximum text length (truncates if longer)
        """
        parts = ["Analyze the following text and provide output in SERAX format:"]
        
        if context:
            parts.append("\n## Context from Previous Steps")
            for key, value in context.items():
                if not key.startswith("_"):
                    parts.append(f"- {key}: {value}")
        
        parts.append("\n## Text to Analyze")
        
        if len(text) > max_text_length:
            parts.append(f"{text[:max_text_length]}...")
            parts.append(f"\n[Truncated from {len(text)} to {max_text_length} chars]")
        else:
            parts.append(text)
        
        parts.append("\n## Your SERAX Output")
        
        return "\n".join(parts)
    
    def build_multi_task_prompt(
        self,
        text: str,
        tasks: List[str] = None,
        context: Dict[str, Any] = None
    ) -> tuple[str, str]:
        """
        Build prompts for multi-task extraction.
        
        Args:
            text: Input text
            tasks: List of tasks to perform (default: all)
            context: Previous step context
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        default_tasks = [
            "named_entity_recognition",
            "domain_classification",
            "sentiment_analysis",
            "summarization",
            "language_detection",
            "relevancy_scoring"
        ]
        
        tasks = tasks or default_tasks
        
        task_description = f"Perform the following NLP tasks: {', '.join(tasks)}"
        
        system_prompt = self.build_system_prompt(
            task_description=task_description,
            schema=MULTI_TASK_SCHEMA,
            include_cot=True
        )
        
        user_prompt = self.build_user_prompt(text, context)
        
        return system_prompt, user_prompt
    
    def build_single_task_prompt(
        self,
        task_name: str,
        text: str,
        context: Dict[str, Any] = None
    ) -> tuple[str, str]:
        """
        Build prompts for a single task.
        
        Args:
            task_name: Name of the task
            text: Input text
            context: Previous step context
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        schemas = {
            "domain_detection": DOMAIN_DETECTION_SCHEMA,
            "sentiment_analysis": SENTIMENT_SCHEMA,
            "named_entity_recognition": NER_SCHEMA,
            "summarization": SUMMARIZATION_SCHEMA,
            "language_detection": LANGUAGE_DETECTION_SCHEMA,
        }
        
        schema = schemas.get(task_name, DOMAIN_DETECTION_SCHEMA)
        
        task_descriptions = {
            "domain_detection": "Classify the text into a domain category",
            "sentiment_analysis": "Analyze the sentiment of the text",
            "named_entity_recognition": "Extract named entities from the text",
            "summarization": "Summarize the key points of the text",
            "language_detection": "Detect the language and script of the text",
        }
        
        task_desc = task_descriptions.get(task_name, f"Perform {task_name}")
        
        system_prompt = self.build_system_prompt(
            task_description=task_desc,
            schema=schema,
            include_cot=True
        )
        
        user_prompt = self.build_user_prompt(text, context)
        
        return system_prompt, user_prompt


# ============== Pre-built Prompt Templates ==============

DOMAIN_CLASSIFICATION_PROMPT = """
# Domain Classification with SERAX Output

Classify the text into ONE of these domains:
- **technology**: Software, AI, engineering, programming, IT
- **business**: Companies, products, finance, marketing, corporate
- **general**: News, entertainment, lifestyle, education, other

Think step-by-step:
1. Identify key topics and terminology
2. Look for domain-specific indicators
3. Consider the overall purpose

Output format:
⟐⊶domain⊷[technology/business/general]⊸⊶confidence⊷[0.0-1.0]⊸⊶subcats⊷[list]⊸⊶reasoning⊷[brief explanation]⊹
"""

SENTIMENT_ANALYSIS_PROMPT = """
# Sentiment Analysis with SERAX Output

Analyze the sentiment of the text:
- **positive**: Favorable, optimistic, satisfied
- **negative**: Unfavorable, pessimistic, dissatisfied
- **neutral**: Objective, factual, balanced
- **mixed**: Contains both positive and negative

Output format:
⟐⊶sentiment⊷[positive/negative/neutral/mixed]⊸⊶score⊷[-1.0 to 1.0]⊸⊶confidence⊷[0.0-1.0]⊸⊶aspects⊷[list of aspect sentiments]⊹
"""

NER_EXTRACTION_PROMPT = """
# Named Entity Recognition with SERAX Output

Extract named entities by category:
- **persons**: People names
- **orgs**: Organizations, companies
- **locations**: Places, addresses
- **dates**: Dates, times, periods
- **other**: Products, events, misc

Output format:
⟐⊶persons⊷[list]⊸⊶orgs⊷[list]⊸⊶locations⊷[list]⊸⊶dates⊷[list]⊸⊶other⊷[list]⊹
"""

MULTI_TASK_PROMPT = """
# Multi-Task NLP Extraction with SERAX Output

Perform comprehensive text analysis:

1. **Entity Extraction**: Extract named entities
2. **Domain Classification**: Classify into technology/business/general
3. **Sentiment Analysis**: Determine positive/negative/neutral sentiment
4. **Summarization**: Create brief summary
5. **Event Extraction**: Identify key events and dates
6. **Language Detection**: Detect language and script type
7. **Relevancy Scoring**: Score relevancy 0-1

Think through each task systematically, then output all results.

Output format:
⟐⊶entities⊷[Type: Name, ...]⊸⊶domain⊷[category]⊸⊶domain_conf⊷[0-1]⊸⊶sentiment⊷[pos/neg/neu]⊸⊶sent_score⊷[-1 to 1]⊸⊶summary⊷[text]⊸⊶events⊷[date: event, ...]⊸⊶lang⊷[code]⊸⊶script⊷[roman/non_roman/mixed]⊸⊶relevancy⊷[0-1]⊹
"""
