# Chain of Thought Text Processing Pipeline

A modern, modular text processing pipeline using **Chain-of-Thought (CoT)** reasoning and **function calling** with Groq's LLM.

## ✨ Features

- **🔗 Chain-of-Thought Reasoning** - Explicit step-by-step reasoning at each stage
- **📋 Function Calling** - Structured JSON outputs via function definitions
- **🔄 Context Passing** - Results from each step feed into the next
- **✅ Validation & Feedback** - Built-in output validation with retry logic
- **🎯 Self-Consistency** - Optional multi-run voting for reliability
- **⚡ Fast Local Processing** - Text cleaning and language detection run locally

## Quick Start

```bash
cd chain_of_thoughts
pip install -r requirements.txt

# Copy and configure API key (optional - only for domain detection)
cp .env.example .env

# Run the pipeline
python main.py --text "Your text here"

# Skip LLM steps (fast mode)
python main.py -i input.txt --skip-domain

# Full pipeline with API key
python main.py -i input.txt --api-key YOUR_GROQ_KEY
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Chain-of-Thought Pipeline v2.0                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Step 1              Step 2              Step 3              │
│  ┌─────────┐        ┌─────────┐        ┌─────────┐           │
│  │  TEXT   │ ────▶  │ DOMAIN  │ ────▶  │  LANG   │           │
│  │CLEANING │        │DETECTION│        │DETECTION│           │
│  └─────────┘        └─────────┘        └─────────┘           │
│       │                  │                  │                 │
│       ▼                  ▼                  ▼                 │
│  ┌─────────┐        ┌─────────┐        ┌─────────┐           │
│  │ Local   │        │ Groq    │        │ Local   │           │
│  │Functions│        │   LLM   │        │langdetect│          │
│  └─────────┘        └─────────┘        └─────────┘           │
│                                                                │
│                    ┌─────────────┐                            │
│   Step 4:          │ VALIDATION  │  ◀── Feedback Loop         │
│                    └─────────────┘                            │
│                                                                │
│   OUTPUT: Structured JSON with reasoning at each step          │
└─────────────────────────────────────────────────────────────────┘
```

## Function Calling & Structured Outputs

Each step uses a **function definition** with JSON schema for guaranteed structured output:

```python
# Domain Detection Function Definition
DOMAIN_DETECT_FUNCTION = FunctionDefinition(
    name="detect_domain",
    description="Classify text into technology, business, or general",
    parameters={
        "primary_domain": {"type": "string", "enum": ["technology", "business", "general"]},
        "confidence": {"type": "number"},
        "domain_scores": {"type": "object"},
        "reasoning": {"type": "string"}
    },
    required=["primary_domain", "confidence", "reasoning"]
)
```

## Usage Examples

### Python API

```python
from src.cot.pipeline import CoTPipeline, create_pipeline

# Create pipeline
pipeline = create_pipeline(api_key="your-key")

# Process text
result = pipeline.run("""
    Check out https://example.com for more!
    I'm building an AI system using TensorFlow.
""")

# Access step results
print(result["1_text_cleaning"]["output"]["cleaned_text"])
print(result["2_domain_detection"]["output"]["primary_domain"])
print(result["3_language_detection"]["output"]["language_name"])

# Get chain-of-thought summary
print(result["chain_of_thought_summary"])
```

### CLI Options

```bash
# Basic processing
python main.py --text "Your text"

# Process file, output to JSON
python main.py -i input.txt -o result.json

# Skip domain detection (no API needed)
python main.py -i input.txt --skip-domain

# Self-consistency mode (3 runs, majority vote)
python main.py -i input.txt --self-consistency

# Run single step only
python main.py --text "Text" --step domain_detection

# Get only the reasoning chain
python main.py --text "Text" --reasoning-only

# Use legacy v1.0 pipeline
python main.py --text "Text" --legacy
```

## Output Format

```json
{
  "1_text_cleaning": {
    "status": "success",
    "output": {
      "cleaned_text": "Cleaned text here...",
      "removed_elements": ["urls", "html_tags"],
      "original_length": 500,
      "cleaned_length": 350,
      "reduction_percent": 30.0
    },
    "reasoning": "Applied global cleaning (removed urls, html_tags)...",
    "confidence": 1.0,
    "duration_ms": 5
  },
  "2_domain_detection": {
    "status": "success",
    "output": {
      "primary_domain": "technology",
      "confidence": 0.92,
      "domain_scores": {"technology": 0.92, "business": 0.05, "general": 0.03},
      "sub_categories": ["artificial-intelligence", "software"],
      "reasoning": "Key indicators: AI, TensorFlow, machine learning..."
    },
    "reasoning": "Classified as technology based on AI/ML terminology...",
    "confidence": 0.92,
    "duration_ms": 850
  },
  "3_language_detection": {
    "status": "success",
    "output": {
      "language_code": "en",
      "language_name": "English",
      "script_type": "roman",
      "confidence": 1.0
    },
    "reasoning": "Detected English (en) with roman script at 100% confidence.",
    "confidence": 1.0,
    "duration_ms": 15
  },
  "4_validation": {
    "status": "success",
    "output": {
      "is_valid": true,
      "issues_found": [],
      "quality_score": 1.0
    },
    "reasoning": "Validation passed: all checks passed",
    "confidence": 1.0
  },
  "metadata": {
    "pipeline_version": "2.0.0",
    "pipeline_type": "chain_of_thought",
    "model_used": "llama-3.3-70b-versatile",
    "total_duration_ms": 875,
    "steps_executed": 4
  },
  "chain_of_thought_summary": "- text_cleaning: Applied global cleaning...\n- domain_detection: Classified as technology...\n- language_detection: Detected English..."
}
```

## Project Structure

```
chain_of_thoughts/
├── src/
│   ├── cot/                        # 🆕 Chain-of-Thought Framework
│   │   ├── __init__.py            # Core abstractions, function definitions
│   │   ├── executor.py            # Step executor with validation
│   │   ├── steps.py               # Concrete step implementations
│   │   └── pipeline.py            # Main CoT pipeline runner
│   │
│   ├── cleaners/                   # Text cleaning modules
│   │   ├── global_cleaner.py      # URL, HTML, contractions
│   │   └── temp_cleaner.py        # Short lines, navigation
│   │
│   ├── processors/                 # Processing modules
│   │   ├── domain_detector.py     # Domain classification
│   │   └── language_detector.py   # Language detection
│   │
│   ├── utils/                      # Utilities
│   │   ├── groq_client.py         # Groq API wrapper
│   │   └── markdown_converter.py  # Text → Markdown
│   │
│   ├── pipeline.py                # Legacy v1.0 pipeline
│   └── config.py                  # Configuration
│
├── tests/
├── main.py                        # CLI entry point
├── requirements.txt
└── .env.example
```

## Key Concepts

### Chain-of-Thought Prompting
Each step includes explicit CoT instructions:
```
Think step-by-step:
1. First, understand what is being asked
2. Then, analyze the input systematically
3. Consider edge cases and potential issues
4. Finally, provide your structured output
```

### Context Passing
Results from each step are passed to subsequent steps:
```python
context = PipelineContext(original_input=text)

# Step 1 adds its result
context.add_result(text_cleaning_result)

# Step 2 can access previous results
previous = context.get_step_output("text_cleaning")
```

### Self-Consistency
For critical decisions, run multiple times and vote:
```bash
python main.py --text "..." --self-consistency
```
This runs domain detection 3 times with varied temperatures and returns the majority vote.

### Validation & Feedback
Built-in validation checks outputs and can trigger retries:
- Schema validation (required fields, types, enums)
- Consistency checks (scores sum to 1.0)
- Quality validation (not too aggressive cleaning)

## Extending the Pipeline

Add new steps easily:

```python
from src.cot import PipelineStep, FunctionDefinition, StepResult

class SentimentStep(PipelineStep):
    def get_function_definition(self):
        return FunctionDefinition(
            name="analyze_sentiment",
            parameters={"sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]}}
        )
    
    def get_cot_prompt(self, context):
        return f"Analyze sentiment of: {context.current_text}"
    
    def execute(self, context):
        # Your implementation
        pass
```

## License

MIT
