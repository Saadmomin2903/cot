#!/usr/bin/env python3
"""
Chain of Thought Text Processing Pipeline v3.0

Modern implementation with:
- SERAX format (rare Unicode delimiters) for reliable extraction
- JSON fallback option
- Function calling with structured outputs
- Chain-of-thought reasoning at each step
- Multi-task NLP in single call (NER, domain, sentiment, summary, events, language)
- Self-consistency checks (optional)

Usage:
    # Basic usage (SERAX format)
    python main.py --text "Your text here"
    
    # JSON format output
    python main.py --text "Your text" --format json
    
    # Skip LLM tasks (local only)
    python main.py -i input.txt --skip-llm
    
    # Multi-task extraction
    python main.py -i input.txt --multi-task
    
    # Specific tasks only
    python main.py -i input.txt --tasks domain,sentiment,summary
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for proper imports
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Chain of Thought Text Processing Pipeline v3.0 (SERAX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--input",
        type=str,
        help="Path to input text file"
    )
    input_group.add_argument(
        "-t", "--text",
        type=str,
        help="Direct text input"
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to output file (default: stdout)"
    )
    
    # Format options
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "serax"],
        default="json",
        help="Output format: json (default) or serax"
    )
    
    # Pipeline selection
    parser.add_argument(
        "--multi-task",
        action="store_true",
        help="Use unified multi-task pipeline (NER, domain, sentiment, etc.)"
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Use CoT v2 pipeline (function calling)"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy v1.0 pipeline"
    )
    
    # Task configuration
    parser.add_argument(
        "--tasks",
        type=str,
        help="Comma-separated list of tasks: ner,domain,sentiment,summary,events,language,relevancy"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip all LLM-based tasks (only run local cleaning/language detection)"
    )
    parser.add_argument(
        "--skip-domain",
        action="store_true",
        help="Skip domain detection"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation step"
    )
    parser.add_argument(
        "--semantic-clean",
        action="store_true",
        help="Use LLM-powered semantic cleaning (preserves important content, expands abbreviations)"
    )
    parser.add_argument(
        "--ner",
        action="store_true",
        help="Enable Named Entity Recognition with ERA-CoT (extracts entities and relationships)"
    )
    parser.add_argument(
        "--no-relationships",
        action="store_true",
        help="Disable relationship extraction in NER (entities only)"
    )
    parser.add_argument(
        "--events",
        action="store_true",
        help="Extract temporal events and create calendar (dates, deadlines, announcements)"
    )
    parser.add_argument(
        "--sentiment",
        action="store_true",
        help="Analyze sentiment with emotion detection (positive/negative/neutral/mixed)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate text summary with key points"
    )
    parser.add_argument(
        "--summary-style",
        type=str,
        choices=["bullets", "paragraph", "executive", "headlines", "tldr"],
        default="bullets",
        help="Summary style: bullets (default), paragraph, executive, headlines, tldr"
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate content to English before processing"
    )
    parser.add_argument(
        "--relevancy",
        action="store_true",
        help="Analyze text relevancy to topics and concepts"
    )
    parser.add_argument(
        "--topics",
        type=str,
        help="Comma-separated list of custom topics for relevancy analysis"
    )
    
    # Advanced options
    parser.add_argument(
        "--self-consistency",
        action="store_true",
        help="Use self-consistency for LLM tasks (3 runs, majority vote)"
    )
    parser.add_argument(
        "--step",
        type=str,
        help="Run only a specific step"
    )
    parser.add_argument(
        "--reasoning-only",
        action="store_true",
        help="Only output the chain-of-thought reasoning"
    )
    
    # Output formatting
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Output compact format (no indentation)"
    )
    parser.add_argument(
        "--show-serax",
        action="store_true",
        help="Show SERAX formatted output (raw delimiters)"
    )
    
    # API configuration
    parser.add_argument(
        "--api-key",
        type=str,
        help="Groq API key (overrides .env)"
    )
    
    args = parser.parse_args()
    
    # Get input text
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text
    
    # Select pipeline based on options
    if args.multi_task:
        result = run_multi_task_pipeline(text, args)
    elif args.cot:
        result = run_cot_pipeline(text, args)
    elif args.legacy:
        result = run_legacy_pipeline(text, args)
    else:
        # Default: CoT pipeline
        result = run_cot_pipeline(text, args)
    
    # Handle reasoning-only output
    if args.reasoning_only:
        reasoning = result.get("chain_of_thought_summary", "")
        print(reasoning)
        return
    
    # Format output
    if args.show_serax and args.format == "serax":
        # Show raw SERAX output
        from src.serax import SeraxFormatter
        formatter = SeraxFormatter()
        flat_data = {}
        for key, value in result.items():
            if isinstance(value, dict) and "output" in value:
                flat_data.update(value["output"])
        serax_output = formatter.format(flat_data)
        print(serax_output)
    else:
        # JSON output
        if args.compact:
            json_output = json.dumps(result, ensure_ascii=False, default=str)
        else:
            json_output = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_output)
            print(f"✅ Output written to: {args.output}", file=sys.stderr)
        else:
            print(json_output)
    
    # Print summary
    print_summary(result, args)


def run_multi_task_pipeline(text: str, args) -> dict:
    """Run the unified multi-task pipeline with SERAX support."""
    from src.serax.unified_pipeline import UnifiedPipeline, PipelineOptions, OutputFormat
    
    print("🔗 Running Multi-Task Pipeline (SERAX)...", file=sys.stderr)
    
    # Parse task list
    tasks = args.tasks.split(",") if args.tasks else None
    
    # Build options
    fmt = OutputFormat.SERAX if args.format == "serax" else OutputFormat.JSON
    options = PipelineOptions(
        output_format=fmt,
        enable_ner="ner" in tasks if tasks else not args.skip_llm,
        enable_domain="domain" in tasks if tasks else (not args.skip_llm and not args.skip_domain),
        enable_sentiment="sentiment" in tasks if tasks else not args.skip_llm,
        enable_summary="summary" in tasks if tasks else not args.skip_llm,
        enable_events="events" in tasks if tasks else not args.skip_llm,
        enable_language="language" in tasks if tasks else True,
        enable_relevancy="relevancy" in tasks if tasks else not args.skip_llm,
        enable_validation=not args.skip_validation,
        use_self_consistency=args.self_consistency
    )
    
    pipeline = UnifiedPipeline(api_key=args.api_key, options=options)
    return pipeline.run(text)


def run_cot_pipeline(text: str, args) -> dict:
    """Run the CoT v2 pipeline with function calling."""
    from src.cot.pipeline import CoTPipeline, PipelineConfig
    
    print("🔗 Running Chain-of-Thought Pipeline...", file=sys.stderr)
    
    semantic_clean = getattr(args, 'semantic_clean', False)
    enable_ner = getattr(args, 'ner', False)
    no_relationships = getattr(args, 'no_relationships', False)
    enable_events = getattr(args, 'events', False)
    enable_sentiment = getattr(args, 'sentiment', False)
    enable_summary = getattr(args, 'summary', False)
    summary_style = getattr(args, 'summary_style', 'bullets')
    enable_translation = getattr(args, 'translate', False)
    enable_relevancy = getattr(args, 'relevancy', False)
    
    config = PipelineConfig(
        enable_validation=not args.skip_validation,
        enable_domain_detection=not args.skip_domain and not args.skip_llm,
        enable_semantic_cleaning=semantic_clean and not args.skip_llm,
        enable_ner=enable_ner and not args.skip_llm,
        enable_relationships=not no_relationships,
        enable_events=enable_events and not args.skip_llm,
        enable_sentiment=enable_sentiment and not args.skip_llm,
        enable_summary=enable_summary and not args.skip_llm,
        summary_style=summary_style,
        enable_translation=enable_translation and not args.skip_llm,
        enable_relevancy=enable_relevancy and not args.skip_llm,
        use_self_consistency=args.self_consistency
    )
    
    if semantic_clean:
        print("🧠 Semantic cleaning enabled (LLM-powered)", file=sys.stderr)
    if enable_translation:
        print("🌐 Translation to English enabled", file=sys.stderr)
    if enable_ner:
        print("🏷️  NER enabled (ERA-CoT)", file=sys.stderr)
    if enable_events:
        print("📅 Event extraction enabled", file=sys.stderr)
    if enable_sentiment:
        print("💭 Sentiment analysis enabled", file=sys.stderr)
    if enable_summary:
        print(f"📝 Summarization enabled ({summary_style} style)", file=sys.stderr)
    if enable_relevancy:
        print("🎯 Relevancy analysis enabled", file=sys.stderr)
    
    pipeline = CoTPipeline(api_key=args.api_key, pipeline_config=config)
    
    if args.step:
        return pipeline.run_step(args.step, text)
    else:
        return pipeline.run(text)


def run_legacy_pipeline(text: str, args) -> dict:
    """Run the legacy v1.0 pipeline."""
    from src.pipeline import ChainOfThoughtPipeline
    
    print("🔗 Running Legacy Pipeline (v1.0)...", file=sys.stderr)
    
    pipeline = ChainOfThoughtPipeline(groq_api_key=args.api_key)
    return pipeline.process(
        text,
        skip_domain=args.skip_domain or args.skip_llm,
        skip_language=False
    )


def print_summary(result: dict, args):
    """Print a human-readable summary."""
    print("\n" + "="*50, file=sys.stderr)
    print("📊 Pipeline Summary", file=sys.stderr)
    print("="*50, file=sys.stderr)
    
    # Text cleaning stats
    clean_key = next((k for k in result.keys() if "text_cleaning" in k), None)
    if clean_key and result[clean_key].get("status") == "success":
        stats = result[clean_key].get("output", {})
        print(f"📝 Text Cleaning:", file=sys.stderr)
        print(f"   Original: {stats.get('original_length', stats.get('original_len', 'N/A'))} chars", file=sys.stderr)
        print(f"   Cleaned:  {stats.get('cleaned_length', stats.get('cleaned_len', 'N/A'))} chars", file=sys.stderr)
        reduction = stats.get('reduction_percent', stats.get('reduction', 'N/A'))
        if isinstance(reduction, float) and reduction < 1:
            reduction = f"{reduction*100:.1f}%"
        elif isinstance(reduction, (int, float)):
            reduction = f"{reduction}%"
        print(f"   Reduced:  {reduction}", file=sys.stderr)
    
    # Domain detection
    domain_key = next((k for k in result.keys() if "domain" in k and "detection" not in k.lower()), None)
    if domain_key:
        domain_result = result[domain_key]
        if domain_result.get("status") == "success":
            output = domain_result.get("output", {})
            conf = output.get('confidence', output.get('domain_conf', 0))
            print(f"🏷️  Domain: {output.get('domain', 'N/A')} ({conf:.0%} confidence)", file=sys.stderr)
    
    # Sentiment
    sent_key = next((k for k in result.keys() if "sentiment" in k), None)
    if sent_key:
        sent_result = result[sent_key]
        if sent_result.get("status") == "success":
            output = sent_result.get("output", {})
            print(f"😊 Sentiment: {output.get('sentiment', 'N/A')}", file=sys.stderr)
    
    # Language detection
    lang_key = next((k for k in result.keys() if "language" in k), None)
    if lang_key:
        lang_result = result[lang_key]
        if lang_result.get("status") == "success":
            output = lang_result.get("output", {})
            print(f"🌐 Language: {output.get('language_name', output.get('lang', 'N/A'))} "
                  f"({output.get('script_type', output.get('script', 'N/A'))})", file=sys.stderr)
    
    # NER
    ner_key = next((k for k in result.keys() if "ner" in k), None)
    if ner_key:
        ner_result = result[ner_key]
        if ner_result.get("status") == "success":
            output = ner_result.get("output", {})
            entities = output.get("entities", [])
            if entities:
                print(f"👤 Entities: {len(entities)} found", file=sys.stderr)
    
    # Summary
    sum_key = next((k for k in result.keys() if "summary" in k), None)
    if sum_key:
        sum_result = result[sum_key]
        if isinstance(sum_result, dict) and sum_result.get("status") == "success":
            output = sum_result.get("output", {})
            summary = output.get("summary", "") if isinstance(output, dict) else ""
            if summary:
                print(f"📄 Summary: {summary[:100]}{'...' if len(summary) > 100 else ''}", file=sys.stderr)
    
    # Validation
    val_key = next((k for k in result.keys() if "validation" in k), None)
    if val_key:
        val_result = result[val_key]
        is_valid = val_result.get("is_valid", val_result.get("output", {}).get("is_valid", True))
        quality = val_result.get("quality_score", val_result.get("output", {}).get("quality_score", 1.0))
        status_emoji = "✅" if is_valid else "⚠️"
        print(f"{status_emoji} Validation: {'Passed' if is_valid else 'Issues Found'} "
              f"(quality: {quality:.0%})", file=sys.stderr)
    
    # Metadata
    meta = result.get("metadata", {})
    print(f"\n⏱️  Total time: {meta.get('total_duration_ms', 'N/A')}ms", file=sys.stderr)
    print(f"📦 Format: {meta.get('output_format', 'json')}", file=sys.stderr)
    
    # Chain of thought summary
    cot_summary = result.get("chain_of_thought_summary", "")
    if cot_summary and not args.reasoning_only:
        print("\n💭 Chain of Thought:", file=sys.stderr)
        for line in cot_summary.split("\n")[:5]:  # Show first 5 lines
            print(f"   {line}", file=sys.stderr)
        if cot_summary.count("\n") > 5:
            print(f"   ... ({cot_summary.count(chr(10)) - 5} more)", file=sys.stderr)
    
    print("="*50, file=sys.stderr)


if __name__ == "__main__":
    main()
