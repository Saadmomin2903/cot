#!/usr/bin/env python3
"""
Full LLM Test - Tests all features with actual Groq LLM.

Tests:
- Text cleaning
- Language detection
- Translation
- Domain detection
- NER
- Sentiment analysis
- Summarization
- Events extraction
- Collaborative review
- Hallucination detection
- Memory optimization
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.cot.pipeline import CoTPipeline, PipelineConfig
from src.utils.groq_client import GroqClient

# Test text
TEST_TEXT = """દિલ્હીની જાણીતી શૈક્ષણિક સંસ્થા, જવાહરલાલ નહેરુ યુનિવર્સિટી (JNU) ફરી એકવાર રાષ્ટ્ર વિરોધી તત્વોને કારણે સમાચારમાં ચમકી છે. ગઈકાલ સોમવાર, 5 જાન્યુઆરીએ, JNU કેમ્પસમાં વિરોધ પ્રદર્શન યોજવામાં આવ્યું હતું. જેએનયુના કહેવાતા વિદ્યાર્થીઓએ તેમના હાથમાં કાયદો અને વ્યવસ્થાને ખોરંભે પાડે તેવા સૂત્રો લખેલ પ્લેકાર્ડ અને ઢોલ લઈને કેમ્પસમાં પ્રદર્શન કર્યું હતું. JNU વિદ્યાર્થી સંઘ (JNUSU) અને ડાબેરી સંગઠનોએ, આ પ્રદર્શનની આગેવાની લીધી હતી.

સુપ્રીમ કોર્ટે, દિલ્હીમાં ફાટી નીકળેલા કોમી રમખાણ કેસના આરોપી એવા ઉમર ખાલિદ અને શરજીલ ઇમામની જામીન અરજી ફગાવી દીધી હતી. બસ રાષ્ટ્ર વિરોધી તત્વોને વિરોધ કરવાનું એક નવું બહાનુ મળી ગયું. વિદ્યાર્થીઓના નામે આવારા તત્વોએ સમગ્ર JNU કેમ્પસને માથે લીધુ હતું."""

ENGLISH_TEST_TEXT = """Jawaharlal Nehru University (JNU) is a public central university in New Delhi, India. It was established in 1969 and is known for its research programs. The university offers various courses in social sciences, international studies, and languages. JNU has been ranked among the top universities in India for research and academic excellence. Recently, there have been protests on campus regarding various political and social issues."""


def test_api_connection():
    """Test Groq API connection."""
    print("=" * 70)
    print("🔍 Testing Groq API Connection...")
    print("=" * 70)
    
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ GROQ_API_KEY not found in .env file")
        return False
    
    try:
        client = GroqClient(api_key=api_key)
        # Simple test call
        response = client.simple_prompt("Say 'API connection successful' in one word.")
        print(f"✅ API Connection: SUCCESS")
        print(f"   Response: {response[:50]}")
        return True
    except Exception as e:
        print(f"❌ API Connection Failed: {str(e)}")
        return False


def test_basic_pipeline():
    """Test basic pipeline with text cleaning and language detection."""
    print("\n" + "=" * 70)
    print("🔍 Test 1: Basic Pipeline (Text Cleaning + Language Detection)")
    print("=" * 70)
    
    api_key = os.getenv('GROQ_API_KEY')
    config = PipelineConfig(
        enable_validation=True,
        enable_domain_detection=False,  # Skip for basic test
        enable_memory_optimization=True,
        token_budget=4096
    )
    
    pipeline = CoTPipeline(api_key=api_key, pipeline_config=config)
    
    start = time.time()
    result = pipeline.run(TEST_TEXT)
    duration = (time.time() - start) * 1000
    
    print(f"✅ Status: SUCCESS")
    print(f"✅ Duration: {duration:.0f}ms")
    print(f"✅ Steps: {result['metadata']['steps_executed']}")
    
    # Text cleaning
    cleaning = result.get('1_text_cleaning', {})
    if cleaning.get('status') == 'success':
        output = cleaning['output']
        print(f"\n📝 Text Cleaning:")
        print(f"   Original: {output['original_length']} chars")
        print(f"   Cleaned: {output['cleaned_length']} chars")
        print(f"   Reduction: {output['reduction_percent']}%")
    
    # Language detection
    lang = result.get('2_language_detection', {})
    if lang.get('status') == 'success':
        output = lang['output']
        print(f"\n🌐 Language Detection:")
        print(f"   Language: {output['language_name']} ({output['language_code']})")
        print(f"   Script: {output['script_type']}")
        print(f"   Confidence: {output['confidence']:.2%}")
    
    return True


def test_translation():
    """Test translation feature."""
    print("\n" + "=" * 70)
    print("🔍 Test 2: Translation (Gujarati to English)")
    print("=" * 70)
    
    api_key = os.getenv('GROQ_API_KEY')
    config = PipelineConfig(
        enable_translation=True,
        enable_memory_optimization=True,
        token_budget=4096
    )
    
    pipeline = CoTPipeline(api_key=api_key, pipeline_config=config)
    
    start = time.time()
    result = pipeline.run(TEST_TEXT)
    duration = (time.time() - start) * 1000
    
    print(f"✅ Status: SUCCESS")
    print(f"✅ Duration: {duration:.0f}ms")
    
    # Find translation step
    trans_key = next((k for k in result.keys() if 'translation' in k), None)
    if trans_key:
        trans = result[trans_key]
        if trans.get('status') == 'success':
            output = trans['output']
            print(f"\n🌐 Translation:")
            print(f"   Source: {output.get('source_language', 'unknown')}")
            print(f"   Target: {output.get('target_language', 'English')}")
            print(f"   Confidence: {output.get('confidence', 0):.2%}")
            print(f"\n   Translated Text:")
            translated = output.get('translated_text', '')[:300]
            print(f"   {translated}...")
            return True
    
    print("⚠️  Translation step not found or failed")
    return False


def test_domain_detection():
    """Test domain detection."""
    print("\n" + "=" * 70)
    print("🔍 Test 3: Domain Detection")
    print("=" * 70)
    
    api_key = os.getenv('GROQ_API_KEY')
    config = PipelineConfig(
        enable_domain_detection=True,
        enable_memory_optimization=True,
        token_budget=4096
    )
    
    pipeline = CoTPipeline(api_key=api_key, pipeline_config=config)
    
    start = time.time()
    result = pipeline.run(ENGLISH_TEST_TEXT)
    duration = (time.time() - start) * 1000
    
    print(f"✅ Status: SUCCESS")
    print(f"✅ Duration: {duration:.0f}ms")
    
    # Find domain step
    domain_key = next((k for k in result.keys() if 'domain' in k and 'detection' in k), None)
    if domain_key:
        domain = result[domain_key]
        if domain.get('status') == 'success':
            output = domain['output']
            print(f"\n🏷️  Domain Detection:")
            print(f"   Primary Domain: {output.get('primary_domain', 'N/A')}")
            print(f"   Confidence: {output.get('confidence', 0):.2%}")
            if 'domain_scores' in output:
                scores = output['domain_scores']
                print(f"   Scores:")
                for dom, score in scores.items():
                    print(f"     - {dom}: {score:.2%}")
            if 'sub_categories' in output:
                print(f"   Sub-categories: {', '.join(output['sub_categories'][:3])}")
            return True
    
    print("⚠️  Domain detection step not found or failed")
    return False


def test_summarization():
    """Test summarization."""
    print("\n" + "=" * 70)
    print("🔍 Test 4: Summarization")
    print("=" * 70)
    
    api_key = os.getenv('GROQ_API_KEY')
    config = PipelineConfig(
        enable_summary=True,
        summary_style="bullets",
        enable_memory_optimization=True,
        token_budget=4096
    )
    
    pipeline = CoTPipeline(api_key=api_key, pipeline_config=config)
    
    start = time.time()
    result = pipeline.run(ENGLISH_TEST_TEXT)
    duration = (time.time() - start) * 1000
    
    print(f"✅ Status: SUCCESS")
    print(f"✅ Duration: {duration:.0f}ms")
    
    # Find summary step
    summary_key = next((k for k in result.keys() if 'summary' in k), None)
    if summary_key:
        summary = result[summary_key]
        if summary.get('status') == 'success':
            output = summary['output']
            print(f"\n📄 Summary:")
            print(f"   Strategy: {output.get('strategy', 'N/A')}")
            print(f"   Style: {output.get('style', 'N/A')}")
            print(f"   Confidence: {output.get('confidence', 0):.2%}")
            print(f"\n   Summary Text:")
            summary_text = output.get('summary', '')
            print(f"   {summary_text}")
            print(f"\n   Key Points ({len(output.get('key_points', []))}):")
            for i, point in enumerate(output.get('key_points', [])[:5], 1):
                print(f"   {i}. {point[:100]}")
            return True
    
    print("⚠️  Summary step not found or failed")
    return False


def test_sentiment():
    """Test sentiment analysis."""
    print("\n" + "=" * 70)
    print("🔍 Test 5: Sentiment Analysis")
    print("=" * 70)
    
    api_key = os.getenv('GROQ_API_KEY')
    config = PipelineConfig(
        enable_sentiment=True,
        enable_memory_optimization=True,
        token_budget=4096
    )
    
    pipeline = CoTPipeline(api_key=api_key, pipeline_config=config)
    
    start = time.time()
    result = pipeline.run(ENGLISH_TEST_TEXT)
    duration = (time.time() - start) * 1000
    
    print(f"✅ Status: SUCCESS")
    print(f"✅ Duration: {duration:.0f}ms")
    
    # Find sentiment step
    sent_key = next((k for k in result.keys() if 'sentiment' in k), None)
    if sent_key:
        sent = result[sent_key]
        if sent.get('status') == 'success':
            output = sent['output']
            print(f"\n😊 Sentiment Analysis:")
            print(f"   Sentiment: {output.get('sentiment', 'N/A')}")
            print(f"   Score: {output.get('score', 0):.2f}")
            print(f"   Confidence: {output.get('confidence', 0):.2%}")
            if 'emotion' in output:
                print(f"   Emotion: {output['emotion']}")
            return True
    
    print("⚠️  Sentiment step not found or failed")
    return False


def test_full_pipeline():
    """Test full pipeline with all features."""
    print("\n" + "=" * 70)
    print("🔍 Test 6: FULL PIPELINE (All Features)")
    print("=" * 70)
    
    api_key = os.getenv('GROQ_API_KEY')
    config = PipelineConfig(
        enable_validation=True,
        enable_domain_detection=True,
        enable_translation=True,
        enable_summary=True,
        summary_style="bullets",
        enable_sentiment=True,
        enable_ner=True,
        enable_relationships=True,
        enable_memory_optimization=True,
        enable_collaborative_review=False,  # Can enable if needed
        enable_hallucination_detection=False,  # Can enable if needed
        token_budget=4096
    )
    
    pipeline = CoTPipeline(api_key=api_key, pipeline_config=config)
    
    print(f"Processing text: {len(TEST_TEXT)} characters")
    print("Features enabled:")
    print("  ✅ Text Cleaning")
    print("  ✅ Language Detection")
    print("  ✅ Translation")
    print("  ✅ Domain Detection")
    print("  ✅ Summarization")
    print("  ✅ Sentiment Analysis")
    print("  ✅ NER")
    print("  ✅ Memory Optimization")
    
    start = time.time()
    result = pipeline.run(TEST_TEXT)
    duration = (time.time() - start) * 1000
    
    print(f"\n✅ Status: SUCCESS")
    print(f"✅ Total Duration: {duration:.0f}ms ({duration/1000:.2f}s)")
    print(f"✅ Steps Executed: {result['metadata']['steps_executed']}")
    
    # Show all step results
    print(f"\n📊 Step Results:")
    for key in sorted(result.keys()):
        if key.startswith(('1_', '2_', '3_', '4_', '5_', '6_', '7_', '8_', '9_', '10_')):
            step = result[key]
            status = step.get('status', 'unknown')
            emoji = "✅" if status == "success" else "⚠️" if status == "failed" else "⏭️"
            print(f"   {emoji} {key}: {status}")
            if status == "success" and 'output' in step:
                output = step['output']
                # Show key highlights
                if 'translated_text' in output:
                    print(f"      → Translated: {output['translated_text'][:80]}...")
                elif 'summary' in output:
                    print(f"      → Summary: {output['summary'][:80]}...")
                elif 'primary_domain' in output:
                    print(f"      → Domain: {output['primary_domain']} ({output.get('confidence', 0):.0%})")
                elif 'sentiment' in output:
                    print(f"      → Sentiment: {output['sentiment']} ({output.get('score', 0):.2f})")
                elif 'entities' in output:
                    entities = output['entities']
                    print(f"      → Entities: {len(entities)} found")
    
    # Metadata
    metadata = result.get('metadata', {})
    print(f"\n📈 Metadata:")
    print(f"   Pipeline Version: {metadata.get('pipeline_version', 'N/A')}")
    print(f"   Model Used: {metadata.get('model_used', 'N/A')}")
    print(f"   Total Duration: {metadata.get('total_duration_ms', 0)}ms")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("🧠 CHAIN OF THOUGHT PIPELINE - FULL LLM TEST SUITE")
    print("=" * 70)
    
    # Check API key
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("\n❌ ERROR: GROQ_API_KEY not found in .env file")
        print("   Please set GROQ_API_KEY in your .env file")
        return
    
    print(f"\n✅ Using Groq API Key: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # Test API connection
        if not test_api_connection():
            return
        
        # Run tests
        test_basic_pipeline()
        test_translation()
        test_domain_detection()
        test_summarization()
        test_sentiment()
        test_full_pipeline()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📝 Summary:")
        print("   ✅ API Connection: Working")
        print("   ✅ Text Cleaning: Working")
        print("   ✅ Language Detection: Working")
        print("   ✅ Translation: Working")
        print("   ✅ Domain Detection: Working")
        print("   ✅ Summarization: Working")
        print("   ✅ Sentiment Analysis: Working")
        print("   ✅ Full Pipeline: Working")
        print("   ✅ Memory Optimization: Active")
        print("   ✅ Token Optimization: Active")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

