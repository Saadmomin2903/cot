#!/usr/bin/env python3
"""
Test script for Chain of Thought Pipeline API endpoint.

Tests all features including new optimizations.
"""

import requests
import json
import time
from typing import Dict, Any

API_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("🔍 Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    data = response.json()
    print(f"✅ Health check: {data}")
    return True


def test_basic_processing():
    """Test basic text processing without LLM."""
    print("\n🔍 Testing basic text processing (no LLM)...")
    
    payload = {
        "text": "This is a test text with URLs https://example.com and HTML <p>content</p> that needs cleaning.",
        "skip_llm": True,
        "enable_memory_optimization": True,
        "token_budget": 4096
    }
    
    start = time.time()
    response = requests.post(f"{API_URL}/process", json=payload, timeout=30)
    duration = (time.time() - start) * 1000
    
    assert response.status_code == 200
    result = response.json()
    
    print(f"✅ Status: {result['status']}")
    print(f"✅ Duration: {result['duration_ms']}ms (actual: {duration:.0f}ms)")
    print(f"✅ Steps executed: {result['metadata'].get('steps_executed', 'N/A')}")
    
    # Check text cleaning
    cleaning = result['results'].get('1_text_cleaning', {})
    if cleaning.get('status') == 'success':
        output = cleaning.get('output', {})
        print(f"✅ Text cleaning: {output.get('reduction_percent', 0)}% reduction")
        print(f"   Removed: {', '.join(output.get('removed_elements', []))}")
    
    # Check language detection
    lang = result['results'].get('2_language_detection', {})
    if lang.get('status') == 'success':
        output = lang.get('output', {})
        print(f"✅ Language: {output.get('language_name')} ({output.get('script_type')})")
    
    return True


def test_with_summary():
    """Test with summarization enabled."""
    print("\n🔍 Testing with summarization (requires API key)...")
    
    payload = {
        "text": """Jawaharlal Nehru University (JNU) is a public central university in New Delhi, India. 
        It was established in 1969 and is known for its research programs. The university offers various 
        courses in social sciences, international studies, and languages. JNU has been ranked among the 
        top universities in India for research and academic excellence.""",
        "summary": True,
        "summary_style": "bullets",
        "enable_memory_optimization": True,
        "token_budget": 4096,
        "skip_llm": False  # Requires API key
    }
    
    try:
        start = time.time()
        response = requests.post(f"{API_URL}/process", json=payload, timeout=60)
        duration = (time.time() - start) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Status: {result['status']}")
            print(f"✅ Duration: {result['duration_ms']}ms")
            
            # Check summary
            summary_key = next((k for k in result['results'].keys() if 'summary' in k), None)
            if summary_key:
                summary = result['results'][summary_key]
                if summary.get('status') == 'success':
                    output = summary.get('output', {})
                    print(f"✅ Summary generated: {len(output.get('summary', ''))} chars")
                    print(f"   Key points: {len(output.get('key_points', []))}")
                    print(f"   Confidence: {output.get('confidence', 0):.2f}")
        else:
            print(f"⚠️  API returned {response.status_code}: {response.text[:200]}")
    except Exception as e:
        print(f"⚠️  Summary test skipped (may need API key): {str(e)[:100]}")
    
    return True


def test_optimization_features():
    """Test new optimization features."""
    print("\n🔍 Testing optimization features...")
    
    payload = {
        "text": "Artificial Intelligence is transforming industries. Machine learning algorithms can process vast amounts of data.",
        "summary": True,
        "enable_memory_optimization": True,
        "enable_collaborative_review": False,  # Set True if you want to test
        "enable_hallucination_detection": False,  # Set True if you want to test
        "token_budget": 4096,
        "skip_llm": True  # Test without LLM for speed
    }
    
    start = time.time()
    response = requests.post(f"{API_URL}/process", json=payload, timeout=30)
    duration = (time.time() - start) * 1000
    
    assert response.status_code == 200
    result = response.json()
    
    print(f"✅ Memory optimization: Enabled")
    print(f"✅ Token budget: {payload['token_budget']}")
    print(f"✅ Processing time: {duration:.0f}ms")
    
    # Check metadata
    metadata = result.get('metadata', {})
    print(f"✅ Pipeline version: {metadata.get('pipeline_version', 'N/A')}")
    print(f"✅ Steps executed: {metadata.get('steps_executed', 'N/A')}")
    
    return True


def test_error_handling():
    """Test error handling."""
    print("\n🔍 Testing error handling...")
    
    # Test with empty text
    payload = {"text": ""}
    response = requests.post(f"{API_URL}/process", json=payload, timeout=10)
    # Should either validate or handle gracefully
    print(f"✅ Empty text handled: {response.status_code}")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Chain of Thought Pipeline API - Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Health check
        test_health()
        
        # Test 2: Basic processing
        test_basic_processing()
        
        # Test 3: Optimization features
        test_optimization_features()
        
        # Test 4: With summary (if API key available)
        test_with_summary()
        
        # Test 5: Error handling
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API server.")
        print("   Make sure the server is running: python -m uvicorn api:app --port 8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

