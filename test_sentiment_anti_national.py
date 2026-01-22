#!/usr/bin/env python3
"""
Test script for anti-national sentiment score functionality.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.groq_client import GroqClient
from src.cot.executor import StepExecutor
from src.processors.sentiment_analyzer import SentimentAnalyzer

def test_sentiment_scores():
    """Test that sentiment analysis returns all four scores including anti_national."""
    
    print("=" * 60)
    print("Testing Sentiment Analysis with Anti-National Score")
    print("=" * 60)
    
    # Initialize analyzer
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in environment")
        return False
    
    client = GroqClient(api_key=api_key)
    executor = StepExecutor(client)
    analyzer = SentimentAnalyzer(executor=executor)
    
    # Test cases
    test_cases = [
        {
            "name": "Positive Text",
            "text": "I love my country and I'm proud to be an Indian. The government is doing great work!",
            "expected_high": "positive"
        },
        {
            "name": "Negative Text",
            "text": "This product is terrible. I'm very disappointed with the quality.",
            "expected_high": "negative"
        },
        {
            "name": "Neutral Text",
            "text": "The weather today is 25 degrees Celsius. It's a normal day.",
            "expected_high": "neutral"
        },
        {
            "name": "Anti-National Text (Test)",
            "text": "We should destroy the nation and overthrow the government. Death to the country!",
            "expected_high": "anti_national"
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n📝 Test: {test_case['name']}")
        print(f"   Text: {test_case['text'][:60]}...")
        
        try:
            result = analyzer.analyze(test_case['text'])
            
            # Check that all four scores are present
            scores = result.scores
            required_scores = ["positive", "negative", "neutral", "anti_national"]
            missing = [s for s in required_scores if s not in scores]
            
            if missing:
                print(f"   ❌ Missing scores: {missing}")
                all_passed = False
                continue
            
            # Check that scores sum to approximately 1.0
            total = sum(scores.values())
            if abs(total - 1.0) > 0.1:
                print(f"   ⚠️  Scores don't sum to 1.0 (sum: {total:.2f})")
            
            # Display scores
            print(f"   ✅ Scores:")
            for score_name in required_scores:
                score_value = scores.get(score_name, 0.0)
                marker = "⭐" if score_name == test_case['expected_high'] and score_value > 0.3 else "  "
                print(f"      {marker} {score_name:15s}: {score_value:.3f}")
            
            print(f"   Overall Sentiment: {result.overall_sentiment}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Is Concerning: {result.is_concerning}")
            
            # Verify expected high score
            expected_score = scores.get(test_case['expected_high'], 0.0)
            if expected_score < 0.2:
                print(f"   ⚠️  Expected {test_case['expected_high']} to be higher, but got {expected_score:.3f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = test_sentiment_scores()
    sys.exit(0 if success else 1)

