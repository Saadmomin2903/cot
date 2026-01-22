# Anti-National Score Implementation

## Overview
Added anti-national sentiment score as a separate category alongside positive, negative, and neutral scores in the sentiment analysis system.

## Changes Made

### 1. Sentiment Function Definition (`src/processors/sentiment_analyzer.py`)
- Updated `SENTIMENT_FUNCTION` to include `scores` as a required parameter
- Added `anti_national` as a property in the scores object
- All four scores (positive, negative, neutral, anti_national) are now required in the structured output

### 2. System Prompt Updates
- Enhanced the system prompt to emphasize anti-national content detection
- Added detailed criteria for identifying anti-national content:
  - Hatred toward a nation or its people
  - Incitement of violence against national institutions
  - Promotion of terrorism, separatism, or sedition
  - Blasphemy or hate speech against national symbols
  - Deliberate misinformation against national interest
  - Calls to destroy or overthrow constitutional institutions
  - Anti-national slogans or chants

### 3. Score Calculation (`_calculate_scores` method)
- Updated to always return all four scores: `positive`, `negative`, `neutral`, `anti_national`
- When anti-national sentiment is detected:
  - `anti_national` score = confidence
  - `negative` score = (1 - confidence) * 0.6
  - `neutral` score = (1 - confidence) * 0.3
  - `positive` score = (1 - confidence) * 0.1
- Scores are normalized to sum to approximately 1.0

### 4. LLM Integration
- Updated `_analyze_with_llm` to use `execute_with_function` for structured output
- Falls back to text parsing if structured output is not available
- Ensures all four scores are present in the result

### 5. Frontend Display (`app.py`)
- Added sentiment scores visualization section
- Displays all four scores as:
  - Bar chart showing score distribution
  - Four metric cards (Positive, Negative, Neutral, Anti-National)
  - Warning indicator (⚠️) when anti-national score > 0.3
- Maintains backward compatibility with existing sentiment aspects display

## Test Results

All tests passed successfully:

✅ **Positive Text**: Correctly identifies positive sentiment (0.800 positive score)
✅ **Negative Text**: Correctly identifies negative sentiment (0.900 negative score)
✅ **Neutral Text**: Correctly identifies neutral sentiment (0.900 neutral score)
✅ **Anti-National Text**: Correctly detects anti-national content (0.526 anti_national score, marked as concerning)

## Usage

The sentiment analysis now returns scores in the following format:

```python
{
    "sentiment": "anti_national",  # or positive, negative, neutral, mixed
    "confidence": 1.0,
    "scores": {
        "positive": 0.0,
        "negative": 0.474,
        "neutral": 0.0,
        "anti_national": 0.526
    },
    "is_concerning": True,  # True when anti_national > 0.3
    "emotion": "anger",
    "reasoning": "..."
}
```

## API Response

The API endpoint `/process` now returns sentiment scores including anti-national:

```json
{
    "results": {
        "sentiment": {
            "status": "success",
            "output": {
                "sentiment": "anti_national",
                "scores": {
                    "positive": 0.0,
                    "negative": 0.474,
                    "neutral": 0.0,
                    "anti_national": 0.526
                },
                "is_concerning": true,
                ...
            }
        }
    }
}
```

## Frontend Display

The Streamlit frontend now shows:
1. **Sentiment Scores Bar Chart**: Visual representation of all four scores
2. **Score Metrics**: Four metric cards showing individual scores
3. **Warning Indicator**: ⚠️ appears when anti-national score exceeds 0.3
4. **Sentiment Aspects**: Existing aspect-based sentiment analysis (unchanged)

## Backward Compatibility

- Existing code that accesses `scores["positive"]`, `scores["negative"]`, `scores["neutral"]` continues to work
- New `scores["anti_national"]` field is always present (defaults to 0.0 if not detected)
- `is_concerning` flag is automatically set based on anti-national score threshold (> 0.3)

## Files Modified

1. `src/processors/sentiment_analyzer.py` - Core sentiment analysis logic
2. `app.py` - Frontend display updates

## Files Created

1. `test_sentiment_anti_national.py` - Test script for verification
2. `ANTI_NATIONAL_SCORE_IMPLEMENTATION.md` - This documentation

