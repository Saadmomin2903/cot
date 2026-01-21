"""
Event Calendar Extraction Module

Extracts temporal events from text including:
- Past events (what happened and when)
- Future events (what will happen and when)
- Recurring events (schedules, patterns)
- Event relationships (sequences, dependencies)

Uses ERA-CoT approach for accurate extraction with:
- Date/time normalization
- Event categorization
- Temporal reasoning
"""

import re
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..cot import PipelineStep, PipelineContext, StepResult, StepStatus, FunctionDefinition
from ..cot.executor import StepExecutor


class EventType(Enum):
    """Types of temporal events."""
    ANNOUNCEMENT = "announcement"
    MEETING = "meeting"
    LAUNCH = "launch"
    DEADLINE = "deadline"
    CONFERENCE = "conference"
    HOLIDAY = "holiday"
    MILESTONE = "milestone"
    GENERAL = "general"


class TemporalRelation(Enum):
    """Temporal relations between events."""
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    SIMULTANEOUS = "simultaneous"


@dataclass
class TemporalEvent:
    """Represents an extracted temporal event."""
    description: str
    date_text: str  # Original date text from source
    normalized_date: str = None  # ISO format if parseable
    event_type: str = "general"
    is_future: bool = False
    is_recurring: bool = False
    confidence: float = 0.85
    entities_involved: List[str] = field(default_factory=list)
    location: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "date": self.date_text,
            "normalized_date": self.normalized_date,
            "type": self.event_type,
            "is_future": self.is_future,
            "is_recurring": self.is_recurring,
            "confidence": self.confidence,
            "entities": self.entities_involved if self.entities_involved else None,
            "location": self.location
        }


# Function definition for CoT executor
EVENT_FUNCTION = FunctionDefinition(
    name="extract_events",
    description="Extract temporal events with dates from text",
    parameters={
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "date": {"type": "string"},
                    "type": {"type": "string", "enum": ["announcement", "meeting", "launch", "deadline", "conference", "holiday", "milestone", "general"]},
                    "is_future": {"type": "boolean"}
                }
            },
            "description": "List of extracted events"
        },
        "reasoning": {
            "type": "string",
            "description": "Step-by-step reasoning for the extraction"
        }
    },
    required=["events", "reasoning"]
)


class EventCalendarExtractor:
    """
    Event Calendar Extraction System.
    
    Extracts temporal events from text using:
    - LLM for complex event understanding
    - Regex patterns for date extraction
    - Temporal normalization
    
    Features:
    - Past/future event classification
    - Recurring event detection
    - Entity linking to events
    - Calendar format output
    
    Usage:
        extractor = EventCalendarExtractor(executor)
        result = extractor.extract("Apple will announce iPhone 16 on September 10, 2025")
        print(result["events"])
    """
    
    SYSTEM_PROMPT = """You are an expert at extracting temporal events from text.

## Your Task
Extract ALL events with their dates/times from the input text.

## Event Types
- announcement: Product launches, press releases, company announcements
- meeting: Scheduled meetings, conferences, gatherings
- launch: Product/service launches, release dates
- deadline: Due dates, submission deadlines
- conference: Industry events, summits, conventions
- holiday: National holidays, celebrations
- milestone: Achievements, anniversaries, records
- general: Other dated events

## What to Extract
For each event, identify:
1. WHAT happened/will happen (description)
2. WHEN (date, time, or relative time reference)
3. WHO is involved (people, organizations)
4. WHERE (location if mentioned)
5. Is it FUTURE or PAST relative to today
6. Is it RECURRING (weekly, monthly, annual)

## Output Format
List each event as:
EVENT: [description]
DATE: [date/time text]
TYPE: [event_type]
FUTURE: [yes/no]

## Important Rules
- Include relative dates (next week, yesterday, Q4 2025)
- Mark recurring events (every Monday, annually)
- Link entities to events when clear
- Normalize dates when possible (January 15 → 2025-01-15)"""

    # Common date patterns
    DATE_PATTERNS = [
        # Full dates
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
        r'\d{1,2}/\d{1,2}/\d{4}',
        r'\d{4}-\d{2}-\d{2}',
        # Relative dates
        r'(?:next|last|this)\s+(?:week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',
        r'(?:yesterday|today|tomorrow)',
        r'(?:in|within)\s+\d+\s+(?:days?|weeks?|months?|years?)',
        # Quarter references
        r'Q[1-4]\s+\d{4}',
        # Time references
        r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?',
    ]

    def __init__(
        self,
        executor: StepExecutor = None,
        reference_date: datetime = None
    ):
        """
        Initialize Event Calendar Extractor.
        
        Args:
            executor: StepExecutor for LLM calls
            reference_date: Reference date for relative calculations (default: now)
        """
        self.executor = executor
        self.reference_date = reference_date or datetime.now()
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract events from text.
        
        Args:
            text: Input text
            
        Returns:
            Dict with events, calendar, and timeline
        """
        start_time = time.time()
        
        if not self.executor:
            return self._fallback_extraction(text)
        
        # Extract events using LLM
        events = self._extract_events_llm(text)
        
        # Organize into calendar format
        calendar = self._organize_calendar(events)
        
        # Create timeline
        timeline = self._create_timeline(events)
        
        return {
            "events": [e.to_dict() for e in events],
            "event_count": len(events),
            "calendar": calendar,
            "timeline": timeline,
            "future_events": len([e for e in events if e.is_future]),
            "past_events": len([e for e in events if not e.is_future]),
            "reasoning": self._generate_reasoning(events),
            "duration_ms": int((time.time() - start_time) * 1000)
        }
    
    def _extract_events_llm(self, text: str) -> List[TemporalEvent]:
        """Extract events using LLM."""
        today = self.reference_date.strftime("%Y-%m-%d")
        
        user_prompt = f"""Extract all temporal events from the following text.
Today's date is: {today}

## Text
{text[:4000]}

## Output Format
For each event found, output:
EVENT: [what happened/will happen]
DATE: [when - exact or relative]
TYPE: [announcement/meeting/launch/deadline/conference/holiday/milestone/general]
FUTURE: [yes/no - relative to today]
ENTITIES: [who is involved, comma separated]
LOCATION: [where, if mentioned]

List all events:"""

        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.executor.client.chat(
                messages=messages,
                temperature=0.2
            )
            
            return self._parse_events(response)
            
        except Exception as e:
            return self._fallback_extraction(text).get("events", [])
    
    def _parse_events(self, response: str) -> List[TemporalEvent]:
        """Parse events from LLM response."""
        events = []
        
        # Split by EVENT: markers
        event_blocks = re.split(r'\n(?=EVENT:)', response)
        
        for block in event_blocks:
            if not block.strip() or 'EVENT:' not in block:
                continue
            
            # Extract fields
            description = self._extract_field(block, 'EVENT')
            date_text = self._extract_field(block, 'DATE')
            event_type = self._extract_field(block, 'TYPE') or 'general'
            is_future = self._extract_field(block, 'FUTURE', '').lower() in ['yes', 'true', '1']
            entities_str = self._extract_field(block, 'ENTITIES')
            location = self._extract_field(block, 'LOCATION')
            
            if description and date_text:
                entities = [e.strip() for e in entities_str.split(',')] if entities_str else []
                
                events.append(TemporalEvent(
                    description=description,
                    date_text=date_text,
                    normalized_date=self._normalize_date(date_text),
                    event_type=event_type.lower(),
                    is_future=is_future,
                    entities_involved=entities,
                    location=location if location and location.lower() != 'none' else None
                ))
        
        return events
    
    def _extract_field(self, block: str, field_name: str, default: str = '') -> str:
        """Extract a field value from event block."""
        pattern = rf'{field_name}:\s*(.+?)(?:\n|$)'
        match = re.search(pattern, block, re.IGNORECASE)
        return match.group(1).strip() if match else default
    
    def _normalize_date(self, date_text: str) -> Optional[str]:
        """Attempt to normalize date to ISO format."""
        # Common month names
        months = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        
        # Try Month DD, YYYY format
        match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})', date_text, re.IGNORECASE)
        if match:
            month = months[match.group(1).lower()]
            day = match.group(2).zfill(2)
            year = match.group(3)
            return f"{year}-{month}-{day}"
        
        # Try YYYY-MM-DD format
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_text)
        if match:
            return match.group(0)
        
        # Try MM/DD/YYYY format
        match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_text)
        if match:
            month = match.group(1).zfill(2)
            day = match.group(2).zfill(2)
            year = match.group(3)
            return f"{year}-{month}-{day}"
        
        return None
    
    def _organize_calendar(self, events: List[TemporalEvent]) -> Dict[str, List[Dict]]:
        """Organize events into calendar structure."""
        calendar = {}
        
        for event in events:
            date_key = event.normalized_date or event.date_text
            if date_key not in calendar:
                calendar[date_key] = []
            calendar[date_key].append({
                "description": event.description,
                "type": event.event_type,
                "time": None  # Could extract time if present
            })
        
        return calendar
    
    def _create_timeline(self, events: List[TemporalEvent]) -> List[Dict]:
        """Create chronological timeline."""
        # Sort by normalized date if available
        def sort_key(e):
            return e.normalized_date or e.date_text
        
        sorted_events = sorted([e for e in events if e.normalized_date], key=sort_key)
        
        return [
            {"date": e.normalized_date, "event": e.description, "type": e.event_type}
            for e in sorted_events
        ]
    
    def _generate_reasoning(self, events: List[TemporalEvent]) -> str:
        """Generate reasoning summary."""
        if not events:
            return "No temporal events found in text."
        
        future = len([e for e in events if e.is_future])
        past = len([e for e in events if not e.is_future])
        
        types = {}
        for e in events:
            types[e.event_type] = types.get(e.event_type, 0) + 1
        
        parts = [f"Extracted {len(events)} events:"]
        parts.append(f"- {past} past events, {future} future events")
        parts.append(f"- Types: {', '.join(f'{t}({c})' for t, c in types.items())}")
        
        return "\n".join(parts)
    
    def _fallback_extraction(self, text: str) -> Dict[str, Any]:
        """Fallback extraction using regex patterns."""
        events = []
        
        # Extract dates using patterns
        for pattern in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                date_text = match.group()
                # Get surrounding context (50 chars before and after)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                events.append(TemporalEvent(
                    description=context.strip(),
                    date_text=date_text,
                    normalized_date=self._normalize_date(date_text),
                    event_type="general"
                ))
        
        return {
            "events": [e.to_dict() for e in events],
            "event_count": len(events),
            "calendar": {},
            "timeline": [],
            "reasoning": "Fallback regex extraction (no LLM available)",
            "duration_ms": 0
        }


class EventCalendarStep(PipelineStep):
    """
    Event Calendar Pipeline Step.
    
    Integrates with the CoT pipeline for temporal
    event extraction and calendar generation.
    """
    
    def __init__(self, executor: StepExecutor = None):
        super().__init__(
            name="event_calendar",
            description="Extract temporal events and create calendar"
        )
        self.extractor = EventCalendarExtractor(executor=executor)
    
    def get_function_definition(self) -> FunctionDefinition:
        return EVENT_FUNCTION
    
    def get_cot_prompt(self, context: PipelineContext) -> str:
        return self.extractor.SYSTEM_PROMPT
    
    def execute(self, context: PipelineContext) -> StepResult:
        """Execute event extraction."""
        start_time = time.time()
        
        text = context.current_text or context.original_input
        result = self.extractor.extract(text)
        
        return StepResult(
            step_name=self.name,
            status=StepStatus.SUCCESS,
            output={
                "events": result["events"],
                "event_count": result["event_count"],
                "calendar": result.get("calendar", {}),
                "timeline": result.get("timeline", []),
                "future_events": result.get("future_events", 0),
                "past_events": result.get("past_events", 0)
            },
            reasoning=result["reasoning"],
            confidence=0.9 if result["event_count"] > 0 else 0.5,
            duration_ms=result.get("duration_ms", int((time.time() - start_time) * 1000))
        )
