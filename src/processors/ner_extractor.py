"""
Advanced Named Entity Recognition (NER) with ERA-CoT

Implements the ERA-CoT (Entity Relationship Analysis with Chain-of-Thought) approach:
1. Entity Extraction (EE) - Extract all entity spans with self-consistency
2. Explicit Relationship Extraction (ERE) - Extract directly stated relations
3. Implicit Relationship Inference (ERI) - Infer multi-step implicit relations
4. Relationship Discrimination - Score and filter inferred relations
5. Final NER Output - Structured entity output with relationships

Based on research:
- ERA-CoT paper methodology
- Few-shot prompting with in-context learning
- Self-consistency voting for robust extraction
- SERAX format for reliable structured output

Entity Types Supported:
- PERSON: People names
- ORG: Organizations, companies, institutions
- LOC: Locations, places, addresses
- DATE: Dates, times, periods
- EVENT: Events, occasions
- PRODUCT: Products, services
- TECH: Technologies, tools, frameworks
- MONEY: Monetary values
- PERCENT: Percentages
- MISC: Other entities
"""

import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from enum import Enum

from ..cot import PipelineStep, PipelineContext, StepResult, StepStatus, FunctionDefinition
from ..cot.executor import StepExecutor
from ..serax import SeraxSchema, FieldDefinition, FieldType, SeraxParser, SeraxDelimiters


class EntityType(Enum):
    """Supported entity types."""
    PERSON = "PERSON"
    ORG = "ORG"
    LOC = "LOC"
    DATE = "DATE"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    TECH = "TECH"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    MISC = "MISC"


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    entity_type: str
    start_pos: int = -1
    end_pos: int = -1
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.entity_type,
            "confidence": self.confidence,
            "span": [self.start_pos, self.end_pos] if self.start_pos >= 0 else None,
            "metadata": self.metadata if self.metadata else None
        }


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source: str
    target: str
    relation_type: str
    confidence: float = 1.0
    is_explicit: bool = True
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation_type,
            "confidence": self.confidence,
            "explicit": self.is_explicit,
            "reasoning": self.reasoning if self.reasoning else None
        }


# SERAX Schema for NER output
NER_SERAX_SCHEMA = SeraxSchema(
    name="ner_extraction",
    description="Named Entity Recognition with ERA-CoT",
    fields=[
        FieldDefinition("entities", FieldType.LIST, description="Extracted entities as [TYPE:text, ...]"),
        FieldDefinition("persons", FieldType.LIST, required=False, description="Person names"),
        FieldDefinition("orgs", FieldType.LIST, required=False, description="Organizations"),
        FieldDefinition("locations", FieldType.LIST, required=False, description="Locations"),
        FieldDefinition("dates", FieldType.LIST, required=False, description="Dates/times"),
        FieldDefinition("techs", FieldType.LIST, required=False, description="Technologies"),
        FieldDefinition("products", FieldType.LIST, required=False, description="Products"),
        FieldDefinition("relations", FieldType.LIST, required=False, description="Entity relationships"),
        FieldDefinition("reasoning", FieldType.STRING, description="CoT reasoning for extraction"),
    ]
)

# Function definition for CoT executor
NER_FUNCTION = FunctionDefinition(
    name="extract_entities",
    description="Extract named entities with relationships",
    parameters={
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string", "enum": ["PERSON", "ORG", "LOC", "DATE", "EVENT", "PRODUCT", "TECH", "MONEY", "PERCENT", "MISC"]},
                    "confidence": {"type": "number"}
                }
            },
            "description": "List of extracted entities"
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {"type": "string"},
                    "explicit": {"type": "boolean"}
                }
            },
            "description": "Relationships between entities"
        },
        "reasoning": {
            "type": "string",
            "description": "Step-by-step reasoning for the extraction"
        }
    },
    required=["entities", "reasoning"]
)


class ERACotNER:
    """
    ERA-CoT Named Entity Recognition System.
    
    Implements the 5-step ERA-CoT pipeline:
    1. Entity Extraction with self-consistency
    2. Explicit Relationship Extraction
    3. Implicit Relationship Inference
    4. Relationship Discrimination (scoring)
    5. Final structured output
    
    Features:
    - Self-consistency voting (k=3 by default)
    - Relationship scoring with threshold filtering (v_th=6)
    - Chain-of-thought reasoning at each step
    - SERAX format for reliable output
    
    Usage:
        ner = ERACotNER(executor)
        result = ner.extract("Apple CEO Tim Cook announced...")
        print(result["entities"])
        print(result["relationships"])
    """
    
    # Entity extraction system prompt
    ENTITY_SYSTEM_PROMPT = """You are an expert Named Entity Recognition (NER) system using Chain-of-Thought reasoning.

## Your Task
Extract ALL named entities from the input text with their types.

## Entity Types
- PERSON: Names of people (e.g., Tim Cook, Elon Musk)
- ORG: Organizations, companies, institutions (e.g., Apple Inc., MIT, WHO)
- LOC: Locations, places, countries, cities (e.g., California, Tokyo, Mount Everest)
- DATE: Dates, times, periods (e.g., March 15, 2024, next week, Q4 2023)
- EVENT: Named events (e.g., World Cup, CES 2024)
- PRODUCT: Products, services (e.g., iPhone 15, ChatGPT)
- TECH: Technologies, frameworks, tools (e.g., TensorFlow, Python, AWS)
- MONEY: Monetary values (e.g., $5 million, €100)
- PERCENT: Percentages (e.g., 95%, 0.5%)
- MISC: Other named entities

## Chain-of-Thought Process
1. Read the text carefully
2. Identify potential entity mentions
3. Classify each entity by type
4. Verify entity boundaries (exact text span)
5. Check for nested or overlapping entities
6. Assign confidence scores

## Important Rules
- Extract exact text spans (not paraphrased)
- Don't miss entities - be thorough
- When in doubt about type, use MISC
- Include all dates, even relative ones
- Technical terms (frameworks, languages) are TECH"""

    RELATIONSHIP_SYSTEM_PROMPT = """You are an expert at extracting relationships between entities.

## Your Task
Given a text and extracted entities, identify relationships between them.

## Relationship Types
- EMPLOYED_BY: Person works for Organization
- CEO_OF / LEADS: Person leads Organization
- LOCATED_IN: Entity is located in Location
- FOUNDED_BY: Organization founded by Person
- CREATED_BY: Product/Tech created by Person/Org
- PART_OF: Entity is part of another
- USES: Entity uses another entity
- ANNOUNCED: Person/Org announced something
- OCCURRED_ON: Event occurred on Date
- VALUED_AT: Entity valued at Money

## Instructions
1. For each entity pair, determine if there's a relationship
2. Extract EXPLICIT relationships (directly stated)
3. Infer IMPLICIT relationships (implied but not stated)
4. Score each relationship 0-10 for confidence
5. Only include relationships with score >= 6"""

    def __init__(
        self,
        executor: StepExecutor = None,
        use_self_consistency: bool = True,
        num_consistency_runs: int = 3,
        relation_threshold: float = 6.0,
        max_implicit_per_pair: int = 3,
        use_serax: bool = True
    ):
        """
        Initialize ERA-CoT NER system.
        
        Args:
            executor: StepExecutor for LLM calls
            use_self_consistency: Use SC for entity extraction (recommended)
            num_consistency_runs: Number of runs for self-consistency (default: 3)
            relation_threshold: Min score for keeping inferred relations (default: 6)
            max_implicit_per_pair: Max implicit relations per entity pair (default: 3)
            use_serax: Use SERAX format for output
        """
        self.executor = executor
        self.use_sc = use_self_consistency
        self.sc_runs = num_consistency_runs
        self.v_th = relation_threshold
        self.k = max_implicit_per_pair
        self.use_serax = use_serax
        self.parser = SeraxParser(NER_SERAX_SCHEMA) if use_serax else None
    
    def extract(
        self,
        text: str,
        extract_relationships: bool = True,
        infer_implicit: bool = True
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships using ERA-CoT.
        
        Args:
            text: Input text
            extract_relationships: Whether to extract relationships
            infer_implicit: Whether to infer implicit relationships
            
        Returns:
            Dict with entities, relationships, and reasoning
        """
        start_time = time.time()
        
        if not self.executor:
            return self._fallback_extraction(text)
        
        # Step 1: Entity Extraction (with optional self-consistency)
        if self.use_sc:
            entities = self._extract_entities_with_sc(text)
        else:
            entities = self._extract_entities(text)
        
        if not entities:
            return {
                "entities": [],
                "relationships": [],
                "by_type": {},
                "reasoning": "No entities found in text",
                "duration_ms": int((time.time() - start_time) * 1000)
            }
        
        # Step 2 & 3: Relationship Extraction (explicit + implicit)
        relationships = []
        if extract_relationships and len(entities) > 1:
            explicit_rels = self._extract_explicit_relationships(text, entities)
            relationships.extend(explicit_rels)
            
            if infer_implicit:
                implicit_rels = self._infer_implicit_relationships(text, entities, explicit_rels)
                # Step 4: Relationship Discrimination
                filtered_implicit = self._filter_relationships(implicit_rels)
                relationships.extend(filtered_implicit)
        
        # Organize by type
        by_type = self._organize_by_type(entities)
        
        return {
            "entities": [e.to_dict() for e in entities],
            "relationships": [r.to_dict() for r in relationships],
            "by_type": by_type,
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "reasoning": self._generate_reasoning(entities, relationships),
            "duration_ms": int((time.time() - start_time) * 1000)
        }
    
    def _extract_entities_with_sc(self, text: str) -> List[Entity]:
        """Extract entities using self-consistency voting."""
        all_extractions = []
        
        for i in range(self.sc_runs):
            # Vary temperature slightly for diversity
            temp = 0.3 + (i * 0.1)
            entities = self._extract_entities(text, temperature=temp)
            all_extractions.append(entities)
        
        # Vote on entities
        return self._vote_entities(all_extractions)
    
    def _vote_entities(self, all_extractions: List[List[Entity]]) -> List[Entity]:
        """Apply majority voting to entity extractions."""
        # Count (text, type) pairs
        entity_counts = Counter()
        entity_examples = {}
        
        for extraction in all_extractions:
            for entity in extraction:
                key = (entity.text.lower(), entity.entity_type)
                entity_counts[key] += 1
                if key not in entity_examples:
                    entity_examples[key] = entity
        
        # Keep entities that appear in majority of runs
        threshold = self.sc_runs // 2 + 1
        voted_entities = []
        
        for key, count in entity_counts.items():
            if count >= threshold:
                entity = entity_examples[key]
                # Adjust confidence based on voting
                entity.confidence = count / self.sc_runs
                voted_entities.append(entity)
        
        return voted_entities
    
    def _extract_entities(self, text: str, temperature: float = 0.3) -> List[Entity]:
        """Single extraction run."""
        user_prompt = f"""Extract all named entities from the following text.

## Text
{text[:4000]}

## Chain-of-Thought
Think step by step:
1. What people are mentioned?
2. What organizations/companies?
3. What locations?
4. What dates/times?
5. What technologies/products?
6. Any other named entities?

## Output Format
List each entity on a new line as:
TYPE: entity_text (confidence%)

Example:
PERSON: Tim Cook (95%)
ORG: Apple Inc. (90%)
LOC: California (85%)
DATE: March 15, 2024 (95%)
TECH: TensorFlow (90%)

Now extract all entities:"""

        try:
            messages = [
                {"role": "system", "content": self.ENTITY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.executor.client.chat(
                messages=messages,
                temperature=temperature
            )
            
            return self._parse_entities(response)
            
        except Exception as e:
            return []
    
    def _parse_entities(self, response: str) -> List[Entity]:
        """Parse entities from LLM response."""
        entities = []
        seen_texts = set()  # Track seen entity texts to avoid duplicates
        
        # Pattern: TYPE: entity_text (confidence%)
        # Examples: PERSON: Tim Cook (95%), ORG: Apple Inc. (90%)
        pattern = r'(PERSON|ORG|LOC|DATE|EVENT|PRODUCT|TECH|MONEY|PERCENT|MISC)\s*[:：]\s*(.+?)(?:\s*\((\d+(?:\.\d+)?)\s*%?\s*\))?(?:\n|$)'
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        
        for entity_type, text, conf_str in matches:
            text = text.strip().strip('"\'')
            # Clean up common artifacts
            text = re.sub(r'^\d+\.\s*', '', text)  # Remove numbering
            text = re.sub(r'^[-•]\s*', '', text)   # Remove bullet points
            text = text.strip()
            
            # Parse confidence
            conf = float(conf_str) / 100 if conf_str else 0.85
            
            if text and len(text) > 1 and text.lower() not in seen_texts:
                seen_texts.add(text.lower())
                entities.append(Entity(
                    text=text,
                    entity_type=entity_type.upper(),
                    confidence=conf
                ))
        
        # Also try to parse comma-separated lists for each type
        for etype in ['PERSON', 'ORG', 'LOC', 'DATE', 'TECH', 'PRODUCT', 'EVENT', 'MONEY', 'PERCENT', 'MISC']:
            pattern2 = rf'{etype}S?\s*[:：]\s*([^\n]+)'
            match = re.search(pattern2, response, re.IGNORECASE)
            if match:
                items_text = match.group(1)
                # Split by comma, handling confidence scores
                items = re.split(r',\s*(?=[A-Z])', items_text)
                for item in items:
                    item = item.strip()
                    # Extract confidence if present
                    conf_match = re.search(r'\((\d+(?:\.\d+)?)\s*%?\)', item)
                    conf = float(conf_match.group(1)) / 100 if conf_match else 0.85
                    # Clean item
                    item = re.sub(r'\s*\(\d+(?:\.\d+)?%?\)', '', item).strip().strip('"\'')
                    if item and len(item) > 1 and item.lower() not in seen_texts:
                        seen_texts.add(item.lower())
                        entities.append(Entity(
                            text=item,
                            entity_type=etype,
                            confidence=conf
                        ))
        
        return entities
    
    def _extract_explicit_relationships(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relationship]:
        """Extract explicitly stated relationships."""
        entity_list = ", ".join([f"{e.entity_type}:{e.text}" for e in entities[:20]])
        
        user_prompt = f"""Given the text and entities, extract EXPLICIT relationships.

## Text
{text[:3000]}

## Entities
{entity_list}

## Instructions
Find relationships that are DIRECTLY STATED in the text.
Output as triplets: (Source Entity, Target Entity, Relation Type)

## Output Format
List each relationship as: source -> relation -> target"""

        try:
            messages = [
                {"role": "system", "content": self.RELATIONSHIP_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.executor.client.chat(
                messages=messages,
                temperature=0.2
            )
            
            return self._parse_relationships(response, is_explicit=True)
            
        except Exception:
            return []
    
    def _infer_implicit_relationships(
        self,
        text: str,
        entities: List[Entity],
        explicit_rels: List[Relationship]
    ) -> List[Relationship]:
        """Infer implicit relationships from context."""
        entity_list = ", ".join([f"{e.entity_type}:{e.text}" for e in entities[:20]])
        explicit_list = ", ".join([f"{r.source} -> {r.relation_type} -> {r.target}" for r in explicit_rels])
        
        user_prompt = f"""Given the text, entities, and explicit relationships, INFER implicit relationships.

## Text
{text[:2500]}

## Entities
{entity_list}

## Known Explicit Relationships
{explicit_list if explicit_list else "None found"}

## Instructions
Infer relationships that are IMPLIED but not directly stated.
For each entity pair, generate up to {self.k} implicit relationships.
Score each 0-10 for confidence.

## Output Format
source -> relation -> target (score: X/10)"""

        try:
            messages = [
                {"role": "system", "content": self.RELATIONSHIP_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.executor.client.chat(
                messages=messages,
                temperature=0.4
            )
            
            return self._parse_relationships(response, is_explicit=False)
            
        except Exception:
            return []
    
    def _parse_relationships(self, response: str, is_explicit: bool) -> List[Relationship]:
        """Parse relationships from LLM response."""
        relationships = []
        
        # Pattern: source -> relation -> target (score: X/10)
        pattern = r'([^->\n]+)\s*->\s*([^->\n]+)\s*->\s*([^(\n]+)(?:\(score:\s*(\d+))?'
        matches = re.findall(pattern, response)
        
        for match in matches:
            source = match[0].strip()
            relation = match[1].strip()
            target = match[2].strip()
            score = int(match[3]) if match[3] else 8
            
            # Clean up source/target - remove numbering prefixes like "1. " or "/10)"
            source = re.sub(r'^[\d]+\.\s*', '', source)
            source = re.sub(r'^/\d+\)\s*', '', source)
            target = re.sub(r'^[\d]+\.\s*', '', target)
            target = re.sub(r'/\d+\)?\s*$', '', target).strip()
            
            if source and target and relation and len(source) > 1 and len(target) > 1:
                relationships.append(Relationship(
                    source=source,
                    target=target,
                    relation_type=relation,
                    confidence=score / 10.0,
                    is_explicit=is_explicit
                ))
        
        return relationships
    
    def _filter_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Filter relationships by confidence threshold."""
        return [r for r in relationships if r.confidence * 10 >= self.v_th]
    
    def _organize_by_type(self, entities: List[Entity]) -> Dict[str, List[str]]:
        """Organize entities by type."""
        by_type = {}
        for entity in entities:
            entity_type = entity.entity_type
            if entity_type not in by_type:
                by_type[entity_type] = []
            if entity.text not in by_type[entity_type]:
                by_type[entity_type].append(entity.text)
        return by_type
    
    def _generate_reasoning(
        self,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> str:
        """Generate CoT reasoning summary."""
        by_type = self._organize_by_type(entities)
        
        parts = ["ERA-CoT NER Analysis:"]
        
        # Entity summary
        parts.append(f"\n1. Extracted {len(entities)} entities:")
        for etype, names in by_type.items():
            parts.append(f"   - {etype}: {', '.join(names[:5])}{' (+ more)' if len(names) > 5 else ''}")
        
        # Relationship summary
        if relationships:
            explicit = [r for r in relationships if r.is_explicit]
            implicit = [r for r in relationships if not r.is_explicit]
            parts.append(f"\n2. Found {len(relationships)} relationships:")
            parts.append(f"   - {len(explicit)} explicit (directly stated)")
            parts.append(f"   - {len(implicit)} implicit (inferred)")
        
        return "\n".join(parts)
    
    def _fallback_extraction(self, text: str) -> Dict[str, Any]:
        """Fallback extraction using regex patterns."""
        entities = []
        
        # Date patterns
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b'
        for match in re.finditer(date_pattern, text):
            entities.append(Entity(text=match.group(), entity_type="DATE", start_pos=match.start(), end_pos=match.end()))
        
        # Money patterns
        money_pattern = r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?|\€[\d,]+(?:\.\d{2})?'
        for match in re.finditer(money_pattern, text, re.IGNORECASE):
            entities.append(Entity(text=match.group(), entity_type="MONEY", start_pos=match.start(), end_pos=match.end()))
        
        # Percentage patterns
        percent_pattern = r'\b\d+(?:\.\d+)?%'
        for match in re.finditer(percent_pattern, text):
            entities.append(Entity(text=match.group(), entity_type="PERCENT", start_pos=match.start(), end_pos=match.end()))
        
        return {
            "entities": [e.to_dict() for e in entities],
            "relationships": [],
            "by_type": self._organize_by_type(entities),
            "entity_count": len(entities),
            "relationship_count": 0,
            "reasoning": "Fallback regex extraction (no LLM available)",
            "duration_ms": 0
        }


class NERStep(PipelineStep):
    """
    NER Pipeline Step using ERA-CoT.
    
    Integrates with the CoT pipeline for comprehensive
    named entity extraction with relationships.
    """
    
    def __init__(
        self,
        executor: StepExecutor = None,
        extract_relationships: bool = True,
        use_self_consistency: bool = False
    ):
        super().__init__(
            name="ner",
            description="Named Entity Recognition with ERA-CoT"
        )
        self.ner = ERACotNER(
            executor=executor,
            use_self_consistency=use_self_consistency
        )
        self.extract_relationships = extract_relationships
    
    def get_function_definition(self) -> FunctionDefinition:
        return NER_FUNCTION
    
    def get_cot_prompt(self, context: PipelineContext) -> str:
        return self.ner.ENTITY_SYSTEM_PROMPT
    
    def execute(self, context: PipelineContext) -> StepResult:
        """Execute NER extraction."""
        start_time = time.time()
        
        text = context.current_text or context.original_input
        
        result = self.ner.extract(
            text,
            extract_relationships=self.extract_relationships
        )
        
        return StepResult(
            step_name=self.name,
            status=StepStatus.SUCCESS,
            output={
                "entities": result.get("entities", []),
                "relationships": result.get("relationships", []),
                "by_type": result.get("by_type", {}),
                "entity_count": result.get("entity_count", 0),
                "relationship_count": result.get("relationship_count", 0)
            },
            reasoning=result.get("reasoning", "Extraction completed"),
            confidence=0.9 if result.get("entity_count", 0) > 0 else 0.5,
            duration_ms=result.get("duration_ms", int((time.time() - start_time) * 1000))
        )


# ============== Quick-use functions ==============

def extract_entities(
    text: str,
    api_key: str = None,
    extract_relationships: bool = True,
    use_self_consistency: bool = False
) -> Dict[str, Any]:
    """
    Quick function for NER extraction.
    
    Args:
        text: Input text
        api_key: Optional Groq API key
        extract_relationships: Extract entity relationships
        use_self_consistency: Use SC voting (slower but more accurate)
        
    Returns:
        Dict with entities, relationships, and reasoning
    """
    from ..utils.groq_client import GroqClient
    from ..cot.executor import StepExecutor
    
    try:
        client = GroqClient(api_key=api_key)
        executor = StepExecutor(client)
        ner = ERACotNER(
            executor=executor,
            use_self_consistency=use_self_consistency
        )
        return ner.extract(text, extract_relationships)
    except Exception as e:
        return {
            "entities": [],
            "relationships": [],
            "error": str(e)
        }
