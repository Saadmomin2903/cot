"""
Hallucination Detection and Fact-Checking

Based on learnings from:
- "LLM Reasoning: Why Models Hallucinate and How to Reduce It"
- Fact-checking techniques
- Claim verification

Features:
- Detect unsupported claims
- Verify factual statements
- Identify contradictions
- Confidence scoring
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import time

from . import PipelineStep, PipelineContext, StepResult, StepStatus, FunctionDefinition
from .executor import StepExecutor


@dataclass
class HallucinationCheck:
    """Result of hallucination check."""
    claim: str
    is_supported: bool
    confidence: float
    evidence: Optional[str] = None
    issue_type: str = "unsupported"  # unsupported, contradiction, exaggeration


@dataclass
class HallucinationResult:
    """Overall hallucination detection result."""
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    hallucination_score: float  # 0-1, higher = more hallucinations
    issues: List[HallucinationCheck]
    recommendations: List[str]


# Function definition for hallucination detection
HALLUCINATION_CHECK_FUNCTION = FunctionDefinition(
    name="check_hallucinations",
    description="Detect potential hallucinations, unsupported claims, and factual errors",
    parameters={
        "claims_checked": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "is_supported": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "issue_type": {"type": "string"},
                    "evidence": {"type": "string"}
                }
            },
            "description": "List of claims checked for support"
        },
        "hallucination_score": {
            "type": "number",
            "description": "Overall hallucination score (0-1, higher = more issues)"
        },
        "recommendations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Recommendations to reduce hallucinations"
        },
        "reasoning": {
            "type": "string",
            "description": "Step-by-step reasoning for the check"
        }
    },
    required=["claims_checked", "hallucination_score", "reasoning"]
)


class HallucinationDetector:
    """
    Detect hallucinations and unsupported claims in LLM outputs.
    
    Techniques:
    1. Extract factual claims
    2. Check if claims are supported by source text
    3. Identify contradictions
    4. Flag unsupported statements
    """
    
    def __init__(self, executor: StepExecutor = None):
        """Initialize hallucination detector."""
        self.executor = executor
    
    def detect(
        self,
        text: str,
        source_text: str,
        context: PipelineContext
    ) -> HallucinationResult:
        """
        Detect hallucinations in text compared to source.
        
        Args:
            text: Text to check (e.g., summary, translation)
            source_text: Original source text
            context: Pipeline context
            
        Returns:
            HallucinationResult with detected issues
        """
        if not self.executor:
            # Fallback: simple heuristic check
            return self._simple_check(text, source_text)
        
        # Use LLM to detect hallucinations
        return self._llm_detect(text, source_text, context)
    
    def _simple_check(self, text: str, source_text: str) -> HallucinationResult:
        """Simple heuristic-based check (no LLM)."""
        issues = []
        
        # Extract potential claims (sentences with numbers, names, dates)
        claims = self._extract_claims(text)
        
        for claim in claims:
            # Check if claim appears in source (simple substring match)
            is_supported = self._check_claim_support(claim, source_text)
            
            if not is_supported:
                issues.append(HallucinationCheck(
                    claim=claim,
                    is_supported=False,
                    confidence=0.6,
                    issue_type="unsupported"
                ))
        
        supported = len(claims) - len(issues)
        hallucination_score = len(issues) / len(claims) if claims else 0.0
        
        return HallucinationResult(
            total_claims=len(claims),
            supported_claims=supported,
            unsupported_claims=len(issues),
            hallucination_score=hallucination_score,
            issues=issues,
            recommendations=self._generate_recommendations(issues)
        )
    
    def _llm_detect(
        self,
        text: str,
        source_text: str,
        context: PipelineContext
    ) -> HallucinationResult:
        """Use LLM to detect hallucinations."""
        system_prompt = """You are an expert fact-checker. Your task is to identify potential hallucinations, unsupported claims, and factual errors in text compared to its source.

A hallucination is:
- A claim not supported by the source text
- Information that contradicts the source
- Exaggerated or fabricated details
- Claims that go beyond what the source states

Be thorough but fair - only flag clear issues."""
        
        user_prompt = f"""Source text:
{source_text[:2000]}

Text to check:
{text[:2000]}

Extract all factual claims from the text to check and verify each one against the source text.
Identify any unsupported claims, contradictions, or hallucinations."""

        try:
            result = self.executor.execute_with_function(
                step_name="hallucination_detection",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                function_def=HALLUCINATION_CHECK_FUNCTION,
                context=context,
                temperature=0.1  # Low temp for accuracy
            )
            
            if result.status == StepStatus.SUCCESS:
                output = result.output
                claims_checked = output.get("claims_checked", [])
                
                issues = [
                    HallucinationCheck(
                        claim=c.get("claim", ""),
                        is_supported=c.get("is_supported", True),
                        confidence=c.get("confidence", 0.8),
                        issue_type=c.get("issue_type", "unsupported"),
                        evidence=c.get("evidence")
                    )
                    for c in claims_checked
                ]
                
                unsupported = [i for i in issues if not i.is_supported]
                
                return HallucinationResult(
                    total_claims=len(claims_checked),
                    supported_claims=len(claims_checked) - len(unsupported),
                    unsupported_claims=len(unsupported),
                    hallucination_score=output.get("hallucination_score", 0.0),
                    issues=unsupported,
                    recommendations=output.get("recommendations", [])
                )
        except Exception:
            pass
        
        # Fallback to simple check
        return self._simple_check(text, source_text)
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract potential factual claims from text."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Filter for sentences that look like claims
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Look for indicators of factual claims
            claim_indicators = [
                r'\d+',  # Contains numbers
                r'[A-Z][a-z]+ [A-Z][a-z]+',  # Proper nouns
                r'\b(is|are|was|were|has|have|had)\b',  # Factual statements
            ]
            
            if any(re.search(indicator, sentence) for indicator in claim_indicators):
                claims.append(sentence)
        
        return claims[:10]  # Limit to 10 claims
    
    def _check_claim_support(self, claim: str, source: str) -> bool:
        """Simple check if claim is supported by source."""
        # Extract key terms from claim
        key_terms = re.findall(r'\b[A-Z][a-z]+\b|\d+', claim)
        
        if not key_terms:
            return True  # No specific terms to check
        
        # Check if at least some key terms appear in source
        matches = sum(1 for term in key_terms[:3] if term.lower() in source.lower())
        return matches >= 1
    
    def _generate_recommendations(self, issues: List[HallucinationCheck]) -> List[str]:
        """Generate recommendations to reduce hallucinations."""
        recommendations = []
        
        if not issues:
            return ["No hallucinations detected. Output appears reliable."]
        
        if len(issues) > 3:
            recommendations.append("High number of unsupported claims detected. Consider more conservative summarization.")
        
        if any(i.issue_type == "contradiction" for i in issues):
            recommendations.append("Contradictions found. Review source text carefully.")
        
        if any(i.issue_type == "exaggeration" for i in issues):
            recommendations.append("Exaggerations detected. Stick closer to source text.")
        
        recommendations.append("Consider adding citations or evidence for key claims.")
        
        return recommendations


class HallucinationDetectionStep(PipelineStep):
    """
    Pipeline step that detects hallucinations in previous step outputs.
    
    Can be used after summarization, translation, or any text generation step.
    """
    
    def __init__(
        self,
        target_step: str,
        executor: StepExecutor = None
    ):
        """
        Initialize hallucination detection step.
        
        Args:
            target_step: Name of step to check (e.g., "summary", "translation")
            executor: Executor for LLM-based detection
        """
        super().__init__(
            name=f"hallucination_check_{target_step}",
            description=f"Detect hallucinations in {target_step} output"
        )
        self.target_step = target_step
        self.detector = HallucinationDetector(executor=executor)
    
    def get_function_definition(self) -> FunctionDefinition:
        return HALLUCINATION_CHECK_FUNCTION
    
    def get_cot_prompt(self, context: PipelineContext) -> str:
        target_result = context.step_results.get(self.target_step)
        if not target_result:
            return "No target step result to check."
        
        return f"""Check the output from step '{self.target_step}' for hallucinations and unsupported claims.
        
Compare against the original source text to identify any:
- Unsupported factual claims
- Contradictions
- Exaggerations
- Fabricated information"""
    
    def execute(self, context: PipelineContext) -> StepResult:
        """Execute hallucination detection."""
        start_time = time.time()
        
        target_result = context.step_results.get(self.target_step)
        if not target_result or target_result.status != StepStatus.SUCCESS:
            return StepResult(
                step_name=self.name,
                status=StepStatus.SKIPPED,
                output={"reason": f"Target step '{self.target_step}' not found or failed"},
                reasoning="",
                confidence=0.0,
                duration_ms=int((time.time() - start_time) * 1000)
            )
        
        # Get text to check (extract from output)
        text_to_check = self._extract_text_from_output(target_result.output)
        source_text = context.original_input
        
        # Detect hallucinations
        result = self.detector.detect(
            text=text_to_check,
            source_text=source_text,
            context=context
        )
        
        # Adjust confidence of target step if hallucinations found
        if result.hallucination_score > 0.3:
            target_result.confidence *= (1.0 - result.hallucination_score)
        
        return StepResult(
            step_name=self.name,
            status=StepStatus.SUCCESS,
            output={
                "hallucination_score": result.hallucination_score,
                "total_claims": result.total_claims,
                "supported_claims": result.supported_claims,
                "unsupported_claims": result.unsupported_claims,
                "issues": [
                    {
                        "claim": i.claim,
                        "issue_type": i.issue_type,
                        "confidence": i.confidence
                    }
                    for i in result.issues[:5]  # Top 5 issues
                ],
                "recommendations": result.recommendations
            },
            reasoning=f"Checked {result.total_claims} claims, found {result.unsupported_claims} unsupported. Hallucination score: {result.hallucination_score:.2f}",
            confidence=1.0 - result.hallucination_score,  # Higher confidence = fewer hallucinations
            duration_ms=int((time.time() - start_time) * 1000)
        )
    
    def _extract_text_from_output(self, output: Dict[str, Any]) -> str:
        """Extract text content from step output."""
        # Try common fields
        for field in ["summary", "translated_text", "text", "content", "output"]:
            if field in output and isinstance(output[field], str):
                return output[field]
        
        # Fallback: convert entire output to string
        return str(output)

