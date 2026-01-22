"""
Chain of LLMs (CoLLM) - Collaborative Review and Refinement

Based on: https://dev.to/daviducolo/chain-of-llms-a-collaborative-approach-to-ai-problem-solving-533

Implements:
- Multiple LLM models reviewing each other's outputs
- Iterative refinement through feedback loops
- Consensus building
- Error correction through collaboration
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from . import PipelineStep, PipelineContext, StepResult, StepStatus, FunctionDefinition
from .executor import StepExecutor
from ..utils.token_optimizer import PromptBuilder, TokenBudget


@dataclass
class ReviewResult:
    """Result from a model review."""
    reviewer_name: str
    feedback: str
    issues_found: List[str]
    suggestions: List[str]
    confidence: float = 0.8


@dataclass
class CollaborativeResult:
    """Result from collaborative review process."""
    original_output: Dict[str, Any]
    refined_output: Dict[str, Any]
    review_history: List[ReviewResult]
    consensus_score: float
    iterations: int


# Function definition for review step
REVIEW_FUNCTION = FunctionDefinition(
    name="review_response",
    description="Review and provide feedback on a response to improve accuracy and completeness",
    parameters={
        "issues_found": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of specific issues, errors, or gaps identified"
        },
        "suggestions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific suggestions for improvement"
        },
        "feedback": {
            "type": "string",
            "description": "Overall feedback on the response quality"
        },
        "confidence_in_original": {
            "type": "number",
            "description": "Confidence in original response (0-1)"
        },
        "reasoning": {
            "type": "string",
            "description": "Step-by-step reasoning for the review"
        }
    },
    required=["issues_found", "suggestions", "feedback", "reasoning"]
)


class CollaborativeReviewer:
    """
    Implements Chain of LLMs (CoLLM) for collaborative review using a SINGLE model.
    
    Uses one model with different perspectives/prompts to simulate multiple reviewers:
    1. Initial response generation
    2. Review from different perspectives (accuracy, completeness, clarity)
    3. Iterative refinement
    4. Consensus building through multiple review passes
    
    This approach allows collaborative review even with a single model by:
    - Varying review focus (accuracy, completeness, clarity)
    - Using different temperature settings
    - Asking different types of questions
    - Multiple review passes with different prompts
    """
    
    # Different review perspectives to simulate multiple reviewers
    REVIEW_PERSPECTIVES = [
        {
            "name": "accuracy_reviewer",
            "focus": "accuracy",
            "instruction": "You are a strict fact-checker. Focus on identifying factual errors, unsupported claims, and contradictions.",
            "temperature": 0.1
        },
        {
            "name": "completeness_reviewer",
            "focus": "completeness",
            "instruction": "You are a thorough analyst. Focus on identifying missing information, incomplete answers, and gaps in coverage.",
            "temperature": 0.2
        },
        {
            "name": "clarity_reviewer",
            "focus": "clarity",
            "instruction": "You are a communication expert. Focus on identifying unclear explanations, confusing parts, and areas needing better structure.",
            "temperature": 0.15
        }
    ]
    
    def __init__(
        self,
        executor: StepExecutor,
        num_reviews: int = 3,
        max_iterations: int = 2,
        consensus_threshold: float = 0.8
    ):
        """
        Initialize collaborative reviewer with a SINGLE model.
        
        Args:
            executor: Single executor (model) to use for all reviews
            num_reviews: Number of review passes to perform (uses different perspectives)
            max_iterations: Maximum refinement iterations
            consensus_threshold: Threshold for consensus (0-1)
        """
        self.executor = executor
        self.num_reviews = min(num_reviews, len(self.REVIEW_PERSPECTIVES))
        self.max_iterations = max_iterations
        self.consensus_threshold = consensus_threshold
        self.prompt_builder = PromptBuilder()
    
    def review_and_refine(
        self,
        step_name: str,
        original_query: str,
        original_response: Dict[str, Any],
        system_prompt: str,
        context: PipelineContext
    ) -> CollaborativeResult:
        """
        Review and refine a response using collaborative LLM approach.
        
        Args:
            step_name: Name of the step being reviewed
            original_query: Original query/task
            original_response: Response to review
            system_prompt: System prompt for the task
            context: Pipeline context
            
        Returns:
            CollaborativeResult with refined output
        """
        current_output = original_response
        review_history = []
        
        for iteration in range(self.max_iterations):
            # Step 1: Get reviews from single model with different perspectives
            reviews = []
            for i in range(self.num_reviews):
                perspective = self.REVIEW_PERSPECTIVES[i % len(self.REVIEW_PERSPECTIVES)]
                review = self._get_review_with_perspective(
                    original_query,
                    current_output,
                    system_prompt,
                    perspective
                )
                if review:
                    reviews.append(review)
                    review_history.append(review)
            
            if not reviews:
                break
            
            # Step 2: Check for consensus
            consensus_score = self._calculate_consensus(reviews)
            if consensus_score >= self.consensus_threshold:
                # High consensus - no major issues found
                break
            
            # Step 3: Refine based on feedback
            if iteration < self.max_iterations - 1:
                current_output = self._refine_output(
                    original_query,
                    current_output,
                    reviews,
                    system_prompt
                )
        
        return CollaborativeResult(
            original_output=original_response,
            refined_output=current_output,
            review_history=review_history,
            consensus_score=consensus_score if review_history else 1.0,
            iterations=len(review_history)
        )
    
    def _get_review_with_perspective(
        self,
        original_query: str,
        response: Dict[str, Any],
        system_prompt: str,
        perspective: Dict[str, Any]
    ) -> Optional[ReviewResult]:
        """
        Get review from single model using a specific perspective.
        
        Uses the same model but with different:
        - Focus area (accuracy, completeness, clarity)
        - Instructions
        - Temperature
        - Prompt structure
        
        This simulates having multiple reviewers even with one model.
        """
        try:
            # Build review prompt with specific focus
            review_prompt = self.prompt_builder.build_review_prompt(
                original_query,
                str(response),
                focus=perspective["focus"]
            )
            
            # Enhance system prompt with perspective-specific instruction
            enhanced_system = f"""You are an expert reviewer with a specific focus.

{perspective['instruction']}

Your task is to review the response from this specific perspective and provide detailed, constructive feedback."""
            
            # Execute review with perspective-specific temperature
            result = self.executor.execute_with_function(
                step_name=f"review_{perspective['name']}",
                system_prompt=enhanced_system,
                user_prompt=review_prompt,
                function_def=REVIEW_FUNCTION,
                context=PipelineContext(original_input=original_query),
                temperature=perspective["temperature"]
            )
            
            if result.status == StepStatus.SUCCESS:
                output = result.output
                return ReviewResult(
                    reviewer_name=perspective["name"],
                    feedback=output.get("feedback", ""),
                    issues_found=output.get("issues_found", []),
                    suggestions=output.get("suggestions", []),
                    confidence=output.get("confidence_in_original", 0.8)
                )
        except Exception as e:
            # Review failed - continue with other perspectives
            pass
        
        return None
    
    def _calculate_consensus(self, reviews: List[ReviewResult]) -> float:
        """
        Calculate consensus score from reviews.
        
        Higher score = more agreement that response is good.
        """
        if not reviews:
            return 1.0
        
        # Average confidence scores
        avg_confidence = sum(r.confidence for r in reviews) / len(reviews)
        
        # Penalize if many issues found
        total_issues = sum(len(r.issues_found) for r in reviews)
        issue_penalty = min(0.3, total_issues * 0.05)
        
        consensus = avg_confidence - issue_penalty
        return max(0.0, min(1.0, consensus))
    
    def _refine_output(
        self,
        original_query: str,
        current_output: Dict[str, Any],
        reviews: List[ReviewResult],
        system_prompt: str
    ) -> Dict[str, Any]:
        """Refine output based on review feedback from multiple perspectives."""
        # Collect all issues and suggestions from all review perspectives
        all_issues = []
        all_suggestions = []
        for review in reviews:
            all_issues.extend(review.issues_found)
            all_suggestions.extend(review.suggestions)
        
        # Remove duplicates while preserving order
        seen_issues = set()
        unique_issues = []
        for issue in all_issues:
            issue_lower = issue.lower()
            if issue_lower not in seen_issues:
                seen_issues.add(issue_lower)
                unique_issues.append(issue)
        
        seen_suggestions = set()
        unique_suggestions = []
        for sug in all_suggestions:
            sug_lower = sug.lower()
            if sug_lower not in seen_suggestions:
                seen_suggestions.add(sug_lower)
                unique_suggestions.append(sug)
        
        # Build refinement prompt
        refinement_prompt = f"""Original query: {original_query}

Current response:
{current_output}

Issues identified by reviewers:
{chr(10).join(f"- {issue}" for issue in unique_issues[:8])}

Suggestions for improvement:
{chr(10).join(f"- {sug}" for sug in unique_suggestions[:8])}

Please provide an improved response that:
1. Addresses all identified issues
2. Incorporates the suggestions
3. Maintains accuracy and completeness
4. Improves clarity where needed"""
        
        try:
            # Use the same executor to refine (single model approach)
            result = self.executor.execute_with_function(
                step_name="refine_response",
                system_prompt=system_prompt + "\n\nYou are refining a response based on comprehensive feedback from multiple review perspectives. Synthesize all feedback and create an improved version.",
                user_prompt=refinement_prompt,
                function_def=REVIEW_FUNCTION,  # Reuse review function for structure
                context=PipelineContext(original_input=original_query),
                temperature=0.1  # Low temperature for focused refinement
            )
            
            if result.status == StepStatus.SUCCESS:
                return result.output
        except Exception:
            pass
        
        # Fallback: return original if refinement fails
        return current_output


class CollaborativeReviewStep(PipelineStep):
    """
    Pipeline step that applies collaborative review to previous step outputs.
    
    Uses a SINGLE model with different perspectives to simulate multiple reviewers.
    Can be inserted after any step to improve quality through collaborative review.
    """
    
    def __init__(
        self,
        target_step: str,
        executor: StepExecutor,
        num_reviews: int = 3,
        max_iterations: int = 2
    ):
        """
        Initialize collaborative review step with SINGLE model.
        
        Args:
            target_step: Name of step to review
            executor: Single executor (model) to use for all reviews
            num_reviews: Number of review passes with different perspectives (default: 3)
            max_iterations: Max refinement iterations
        """
        super().__init__(
            name=f"collaborative_review_{target_step}",
            description=f"Collaborative review and refinement of {target_step} output (single model, multiple perspectives)"
        )
        self.target_step = target_step
        self.reviewer = CollaborativeReviewer(
            executor=executor,
            num_reviews=num_reviews,
            max_iterations=max_iterations
        )
    
    def get_function_definition(self) -> FunctionDefinition:
        return REVIEW_FUNCTION
    
    def get_cot_prompt(self, context: PipelineContext) -> str:
        target_result = context.step_results.get(self.target_step)
        if not target_result:
            return "No target step result to review."
        
        return f"""Review the output from step '{self.target_step}' for accuracy, completeness, and quality.
        
Original output:
{target_result.output}
        
Provide critical feedback and suggestions for improvement."""
    
    def execute(self, context: PipelineContext) -> StepResult:
        """Execute collaborative review."""
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
        
        # Perform collaborative review using single model with multiple perspectives
        collaborative_result = self.reviewer.review_and_refine(
            step_name=self.target_step,
            original_query=context.original_input[:500],
            original_response=target_result.output,
            system_prompt="Review and refine the response for accuracy and completeness.",
            context=context
        )
        
        # Update target step result if improved
        if collaborative_result.consensus_score < 0.7:  # Significant improvement made
            target_result.output = collaborative_result.refined_output
            target_result.confidence = collaborative_result.consensus_score
        
        return StepResult(
            step_name=self.name,
            status=StepStatus.SUCCESS,
            output={
                "original_output": collaborative_result.original_output,
                "refined_output": collaborative_result.refined_output,
                "consensus_score": collaborative_result.consensus_score,
                "iterations": collaborative_result.iterations,
                "reviews": [
                    {
                        "reviewer": r.reviewer_name,
                        "issues": r.issues_found,
                        "suggestions": r.suggestions
                    }
                    for r in collaborative_result.review_history
                ]
            },
            reasoning=f"Collaborative review completed with {collaborative_result.iterations} reviews, consensus: {collaborative_result.consensus_score:.2f}",
            confidence=collaborative_result.consensus_score,
            duration_ms=int((time.time() - start_time) * 1000)
        )

