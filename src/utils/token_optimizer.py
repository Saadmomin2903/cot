"""
Token Optimization Utilities

Based on learnings from:
- "Teaching LLMs to Stop Wasting Tokens" (CTON - Compact Token-Oriented Notation)
- Prompt compression techniques
- Token counting and budget management

Features:
- Prompt compression (remove redundancy, use abbreviations)
- Token counting and budget tracking
- Compact representation formats
- Context window management
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TokenBudget:
    """Track token usage and budget."""
    total_budget: int = 4096
    used_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    def remaining(self) -> int:
        """Get remaining token budget."""
        return max(0, self.total_budget - self.used_tokens)
    
    def can_fit(self, estimated_tokens: int) -> bool:
        """Check if estimated tokens can fit in budget."""
        return self.remaining() >= estimated_tokens
    
    def add_usage(self, prompt: int, completion: int):
        """Add token usage."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.used_tokens = self.prompt_tokens + self.completion_tokens


class TokenOptimizer:
    """
    Optimize prompts and text to reduce token usage.
    
    Techniques:
    1. Remove redundant whitespace and formatting
    2. Compress repetitive patterns
    3. Use abbreviations for common phrases
    4. Truncate long examples while preserving meaning
    5. Remove unnecessary explanations
    """
    
    # Common phrase abbreviations (CTON-like)
    ABBREVIATIONS = {
        r'\bfor example\b': 'e.g.',
        r'\bthat is\b': 'i.e.',
        r'\bwith respect to\b': 'wrt',
        r'\bsuch as\b': 'e.g.',
        r'\bin other words\b': 'i.e.',
        r'\bplease note that\b': 'Note:',
        r'\bit is important to\b': 'Important:',
        r'\bkeep in mind that\b': 'Remember:',
        r'\bdo not\b': "don't",
        r'\bcannot\b': "can't",
        r'\bwill not\b': "won't",
        r'\bwould not\b': "wouldn't",
        r'\bshould not\b': "shouldn't",
    }
    
    # Patterns to remove
    REDUNDANT_PATTERNS = [
        r'\s+',  # Multiple spaces
        r'\n{3,}',  # Multiple newlines
        r'^\s*[-*]\s*',  # List markers at start
        r'\s*[-*]\s*$',  # List markers at end
    ]
    
    def __init__(self, aggressive: bool = False):
        """
        Initialize token optimizer.
        
        Args:
            aggressive: If True, use more aggressive compression
        """
        self.aggressive = aggressive
    
    def compress_prompt(self, prompt: str, max_tokens: Optional[int] = None) -> Tuple[str, int]:
        """
        Compress a prompt to reduce token usage.
        
        Args:
            prompt: Original prompt text
            max_tokens: Maximum tokens allowed (will truncate if needed)
            
        Returns:
            (compressed_prompt, estimated_tokens)
        """
        compressed = prompt
        
        # Step 1: Apply abbreviations
        for pattern, replacement in self.ABBREVIATIONS.items():
            compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)
        
        # Step 2: Remove redundant whitespace
        compressed = re.sub(r'\s+', ' ', compressed)
        compressed = re.sub(r'\n{3,}', '\n\n', compressed)
        compressed = compressed.strip()
        
        # Step 3: Remove unnecessary explanations if aggressive
        if self.aggressive:
            # Remove common verbose phrases
            verbose_patterns = [
                r'Please note that\s+',
                r'It is worth mentioning that\s+',
                r'It should be noted that\s+',
                r'Keep in mind that\s+',
            ]
            for pattern in verbose_patterns:
                compressed = re.sub(pattern, '', compressed, flags=re.IGNORECASE)
        
        # Step 4: Truncate examples if too long
        if max_tokens:
            estimated = self.estimate_tokens(compressed)
            if estimated > max_tokens:
                compressed = self._truncate_smart(compressed, max_tokens)
        
        estimated_tokens = self.estimate_tokens(compressed)
        return compressed, estimated_tokens
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation).
        
        Uses simple heuristic: ~4 characters per token for English.
        More accurate for English, less for other languages.
        """
        # Simple heuristic: ~4 chars per token
        # For code/mixed content, might be ~3 chars per token
        return len(text) // 4
    
    def _truncate_smart(self, text: str, max_tokens: int) -> str:
        """
        Intelligently truncate text while preserving structure.
        
        Prefers to:
        - Keep beginning and end
        - Remove middle sections
        - Preserve sentence boundaries
        """
        max_chars = max_tokens * 4  # Rough estimate
        
        if len(text) <= max_chars:
            return text
        
        # Keep first 40% and last 40%, remove middle 20%
        keep_start = int(max_chars * 0.4)
        keep_end = int(max_chars * 0.4)
        
        start = text[:keep_start]
        end = text[-keep_end:]
        
        # Try to end start at sentence boundary
        last_period = start.rfind('.')
        if last_period > keep_start * 0.8:
            start = start[:last_period + 1]
        
        return f"{start}\n\n[... truncated ...]\n\n{end}"
    
    def compress_context(self, context: Dict[str, Any]) -> str:
        """
        Compress context dictionary into compact string.
        
        Uses CTON-like compact notation for structured data.
        """
        parts = []
        
        for key, value in context.items():
            if isinstance(value, str):
                # Truncate long strings
                if len(value) > 200:
                    value = value[:200] + "..."
                parts.append(f"{key}:{value}")
            elif isinstance(value, (int, float, bool)):
                parts.append(f"{key}:{value}")
            elif isinstance(value, list):
                if len(value) <= 3:
                    parts.append(f"{key}:[{','.join(str(v)[:50] for v in value)}]")
                else:
                    parts.append(f"{key}:[{len(value)} items]")
            elif isinstance(value, dict):
                parts.append(f"{key}:{{...}}")
        
        return "|".join(parts)


class PromptBuilder:
    """
    Build optimized prompts with token awareness.
    
    Features:
    - Automatic compression
    - Token budget tracking
    - Context window management
    """
    
    def __init__(self, optimizer: TokenOptimizer = None, budget: TokenBudget = None):
        """Initialize prompt builder."""
        self.optimizer = optimizer or TokenOptimizer()
        self.budget = budget or TokenBudget()
    
    def build_cot_prompt(
        self,
        task: str,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
        max_prompt_tokens: int = 2000
    ) -> Tuple[str, int]:
        """
        Build a chain-of-thought prompt with token optimization.
        
        Args:
            task: Task description
            input_text: Input text to process
            context: Optional context from previous steps
            max_prompt_tokens: Maximum tokens for prompt
            
        Returns:
            (prompt, estimated_tokens)
        """
        # Build base prompt
        prompt_parts = [
            f"Task: {task}",
            "",
            "Think step-by-step:",
            "1. Understand the task",
            "2. Analyze the input",
            "3. Consider edge cases",
            "4. Provide structured output",
            "",
        ]
        
        # Add context if available
        if context:
            compressed_context = self.optimizer.compress_context(context)
            prompt_parts.append(f"Context: {compressed_context}")
            prompt_parts.append("")
        
        # Add input (may need truncation)
        input_budget = max_prompt_tokens - self.optimizer.estimate_tokens('\n'.join(prompt_parts)) - 100
        if input_budget > 0:
            input_tokens = self.optimizer.estimate_tokens(input_text)
            if input_tokens > input_budget:
                # Truncate input
                max_chars = input_budget * 4
                input_text = input_text[:max_chars] + "\n[... truncated ...]"
            prompt_parts.append(f"Input:\n{input_text}")
        
        prompt = '\n'.join(prompt_parts)
        
        # Compress if needed
        compressed, tokens = self.optimizer.compress_prompt(prompt, max_prompt_tokens)
        
        return compressed, tokens
    
    def build_review_prompt(
        self,
        original_query: str,
        response: str,
        focus: str = "accuracy"
    ) -> str:
        """
        Build a prompt for reviewing/refining a response.
        
        Args:
            original_query: Original question/task
            response: Response to review
            focus: What to focus on (accuracy, completeness, clarity)
        """
        focus_instructions = {
            "accuracy": "Identify any factual errors, hallucinations, or unsupported claims.",
            "completeness": "Identify any missing information or incomplete answers.",
            "clarity": "Identify any unclear explanations or confusing parts.",
        }
        
        return f"""Review the following response to improve {focus}:

Original query: {original_query}

Response to review:
{response}

Focus: {focus_instructions.get(focus, focus_instructions['accuracy'])}

Provide specific feedback and suggest improvements."""


def count_tokens_approximate(text: str) -> int:
    """
    Quick token count approximation.
    
    Uses simple heuristic: ~4 characters per token.
    """
    return len(text) // 4

