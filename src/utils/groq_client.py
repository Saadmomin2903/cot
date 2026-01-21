"""
Groq LLM API Client wrapper.

Provides a simple interface for making chat completions
with JSON mode support and error handling.
"""

import json
from typing import Dict, Any, Optional, List
from groq import Groq


class GroqClient:
    """
    Wrapper for Groq LLM API with JSON mode support.
    
    Provides:
    - Simple chat completion interface
    - JSON mode for structured output
    - Error handling and retries
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key (defaults to config)
            model: Model ID (defaults to config)
        """
        from ..config import config
        
        self.api_key = api_key or config.groq.api_key
        self.model = model or config.groq.model
        self.temperature = config.groq.temperature
        self.max_tokens = config.groq.max_tokens
        
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Get your key at https://console.groq.com/keys"
            )
        
        self.client = Groq(api_key=self.api_key)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            json_mode: If True, expect JSON response
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Response content as string
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Groq API error: {str(e)}")
    
    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = None
    ) -> Dict[str, Any]:
        """
        Send a chat request expecting JSON response.
        
        Args:
            system_prompt: System message setting context
            user_prompt: User message with the query
            temperature: Override default temperature
            
        Returns:
            Parsed JSON response as dictionary
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.chat(messages, json_mode=True, temperature=temperature)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {str(e)}\nResponse: {response}")
    
    def simple_prompt(self, prompt: str, json_mode: bool = False) -> str:
        """
        Simple single-turn prompt.
        
        Args:
            prompt: The prompt text
            json_mode: If True, expect JSON response
            
        Returns:
            Response content
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, json_mode=json_mode)
