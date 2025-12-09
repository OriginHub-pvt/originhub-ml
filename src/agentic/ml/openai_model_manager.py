"""
OpenAI Model Manager
====================

Provides a uniform interface to OpenAI's API models (GPT-3.5, GPT-4, etc.)
Maintains the same interface as ModelManager for seamless integration.
"""

import os
from threading import Lock
from typing import Optional, Dict, Any


class OpenAIModel:
    """Wrapper for OpenAI API calls with llama.cpp-compatible interface."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI model wrapper.
        
        Parameters
        ----------
        api_key : str
            OpenAI API key
        model_name : str
            OpenAI model name (gpt-3.5-turbo, gpt-4, etc.)
        """
        self.api_key = api_key
        self.model_name = model_name
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenAI client not installed. Install with: pip install openai"
            )
    
    def __call__(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None,
        echo: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call OpenAI API with llama.cpp-compatible interface.
        
        Parameters
        ----------
        prompt : str
            Input prompt
        max_tokens : int
            Maximum tokens to generate
        temperature : float
            Sampling temperature
        top_p : float
            Nucleus sampling parameter
        stop : list
            Stop sequences
        echo : bool
            Whether to include prompt in response (ignored for OpenAI)
        **kwargs : dict
            Additional arguments (ignored)
        
        Returns
        -------
        dict
            Response in llama.cpp format with 'choices' key
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
            
            # Convert to llama.cpp format for compatibility
            generated_text = response.choices[0].message.content
            
            return {
                "choices": [
                    {
                        "text": generated_text,
                        "finish_reason": response.choices[0].finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            }
        
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")


class OpenAIModelManager:
    """
    Model manager for OpenAI API.
    Maintains the same interface as ModelManager for seamless switching.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        heavy_model: str = "gpt-4",
        light_model: str = "gpt-3.5-turbo",
        **kwargs
    ):
        """
        Initialize OpenAI Model Manager.
        
        Parameters
        ----------
        api_key : str, optional
            OpenAI API key. If None, uses OPENAI_API_KEY environment variable
        heavy_model : str
            Model for heavy operations (gpt-4, gpt-4-turbo, etc.)
        light_model : str
            Model for light operations (gpt-3.5-turbo, gpt-3.5-turbo-16k, etc.)
        **kwargs : dict
            Additional arguments (for compatibility)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not provided and not found in environment. "
                "Please set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        self.heavy_model_name = heavy_model
        self.light_model_name = light_model
        
        # Initialize models
        self.heavy = OpenAIModel(self.api_key, heavy_model)
        self.light = OpenAIModel(self.api_key, light_model)
        
        # Thread locks for safety
        self.heavy_lock = Lock()
        self.light_lock = Lock()
        
        print(f"[OpenAIModelManager] Initialized with heavy={heavy_model}, light={light_model}")
    
    def get(self, heavy: bool = False) -> tuple:
        """
        Get model and its associated lock.
        
        Parameters
        ----------
        heavy : bool
            If True, returns heavy model; otherwise returns light model
        
        Returns
        -------
        tuple
            (model, lock) tuple
        """
        if heavy:
            return self.heavy, self.heavy_lock
        else:
            return self.light, self.light_lock
    
    def close(self):
        """Cleanup (no-op for OpenAI, but maintains interface compatibility)."""
        print("[OpenAIModelManager] Closing (no cleanup needed for API)")
