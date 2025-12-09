"""
Inference Engine
================

Provides a uniform interface for generating text from either the heavy
or light model, using thread-safe execution.
"""

from typing import Optional
from src.agentic.ml.model_manager import ModelManager


class InferenceEngine:
    """
    Handles inference routing between heavy/light models.
    """

    def __init__(self, model_manager: ModelManager):
        """
        Parameters
        ----------
        model_manager : ModelManager
            Instance providing access to both models and their locks.
        """
        self.manager = model_manager

    def generate(
        self,
        prompt: str,
        heavy: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None,
    ) -> str:
        """
        Generate text using the selected model.

        Parameters
        ----------
        prompt : str
            Input prompt string.
        heavy : bool
            Whether to use the heavy (7B) model.
        max_tokens : int
            Maximum number of tokens to generate.
        temperature : float
            Sampling temperature.
        top_p : float
            Nucleus sampling parameter.
        stop : list
            List of stopping sequences.

        Returns
        -------
        str
            Generated text output.
        """
        model, lock = self.manager.get(heavy=heavy)
        with lock:
            result = model(
                prompt=prompt,           # ← FIXED (test expects this)
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                echo=False,
            )

        # Expected llama.cpp output structure
        text = result.get("choices", [{}])[0].get("text", "")
        return text.strip()
