"""
Inference Engine Module
=======================

Provides unified generate() interface for both LLMs.
"""

from typing import List, Optional
from .model_manager import ModelManager


class InferenceEngine:
    """
    Unified API for LLM inference across all agents.
    """

    def __init__(self, manager: ModelManager):
        """
        Parameters
        ----------
        manager : ModelManager
            Model manager instance containing loaded models.
        """
        self.manager = manager

    def generate(
        self,
        prompt: str,
        heavy: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generates text using heavy or light LLM.

        Parameters
        ----------
        prompt : str
            Text prompt for generation.
        heavy : bool
            If True uses 7B model; else uses 1.5B.
        max_tokens : int
            Maximum output token count.
        temperature : float
            Sampling temperature.
        top_p : float
            Nucleus sampling probability.
        stop : list of str or None
            Stop sequences for generation.

        Returns
        -------
        str
            Generated text.
        """
        model, lock = self.manager.get(heavy)

        with lock:
            output = model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                echo=False,
            )

        return output["choices"][0]["text"].strip()
