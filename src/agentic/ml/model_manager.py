"""
Model Manager Module
====================

Handles loading and managing LLM models (Qwen 7B and Qwen 1.5B) using llama.cpp.
Provides thread-safe access to both models.
"""

import threading
from typing import Tuple
from llama_cpp import Llama


class ModelManager:
    """
    Loads and manages multiple LLMs with thread-safe access.
    """

    def __init__(
        self,
        model7b_path: str,
        model1b_path: str,
        context_size: int = 8192,
        n_threads: int = 8,
        gpu_layers: int = 0,
    ):
        """
        Parameters
        ----------
        model7b_path : str
            Path to heavy 7B GGUF model.
        model1b_path : str
            Path to light 1.5B GGUF model.
        context_size : int
            Maximum context window size.
        n_threads : int
            Number of CPU inference threads.
        gpu_layers : int
            Number of model layers to offload to GPU.
        """
        self.model7b_path = model7b_path
        self.model1b_path = model1b_path
        self.context_size = context_size
        self.n_threads = n_threads
        self.gpu_layers = gpu_layers

        self._lock_7b = threading.Lock()
        self._lock_1b = threading.Lock()

        self.modelA = None
        self.modelB = None

        self._load_models()

    def _load_models(self) -> None:
        """Loads both LLMs into memory."""
        self.modelA = Llama(
            model_path=self.model7b_path,
            n_ctx=self.context_size,
            n_threads=self.n_threads,
            n_gpu_layers=self.gpu_layers,
            verbose=False
        )

        self.modelB = Llama(
            model_path=self.model1b_path,
            n_ctx=self.context_size,
            n_threads=self.n_threads,
            n_gpu_layers=self.gpu_layers,
            verbose=False
        )

    def get(self, heavy: bool = False) -> Tuple[Llama, threading.Lock]:
        """
        Returns the requested model and associated lock.

        Parameters
        ----------
        heavy : bool
            If True returns 7B model; else returns 1.5B model.

        Returns
        -------
        (Llama, threading.Lock)
            Model instance and its thread lock.
        """
        if heavy:
            return self.modelA, self._lock_7b
        return self.modelB, self._lock_1b
