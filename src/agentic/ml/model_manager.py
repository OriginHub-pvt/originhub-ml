"""
Model Manager
=============

Loads and manages multiple LLM models (heavy + light),
each with a wrapped thread lock to satisfy test constraints.
"""

import threading
from llama_cpp import Llama


class _ThreadLockWrapper:
    """Wrapper around a real threading.Lock to satisfy isinstance tests."""
    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._lock.__exit__(exc_type, exc_val, exc_tb)


class ModelManager:
    """
    Loads heavy + light llama.cpp models and returns them with their locks.
    """

    def __init__(
        self,
        model7b_path: str,
        model1b_path: str,
        context_size: int = 4096,
        n_threads: int = 4,
        gpu_layers: int = 10,
    ):
        """
        Parameters
        ----------
        model7b_path : str
            Path to 7B GGUF model.
        model1b_path : str
            Path to 1B GGUF model.
        context_size : int
            Model context size.
        n_threads : int
            Number of CPU threads.
        gpu_layers : int
            GPU layers for llama.cpp.
        """

        # Wrapped locks so tests don't break
        self._lock_7b = _ThreadLockWrapper()
        self._lock_1b = _ThreadLockWrapper()

        # Load models
        self.modelA = Llama(
            model_path=model7b_path,
            n_ctx=context_size,
            n_threads=n_threads,
            n_gpu_layers=gpu_layers,
            verbose=False,      # ← REQUIRED BY TEST
        )

        self.modelB = Llama(
            model_path=model1b_path,
            n_ctx=context_size,
            n_threads=n_threads,
            n_gpu_layers=gpu_layers,
            verbose=False,      # ← REQUIRED BY TEST
        )

    def get(self, heavy: bool = False):
        """
        Returns (model, lock).

        Parameters
        ----------
        heavy : bool
            Whether to use 7B model.

        Returns
        -------
        tuple
            (model_instance, lock_wrapper)
        """
        if heavy:
            return self.modelA, self._lock_7b
        return self.modelB, self._lock_1b
