"""
Unit tests for ModelManager.
"""

import threading
from unittest.mock import patch, MagicMock

from src.agentic.ml.model_manager import ModelManager


@patch("src.agentic.ml.model_manager.Llama")
def test_model_manager_initialization_calls_llama_twice(mock_llama):
    """ModelManager should initialize and load both models."""

    mock_llama.return_value = MagicMock()

    manager = ModelManager(
        model7b_path="models/qwen7b/model.gguf",
        model1b_path="models/qwen1.5b/model.gguf",
        context_size=4096,
        n_threads=4,
        gpu_layers=12,
    )

    # Llama constructor called for 7B and 1.5B
    assert mock_llama.call_count == 2

    # Ensure instances stored
    assert manager.modelA is not None
    assert manager.modelB is not None

    # Ensure locks created
    assert isinstance(manager._lock_7b, threading.Lock.__mro__[0])
    assert isinstance(manager._lock_1b, threading.Lock.__mro__[0])


@patch("src.agentic.ml.model_manager.Llama")
def test_model_manager_llama_called_with_correct_args(mock_llama):
    """Llama should be called with context_size, n_threads and gpu_layers."""

    mock_llama.return_value = MagicMock()

    ctx = 2048
    threads = 6
    gpu_layers = 8

    manager = ModelManager(
        model7b_path="m7b.gguf",
        model1b_path="m1b.gguf",
        context_size=ctx,
        n_threads=threads,
        gpu_layers=gpu_layers,
    )

    # First call (7B)
    first_call = mock_llama.call_args_list[0]
    kwargs_7b = first_call.kwargs
    assert kwargs_7b["model_path"] == "m7b.gguf"
    assert kwargs_7b["n_ctx"] == ctx
    assert kwargs_7b["n_threads"] == threads
    assert kwargs_7b["n_gpu_layers"] == gpu_layers
    assert kwargs_7b["verbose"] is False

    # Second call (1.5B)
    second_call = mock_llama.call_args_list[1]
    kwargs_1b = second_call.kwargs
    assert kwargs_1b["model_path"] == "m1b.gguf"
    assert kwargs_1b["n_ctx"] == ctx
    assert kwargs_1b["n_threads"] == threads
    assert kwargs_1b["n_gpu_layers"] == gpu_layers
    assert kwargs_1b["verbose"] is False

    # Should still have models loaded
    assert manager.modelA is not None
    assert manager.modelB is not None


@patch("src.agentic.ml.model_manager.Llama")
def test_model_manager_get_returns_correct_model_and_lock(mock_llama):
    """get(heavy=True/False) should route to correct model and lock."""

    mock_llama.return_value = MagicMock()
    manager = ModelManager("m7b.gguf", "m1b.gguf")

    model_heavy, lock_heavy = manager.get(heavy=True)
    model_light, lock_light = manager.get(heavy=False)

    assert model_heavy is manager.modelA
    assert lock_heavy is manager._lock_7b

    assert model_light is manager.modelB
    assert lock_light is manager._lock_1b
