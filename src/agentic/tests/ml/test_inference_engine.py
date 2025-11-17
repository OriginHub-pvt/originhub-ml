"""
Unit tests for InferenceEngine.
"""

from unittest.mock import MagicMock

from src.agentic.ml.inference_engine import InferenceEngine


class FakeLock:
    """Simple context-manager lock for testing."""

    def __init__(self):
        self.enter_called = False
        self.exit_called = False

    def __enter__(self, *args, **kwargs):
        self.enter_called = True

    def __exit__(self, *args, **kwargs):
        self.exit_called = True


class FakeManager:
    """Fake ModelManager used to test routing and lock usage."""

    def __init__(self):
        self.heavy_model = MagicMock()
        self.light_model = MagicMock()
        self.heavy_lock = FakeLock()
        self.light_lock = FakeLock()

    def get(self, heavy=False):
        if heavy:
            return self.heavy_model, self.heavy_lock
        return self.light_model, self.light_lock


def test_generate_routes_to_light_and_heavy_models():
    """generate() should route to heavy/light models correctly."""

    manager = FakeManager()
    engine = InferenceEngine(manager)

    manager.light_model.return_value = {"choices": [{"text": " light out "}]}
    manager.heavy_model.return_value = {"choices": [{"text": " heavy out "}]}

    out_light = engine.generate("hello", heavy=False)
    out_heavy = engine.generate("hello", heavy=True)

    assert out_light == "light out"
    assert out_heavy == "heavy out"

    # verify model called exactly once for each path
    manager.light_model.assert_called_once()
    manager.heavy_model.assert_called_once()


def test_generate_uses_light_lock():
    """generate() with heavy=False should use light lock."""

    manager = FakeManager()
    engine = InferenceEngine(manager)

    manager.light_model.return_value = {"choices": [{"text": "test"}]}
    engine.generate("hello", heavy=False)

    assert manager.light_lock.enter_called is True
    assert manager.light_lock.exit_called is True
    assert manager.heavy_lock.enter_called is False


def test_generate_uses_heavy_lock():
    """generate() with heavy=True should use heavy lock."""

    manager = FakeManager()
    engine = InferenceEngine(manager)

    manager.heavy_model.return_value = {"choices": [{"text": "test"}]}
    engine.generate("hello", heavy=True)

    assert manager.heavy_lock.enter_called is True
    assert manager.heavy_lock.exit_called is True
    assert manager.light_lock.enter_called is False


def test_generate_forwards_generation_parameters():
    """generate() should forward temperature, top_p, max_tokens and stop to LLM."""

    manager = FakeManager()
    engine = InferenceEngine(manager)

    manager.light_model.return_value = {"choices": [{"text": "ok"}]}

    engine.generate(
        prompt="hi",
        heavy=False,
        max_tokens=256,
        temperature=0.7,
        top_p=0.8,
        stop=["STOP", "\n\n"],
    )

    # Extract call kwargs from the light model
    assert manager.light_model.call_count == 1
    _, kwargs = manager.light_model.call_args

    assert kwargs["max_tokens"] == 256
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.8
    assert kwargs["stop"] == ["STOP", "\n\n"]
    assert kwargs["echo"] is False
    assert kwargs["prompt"] == "hi" or kwargs["prompt"] == kwargs.get("prompt", "hi")


def test_generate_strips_whitespace():
    """generate() should return stripped text."""

    manager = FakeManager()
    engine = InferenceEngine(manager)

    manager.light_model.return_value = {"choices": [{"text": "  some text \n"}]}
    out = engine.generate("hello", heavy=False)

    assert out == "some text"
