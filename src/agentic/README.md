# Agentic module

This folder implements the OriginHub agentic multi-step reasoning pipeline.

Quick points:

- Use `scripts/run_pipeline.py` to run a single-shot pipeline from the CLI.
- Use `scripts/chat_pipeline.py` for an interactive REPL-like chat.
- Models are loaded from environment variables `MODEL_7B_PATH` and `MODEL_1B_PATH` (defaults are set in the scripts).
- The pipeline expects a Weaviate instance by default; configure `WEAVIATE_HOST`, `WEAVIATE_PORT`, and `WEAVIATE_COLLECTION` as needed.

Running a smoke test (no heavy models required):

```bash
python src/agentic/scripts/smoke_test_pipeline.py
```

See the code in `src/agentic` for more detail about agents, prompts and the state object.
