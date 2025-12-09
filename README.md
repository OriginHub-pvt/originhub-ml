# OriginHub ML

Machine learning repository for OriginHub. Contains:

- **Agentic System**: Multi-agent reasoning pipeline for idea evaluation and strategic analysis
- **SLM Filter**: Text classification model for filtering startup ideas

## Repository Structure

- `src/agentic/` — Multi-agent pipeline for idea analysis (see [Agentic README](src/agentic/README.md))
- `src/slm_filter/` — Training code for the SLM filter classification model
- `configs/` — YAML configuration files for model training
- `data/` — Sample and DVC-tracked datasets
- `models/` — Model artifacts and checkpoints
- `requirements.txt` — Python dependencies
- `.env.example` — Environment configuration template

## Projects

### 1. Agentic System

A sophisticated multi-agent reasoning pipeline for business idea evaluation with GPU-accelerated LLMs.

**Key Features:**

- Interprets unstructured user input into structured business ideas
- Semantic search using Weaviate vector database
- Intelligent routing to specialized analysis agents
- Competitive analysis and SWOT strategy generation
- GPU-accelerated inference with Qwen models

**Quick Start:**

```bash
# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Run interactive chat (terminal)
python src/agentic/scripts/chat_pipeline.py

# Or start REST API for UI connections
python src/agentic/scripts/api_server.py
# API available at: http://localhost:8000
# Docs: http://localhost:8000/docs

# Or single query
python src/agentic/scripts/run_pipeline.py "Your business idea here"
```

See [src/agentic/README.md](src/agentic/README.md) for detailed documentation and [src/agentic/api/API.md](src/agentic/api/API.md) for API documentation.

### 2. SLM Filter

Text classification model for filtering and categorizing startup ideas.

**Setup:**

1. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Use DVC to pull training data:

```bash
dvc pull
```

3. Configure GCP credentials (optional, for model upload):

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp_credentials.json
```

4. Run the training pipeline:

```bash
python src/slm_filter/train.py --config configs/slm_filter.yaml --log-level INFO
```

The pipeline will:

- Load and clean data from the configured path
- Tokenize inputs and fine-tune the model
- Save the best model and metrics to `models/slm_filter/v{version}`
- Upload artifacts to the configured GCS bucket

## Environment Configuration

Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
```

**Key Configuration Options:**

**Agentic System:**

- `MODEL_7B_PATH` / `MODEL_1B_PATH` — Paths to Qwen GGUF model files
- `MODEL_GPU_LAYERS` — Number of layers to offload to GPU (30 for 8GB VRAM)
- `WEAVIATE_HOST` / `WEAVIATE_PORT` — Weaviate vector database connection
- `RAG_NEW_THRESHOLD` — Similarity threshold for novelty detection (0.0-1.0)
- `AGENTIC_DEBUG` — Set to 1 for verbose logging

**SLM Filter:**

- See `configs/slm_filter.yaml` for training configuration
- `base_model` — HuggingFace model backbone
- `data_path` — Training data CSV path
- `gcs_model_bucket` — GCS bucket for model artifacts

## Data Format

**SLM Filter Training Data:**

Expected CSV columns:

- `title` (string)
- `description` (string)
- `label` (integer)

Rows with missing values are automatically dropped.

## Development

**Prerequisites:**

- Python 3.10+
- NVIDIA GPU with CUDA 12.0+ (for agentic system)
- Docker (for Weaviate)
- Google Cloud credentials (for model upload)

**Testing:**

```bash
# Run agentic system tests
pytest src/agentic/tests/

# Run smoke test (no GPU required)
python src/agentic/scripts/smoke_test_pipeline.py
```

## Troubleshooting

**Agentic System:**

- GPU not detected: Reinstall `llama-cpp-python` with CUDA support
- OOM errors: Reduce `MODEL_GPU_LAYERS` or `MODEL_CONTEXT_SIZE`
- Weaviate connection failed: Check Docker container status

**SLM Filter:**

- GCS authentication errors: Verify `GOOGLE_APPLICATION_CREDENTIALS`
- Data loading fails: Confirm CSV path in `configs/slm_filter.yaml`
- GPU training issues: Ensure CUDA-enabled `transformers` installation

## Documentation

- [Agentic System Documentation](src/agentic/README.md)
- [SLM Filter Details](src/slm_filter/README.md)
