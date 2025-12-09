# OriginHub Agentic System

A sophisticated multi-agent reasoning pipeline for idea evaluation, competitive analysis, and strategic planning. The system leverages GPU-accelerated LLMs and semantic search to provide intelligent insights about business ideas and market opportunities.

## Overview

The OriginHub Agentic System implements an intelligent pipeline that processes user ideas through multiple specialized agents, each performing specific reasoning tasks. The system can:

- **Interpret** unstructured user input into structured business ideas
- **Search** existing knowledge bases for similar ideas using semantic retrieval
- **Evaluate** novelty and determine optimal analysis paths
- **Review** existing ideas against competitive landscape
- **Strategize** for new ideas with SWOT analysis and market positioning
- **Clarify** ambiguous inputs through intelligent questioning
- **Summarize** findings into actionable insights

## Architecture

### Pipeline Flow

```
User Input
    ↓
┌───────────────────────────────────────────────────────────┐
│  Interpreter Agent                                        │
│  • Extracts structured data (title, description, etc.)   │
└───────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────┐
│  RAG Agent                                                │
│  • Semantic search via Weaviate vector database          │
│  • Retrieves similar existing ideas                      │
└───────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────┐
│  Evaluator Agent                                          │
│  • Determines: New idea? Need clarification?             │
│  • Routes to appropriate analysis path                   │
└───────────────────────────────────────────────────────────┘
    ↓
    ├─ Need Clarification? ──→ Clarifier Agent ──┐
    │                                             │
    ├─ New Idea? ──→ Strategist Agent ───────────┤
    │                                             │
    └─ Existing Idea? ──→ Reviewer Agent ────────┤
                                                  ↓
                              ┌───────────────────────────────┐
                              │  Summarizer Agent             │
                              │  • Final report generation    │
                              └───────────────────────────────┘
```

### Agent Descriptions

#### 1. **Interpreter Agent**

- **Purpose**: Converts free-form text into structured JSON
- **Model**: Qwen 1.5B (lightweight, fast)
- **Output**:
  ```json
  {
    "title": "Business Idea Title",
    "description": "Detailed description",
    "target_audience": "Primary users",
    "problem_statement": "Problem being solved",
    "proposed_solution": "How it works",
    "tags": ["tag1", "tag2"]
  }
  ```

#### 2. **RAG Agent**

- **Purpose**: Semantic similarity search
- **Technology**: Weaviate vector database
- **Features**:
  - Threshold-based novelty detection
  - Distance-based similarity scoring
  - Configurable via `RAG_NEW_THRESHOLD` environment variable
- **No LLM**: Pure retrieval operation

#### 3. **Evaluator Agent**

- **Purpose**: Decision routing logic
- **Model**: Qwen 1.5B (lightweight decision-making)
- **Decisions**:
  - `need_more_clarification`: Route to Clarifier
  - `is_new_idea`: Route to Strategist
  - Otherwise: Route to Reviewer

#### 4. **Clarifier Agent**

- **Purpose**: Generate intelligent follow-up questions
- **Model**: Qwen 1.5B
- **Behavior**: Loops back to Interpreter until sufficient clarity
- **Use Cases**: Vague inputs, missing critical information

#### 5. **Reviewer Agent**

- **Purpose**: Competitive analysis for existing ideas
- **Model**: Qwen 7B (complex reasoning)
- **Analysis Includes**:
  - Existing solutions comparison
  - Competitive differentiation
  - Market positioning insights
  - Strength/weakness assessment

#### 6. **Strategist Agent**

- **Purpose**: Strategic planning for novel ideas
- **Model**: Qwen 7B (complex reasoning)
- **Analysis Includes**:
  - SWOT analysis (Strengths, Weaknesses, Opportunities, Threats)
  - Market opportunity assessment
  - Implementation recommendations
  - Risk analysis

#### 7. **Summarizer Agent**

- **Purpose**: Final report generation
- **Model**: Qwen 7B
- **Output**: Executive summary with actionable insights

## Quick Start

### Prerequisites

1. **Hardware Requirements**:

   - NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
   - 16GB+ system RAM
   - CUDA 12.0+ installed

2. **Software Requirements**:
   - Python 3.10+
   - Conda/Miniconda
   - Docker (for Weaviate)

### Installation

1. **Clone and Setup Environment**:

```bash
cd originhub-ml
conda create -n oh_agentic python=3.10
conda activate oh_agentic
```

2. **Install GPU-Accelerated Dependencies**:

```bash
# Install llama-cpp-python with CUDA support
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# Install other dependencies
pip install -r src/agentic/requirements.txt
```

3. **Download Models**:

```bash
# Create models directory
mkdir -p models/qwen-7b models/qwen-1.5b

# Download Qwen 2.5 7B Instruct (Q4_K_M quantized)
# Place in: models/qwen-7b/

# Download Qwen 1.5B Instruct
# Place in: models/qwen-1.5b/model.gguf
```

4. **Configure Environment**:

```bash
cp .env.example .env
# Edit .env with your settings
```

### Configuration

Create a `.env` file in the project root:

```bash
# Weaviate Configuration
WEAVIATE_HOST="localhost"
WEAVIATE_PORT=8081
WEAVIATE_GRPC_PORT=50051
WEAVIATE_MODEL="sentence-transformers/all-MiniLM-L6-v2"
WEAVIATE_COLLECTION="ArticleSummary"

# Model Paths (GGUF format)
MODEL_7B_PATH="models/qwen-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
MODEL_1B_PATH="models/qwen-1.5b/model.gguf"

# GPU Acceleration Settings
MODEL_GPU_LAYERS=30              # Number of layers to offload to GPU
MODEL_CONTEXT_SIZE=4096          # Context window size
MODEL_N_THREADS=4                # CPU threads for non-GPU operations
CUDA_VISIBLE_DEVICES=0           # GPU device ID
GGML_CUDA_FORCE_CUBLAS=1        # Force CUDA acceleration

# Model Backend
MODEL_BACKEND=llama_cpp          # Options: llama_cpp, transformers_gpu

# Preloading
MODEL_PRELOAD=true               # Load models at startup

# Debugging
AGENTIC_DEBUG=0                  # Set to 1 for verbose logging

# RAG Settings
RAG_NEW_THRESHOLD=0.8            # Similarity threshold for novelty detection
```

### Starting Weaviate

```bash
# Using Docker Compose
docker-compose up -d weaviate

# Or standalone Docker
docker run -d \
  -p 8081:8080 \
  -p 50051:50051 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  -e DEFAULT_VECTORIZER_MODULE='text2vec-transformers' \
  -e ENABLE_MODULES='text2vec-transformers' \
  -e TRANSFORMERS_INFERENCE_API='http://t2v-transformers:8080' \
  semitechnologies/weaviate:latest
```

## Usage

### 1. Command-Line Single Query

```bash
# Set Python path
export PYTHONPATH=/path/to/originhub-ml

# Run pipeline with a query
python src/agentic/scripts/run_pipeline.py "Build an AI-powered task management app"

# With debug mode
python src/agentic/scripts/run_pipeline.py "Your idea here" --debug
```

### 2. Interactive Chat Mode

```bash
python src/agentic/scripts/chat_pipeline.py
```

Example session:

```
OriginHub Agentic Chat
Type 'exit' or 'quit' to end session.

You: I want to create a platform for freelance designers
Processing...

Analysis: This is an existing market space...
[Competitive analysis follows]

You: What about adding AI-powered design suggestions?
Processing...

Strategic Insight: Novel feature combination...
[SWOT analysis follows]
```

### 3. Smoke Test (No GPU Required)

```bash
python src/agentic/scripts/smoke_test_pipeline.py
```

### 4. Programmatic Usage

```python
from src.agentic.pipeline.pipeline_runner import PipelineRunner
from src.agentic.ml.model_manager import ModelManager
from src.agentic.ml.inference_engine import InferenceEngine
from src.agentic.prompts.prompt_builder import PromptBuilder
from src.agentic.agents import *

# Initialize components
model_manager = ModelManager()
inference = InferenceEngine(model_manager)
prompts = PromptBuilder()

# Initialize agents
interpreter = InterpreterAgent(inference, prompts)
rag = RAGAgent(vector_db=retriever)
evaluator = EvaluatorAgent(inference, prompts)
clarifier = ClarifierAgent(inference, prompts)
reviewer = ReviewerAgent(inference, prompts)
strategist = StrategistAgent(inference, prompts)
summarizer = SummarizerAgent(inference, prompts)

# Create pipeline
pipeline = PipelineRunner(
    interpreter=interpreter,
    clarifier=clarifier,
    rag=rag,
    evaluator=evaluator,
    reviewer=reviewer,
    strategist=strategist,
    summarizer=summarizer
)

# Execute
result = pipeline.run("Your business idea here")
print(result.summary)
```

## Testing

### Unit Tests

```bash
# Run all tests
pytest src/agentic/tests/

# Run specific test file
pytest src/agentic/tests/pipeline/test_pipeline_runner.py

# Run with coverage
pytest --cov=src/agentic src/agentic/tests/
```

### Integration Tests

```bash
# Test full pipeline
pytest src/agentic/tests/integration/

# Test with real models (requires GPU)
pytest src/agentic/tests/integration/ --use-real-models
```

## Advanced Configuration

### GPU Optimization

**For 8GB VRAM (RTX 4070)**:

```bash
MODEL_GPU_LAYERS=30              # Full offload for 7B model
MODEL_CONTEXT_SIZE=4096
```

**For 6GB VRAM (RTX 3060)**:

```bash
MODEL_GPU_LAYERS=20              # Partial offload
MODEL_CONTEXT_SIZE=2048
```

**For 4GB VRAM (GTX 1650)**:

```bash
MODEL_GPU_LAYERS=10              # Minimal offload
MODEL_CONTEXT_SIZE=1024
```

### Model Backend Selection

**Option 1: llama-cpp-python (Recommended)**

- Better GGUF model support
- Efficient memory usage
- Faster for quantized models

```bash
MODEL_BACKEND=llama_cpp
```

**Option 2: Transformers + CUDA**

- Native HuggingFace integration
- Better for fine-tuning workflows

```bash
MODEL_BACKEND=transformers_gpu
GPU_HEAVY_MODEL=Qwen/Qwen2.5-7B-Instruct
GPU_LIGHT_MODEL=Qwen/Qwen2-1.5B-Instruct
GPU_TORCH_DTYPE=float16
```

### RAG Tuning

Adjust novelty detection sensitivity:

```bash
# Stricter (more ideas considered "new")
RAG_NEW_THRESHOLD=0.9

# Moderate (default)
RAG_NEW_THRESHOLD=0.8

# Lenient (fewer ideas considered "new")
RAG_NEW_THRESHOLD=0.7
```

### Debug Mode

Enable detailed logging:

```bash
AGENTIC_DEBUG=1
```

Outputs include:

- Agent execution traces
- Model inference timing
- State transitions
- RAG retrieval details

## Project Structure

```
src/agentic/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── __init__.py
│
├── agents/                        # Agent implementations
│   ├── interpreter_agent.py      # Input structuring
│   ├── rag_agent.py              # Semantic retrieval
│   ├── evaluator_agent.py        # Decision routing
│   ├── clarifier_agent.py        # Question generation
│   ├── reviewer_agent.py         # Competitive analysis
│   ├── strategist_agent.py       # Strategic planning
│   └── summarizer_agent.py       # Report generation
│
├── pipeline/                      # Orchestration
│   ├── pipeline_runner.py        # Main pipeline logic
│   └── interactive_pipeline_runner.py  # Chat interface
│
├── ml/                            # Model management
│   ├── model_manager.py          # Model loading/caching
│   ├── inference_engine.py       # LLM inference wrapper
│   ├── gpu_model_manager.py      # GPU-specific manager
│   └── model_factory.py          # Backend abstraction
│
├── rag/                           # Vector database
│   └── weaviate_retriever.py     # Weaviate client
│
├── prompts/                       # Prompt engineering
│   ├── prompt_builder.py         # Template management
│   └── templates/                # Agent-specific prompts
│
├── core/                          # Shared utilities
│   ├── state.py                  # State object definition
│   └── base_agent.py             # Agent interface
│
├── scripts/                       # Entry points
│   ├── run_pipeline.py           # CLI runner
│   ├── chat_pipeline.py          # Interactive chat
│   └── smoke_test_pipeline.py    # Quick validation
│
├── tests/                         # Test suite
│   ├── agents/                   # Agent tests
│   ├── pipeline/                 # Pipeline tests
│   └── integration/              # E2E tests
│
├── adk/                           # Google ADK integration
│   ├── graph.py                  # ADK graph definition
│   ├── root_agent.py             # Root agent wrapper
│   └── wrapped_agents.py         # ADK-compatible wrappers
│
└── utils/                         # Helper functions
    ├── logger.py                 # Logging utilities
    └── validators.py             # Input validation
```

## Key Components Deep Dive

### State Object

The `State` class is the shared memory between agents:

```python
class State:
    input_text: str                      # Original user input
    interpreted: dict                    # Structured idea JSON
    clarifications: List[str]            # Generated questions
    rag_results: List[dict]              # Retrieved matches
    analysis: str                        # Reviewer output
    strategy: str                        # Strategist output
    summary: str                         # Final report
    is_new_idea: bool                    # Novelty flag
    need_more_clarification: bool        # Clarification flag
    agent_outputs: dict                  # Individual agent outputs
    debug_trace: List[str]               # Execution trace
```

### Model Manager

Handles model lifecycle:

- **Lazy loading**: Models loaded on first use
- **Caching**: In-memory model reuse
- **GPU management**: Automatic layer distribution
- **Error recovery**: Graceful fallback on failure

```python
manager = ModelManager()
model_7b = manager.get_model_7b()      # Heavy reasoning
model_1_5b = manager.get_model_1_5b()  # Fast decisions
```

### Inference Engine

Unified interface for LLM calls:

```python
inference = InferenceEngine(model_manager)

result = inference.generate(
    model="7b",                    # or "1.5b"
    prompt="Your prompt here",
    temperature=0.7,
    max_tokens=512,
    stop_sequences=["</json>"]
)
```

### Prompt Builder

Manages prompt templates:

```python
prompts = PromptBuilder()

prompt = prompts.build(
    agent_name="interpreter",
    user_input="Create a food delivery app",
    context={"previous_clarifications": [...]}
)
```

## Troubleshooting

### GPU Not Detected

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify llama-cpp-python CUDA support
python -c "import llama_cpp; print(llama_cpp.__version__)"

# Reinstall with CUDA
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall
```

### Out of Memory (OOM)

Reduce GPU layers or context size:

```bash
MODEL_GPU_LAYERS=20        # Reduce from 30
MODEL_CONTEXT_SIZE=2048    # Reduce from 4096
```

### Weaviate Connection Failed

```bash
# Check Weaviate status
docker ps | grep weaviate

# Check logs
docker logs <weaviate-container-id>

# Restart Weaviate
docker-compose restart weaviate
```

### Model Loading Slow

Enable model preloading:

```bash
MODEL_PRELOAD=true         # Load at startup
```

Or use background loading:

```bash
MODEL_PRELOAD=false
MODEL_PRELOAD_BACKGROUND=true
```

### Low Inference Speed

1. **Verify GPU usage**: Check `nvidia-smi` during inference
2. **Increase GPU layers**: Set `MODEL_GPU_LAYERS=30`
3. **Enable CUBLAS**: Set `GGML_CUDA_FORCE_CUBLAS=1`
4. **Use quantized models**: Q4_K_M provides best speed/quality balance

## Performance Benchmarks

**Hardware**: RTX 4070 Laptop (8GB VRAM)

| Agent       | Model | GPU Layers | Tokens/sec | Latency |
| ----------- | ----- | ---------- | ---------- | ------- |
| Interpreter | 1.5B  | 20         | 64 tok/s   | ~1s     |
| Evaluator   | 1.5B  | 20         | 64 tok/s   | ~1s     |
| Reviewer    | 7B    | 30         | 41 tok/s   | ~3s     |
| Strategist  | 7B    | 30         | 41 tok/s   | ~3s     |
| Summarizer  | 7B    | 30         | 41 tok/s   | ~3s     |

**Full Pipeline**: 10-15 seconds for typical input

## Contributing

### Adding New Agents

1. Create agent file in `src/agentic/agents/`
2. Inherit from `AgentBase` or implement `run(state) -> state`
3. Add prompt template in `src/agentic/prompts/templates/`
4. Register in `PipelineRunner`
5. Add unit tests in `src/agentic/tests/agents/`

Example:

```python
class MyCustomAgent:
    def __init__(self, inference, prompts, name="MyAgent"):
        self.inference = inference
        self.prompts = prompts
        self.name = name

    def run(self, state: State) -> State:
        # Your logic here
        prompt = self.prompts.build("my_agent", state.input_text)
        response = self.inference.generate("7b", prompt)
        state.custom_output = response
        return state
```

### Code Style

- Follow PEP 8
- Type hints required
- Docstrings in NumPy style
- Run `black` formatter before committing

## References

- **Qwen Models**: [Qwen2.5 Documentation](https://github.com/QwenLM/Qwen2.5)
- **llama.cpp**: [GitHub Repository](https://github.com/ggerganov/llama.cpp)
- **Weaviate**: [Official Docs](https://weaviate.io/developers/weaviate)
- **Google ADK**: [Agent Development Kit](https://github.com/google/adk)

## License

[Your License Here]

## Support

For issues, questions, or contributions:

- **Issues**: [GitHub Issues](https://github.com/OriginHub-pvt/originhub-ml/issues)
- **Discussions**: [GitHub Discussions](https://github.com/OriginHub-pvt/originhub-ml/discussions)

---

**Built by the OriginHub Team**
