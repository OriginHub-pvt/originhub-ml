# Model Backend Configuration Guide

This guide explains how to switch between different model backends (local models, GPU transformers, or OpenAI API).

## Quick Start

### Using Local Models (Default)

```bash
# In .env
MODEL_BACKEND=llama_cpp
MODEL_7B_PATH=models/qwen-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
MODEL_1B_PATH=models/qwen-1.5b/model.gguf
MODEL_GPU_LAYERS=30
```

### Using OpenAI API

```bash
# In .env
MODEL_BACKEND=openai
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_HEAVY_MODEL=gpt-4
OPENAI_LIGHT_MODEL=gpt-3.5-turbo
```

### Using GPU Transformers

```bash
# In .env
MODEL_BACKEND=transformers_gpu
GPU_HEAVY_MODEL=microsoft/DialoGPT-medium
GPU_LIGHT_MODEL=microsoft/DialoGPT-small
GPU_MAX_MEMORY=8.0
GPU_TORCH_DTYPE=float16
```

---

## Detailed Configuration

### Option 1: llama.cpp (Local Models - Default)

**Best for:**

- Running offline
- Reducing API costs
- Privacy-conscious deployments
- GPU-accelerated local inference

**Environment Variables:**

```env
MODEL_BACKEND=llama_cpp

# Model file paths
MODEL_7B_PATH=models/qwen-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
MODEL_1B_PATH=models/qwen-1.5b/model.gguf

# Runtime options
MODEL_CONTEXT_SIZE=4096      # Context window size
MODEL_N_THREADS=4            # CPU threads
MODEL_GPU_LAYERS=30          # Layers to offload to GPU (0 = CPU only)

# GPU optimization (NVIDIA)
CUDA_VISIBLE_DEVICES=0       # Which GPU to use
GGML_CUDA_FORCE_CUBLAS=1    # Force CUDA acceleration

# Preloading
MODEL_PRELOAD=true           # Load models at startup
```

**Installation:**

```bash
pip install llama-cpp-python
```

**GPU Requirements:**

- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x
- At least 8GB VRAM recommended

---

### Option 2: OpenAI API

**Best for:**

- Highest quality responses
- No GPU hardware needed
- Latest model capabilities
- Easy scaling

**Environment Variables:**

```env
MODEL_BACKEND=openai

# Required: Your OpenAI API key
OPENAI_API_KEY=sk-your-api-key-here

# Model selection
OPENAI_HEAVY_MODEL=gpt-4              # For complex analysis
OPENAI_LIGHT_MODEL=gpt-3.5-turbo      # For simpler tasks

# Optional alternatives:
# OPENAI_HEAVY_MODEL=gpt-4-turbo
# OPENAI_HEAVY_MODEL=gpt-4o
# OPENAI_LIGHT_MODEL=gpt-3.5-turbo-16k
```

**Installation:**

```bash
pip install openai
```

**Cost Estimation:**

- GPT-3.5-turbo: ~$0.0005-0.0015 per 1K tokens
- GPT-4: ~$0.01-0.03 per 1K tokens
- GPT-4 Turbo: ~$0.01-0.03 per 1K tokens

**How to get API Key:**

1. Sign up at https://platform.openai.com
2. Go to API keys section
3. Create new secret key
4. Add to `.env` file as `OPENAI_API_KEY`

---

### Option 3: GPU Transformers

**Best for:**

- Fine-tuned models
- Custom model training
- Transformers ecosystem models

**Environment Variables:**

```env
MODEL_BACKEND=transformers_gpu

# Model selection
GPU_HEAVY_MODEL=microsoft/DialoGPT-medium
GPU_LIGHT_MODEL=microsoft/DialoGPT-small

# Memory management
GPU_MAX_MEMORY=8.0           # Max GPU memory in GB
GPU_TORCH_DTYPE=float16      # float16 or float32

# PyTorch configuration
CUDA_VISIBLE_DEVICES=0       # Which GPU to use
```

**Installation:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers
```

---

## Code Architecture

The system uses a factory pattern to switch backends seamlessly:

```python
from src.agentic.ml.model_factory import create_model_manager

# Factory automatically creates the right manager based on MODEL_BACKEND
model_manager = create_model_manager()

# All managers implement the same interface
model, lock = model_manager.get(heavy=True)
with lock:
    response = model(
        prompt="Your prompt",
        max_tokens=256,
        temperature=0.7
    )
```

### Manager Implementations:

1. **ModelManager** (`model_manager.py`)

   - Handles llama.cpp models
   - Thread-safe inference

2. **OpenAIModelManager** (`openai_model_manager.py`)

   - Wraps OpenAI API
   - Maintains same interface as ModelManager
   - Handles API errors gracefully

3. **GPUModelManager** (`gpu_model_manager.py`)
   - Manages transformer models on GPU
   - Memory optimization

---

## Switching Between Backends

### To switch from llama.cpp to OpenAI:

1. **Update .env:**

   ```env
   MODEL_BACKEND=openai
   OPENAI_API_KEY=sk-your-api-key-here
   ```

2. **Restart the API:**

   ```bash
   python src/agentic/api/app.py
   ```

3. **No code changes needed!** - The factory handles everything.

### To switch back to llama.cpp:

1. **Update .env:**

   ```env
   MODEL_BACKEND=llama_cpp
   ```

2. **Restart the API** - that's it!

---

## Monitoring Backend

Check which backend is active:

```bash
# View configuration
python -m src.agentic.ml.model_factory

# Output will show:
# Model Backend: openai
# API Key: SET
# Heavy Model: gpt-4
# Light Model: gpt-3.5-turbo
```

---

## Troubleshooting

### OpenAI Backend Issues

**Error: "OPENAI_API_KEY not provided"**

```bash
# Solution: Add to .env
OPENAI_API_KEY=sk-your-key-here
```

**Error: "Invalid API key"**

```bash
# Check:
1. Key is correctly copied from https://platform.openai.com/api-keys
2. Key hasn't been revoked
3. Organization is set correctly (if using org account)
```

**Error: "openai module not found"**

```bash
# Solution: Install openai
pip install openai
```

### Local Model Issues

**Error: "Model file not found"**

```bash
# Check paths exist:
ls models/qwen-7b/
ls models/qwen-1.5b/
```

**Out of Memory errors**

```bash
# Reduce GPU layers:
MODEL_GPU_LAYERS=20  # Instead of 30
```

### GPU Issues

**Error: "CUDA out of memory"**

```bash
# Option 1: Reduce context size
MODEL_CONTEXT_SIZE=2048

# Option 2: Reduce GPU layers
MODEL_GPU_LAYERS=15
```

---

## Performance Comparison

| Backend          | Speed  | Cost | Quality   | Privacy   | Hardware   |
| ---------------- | ------ | ---- | --------- | --------- | ---------- |
| llama.cpp        | Fast\* | Free | Good      | Excellent | GPU needed |
| OpenAI           | Fast   | $$$  | Excellent | Poor\*\*  | None       |
| GPU Transformers | Medium | Free | Good      | Excellent | GPU needed |

\*Depends on GPU; API has lower latency
\*\*Sent to OpenAI servers

---

## Cost Analysis Example

Analyzing 100 business ideas, each with ~500 token input and ~500 token output:

### llama.cpp

- **Cost:** $0 (after initial GPU investment)
- **Hardware:** RTX 4070 (8GB) - ~$400

### OpenAI GPT-3.5

- **Cost:** 100 × (500 + 500) × $0.001 = $100
- **Hardware:** None

### OpenAI GPT-4

- **Cost:** 100 × (500 + 500) × $0.03 = $3,000
- **Hardware:** None

---

## Recommendations

**Use OpenAI if:**

- You need highest quality
- No GPU hardware available
- Small-scale operations (<100 requests/day)
- Willing to pay per-API call

**Use llama.cpp if:**

- Running at scale
- Privacy is important
- Want to save costs long-term
- Have GPU hardware

**Use GPU Transformers if:**

- Customizing models
- Fine-tuning needed
- Using specific architectures
