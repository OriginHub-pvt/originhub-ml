#!/usr/bin/env python3
"""
Backend-agnostic model factory.
Automatically chooses between llama.cpp, GPU transformers, and OpenAI API
based on MODEL_BACKEND environment variable.
"""

import os
from typing import Union


def create_model_manager(**kwargs) -> Union['ModelManager', 'GPUModelManager', 'OpenAIModelManager']:
    """
    Factory function to create the appropriate model manager based on 
    MODEL_BACKEND environment variable.
    
    Returns
    -------
    Union[ModelManager, GPUModelManager, OpenAIModelManager]
        The appropriate model manager instance
    """
    backend = os.getenv("MODEL_BACKEND", "llama_cpp").lower()
    
    if backend == "openai":
        from .openai_model_manager import OpenAIModelManager
        
        # Extract OpenAI settings from environment
        api_key = os.getenv("OPENAI_API_KEY")
        heavy_model = os.getenv("OPENAI_HEAVY_MODEL", "gpt-4")
        light_model = os.getenv("OPENAI_LIGHT_MODEL", "gpt-3.5-turbo")
        
        if not api_key:
            raise ValueError(
                "MODEL_BACKEND=openai but OPENAI_API_KEY not set in environment. "
                "Please set OPENAI_API_KEY to your OpenAI API key."
            )
        
        print(f"[ModelFactory] Using OpenAI backend with {heavy_model} (heavy), {light_model} (light)")
        
        return OpenAIModelManager(
            api_key=api_key,
            heavy_model=heavy_model,
            light_model=light_model,
        )
    
    elif backend == "transformers_gpu":
        from .gpu_model_manager import GPUModelManager
        
        # Extract GPU-specific settings from environment
        heavy_model = os.getenv("GPU_HEAVY_MODEL", "microsoft/DialoGPT-medium")
        light_model = os.getenv("GPU_LIGHT_MODEL", "microsoft/DialoGPT-small")
        max_memory = float(os.getenv("GPU_MAX_MEMORY", "8.0"))
        torch_dtype = os.getenv("GPU_TORCH_DTYPE", "float16")
        
        print(f"[ModelFactory] Using GPU backend with {heavy_model} (heavy), {light_model} (light)")
        
        return GPUModelManager(
            heavy_model_name=heavy_model,
            light_model_name=light_model,
            max_memory_gb=max_memory,
            torch_dtype=torch_dtype,
            preload=kwargs.get('preload', True),
        )
    
    else:  # Default to llama_cpp
        from .model_manager import ModelManager
        
        print(f"[ModelFactory] Using llama.cpp backend")
        
        return ModelManager(
            model7b_path=kwargs.get('model7b_path', os.getenv("MODEL_7B_PATH")),
            model1b_path=kwargs.get('model1b_path', os.getenv("MODEL_1B_PATH")),
            context_size=kwargs.get('context_size', int(os.getenv("MODEL_CONTEXT_SIZE", 4096))),
            n_threads=kwargs.get('n_threads', int(os.getenv("MODEL_N_THREADS", 4))),
            gpu_layers=kwargs.get('gpu_layers', int(os.getenv("MODEL_GPU_LAYERS", 20))),
            preload=kwargs.get('preload', True),
        )


def get_backend_info():
    """Print information about the current backend configuration."""
    backend = os.getenv("MODEL_BACKEND", "llama_cpp").lower()
    
    print(f"\nModel Backend: {backend}")
    
    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "NOT SET")
        heavy_model = os.getenv("OPENAI_HEAVY_MODEL", "gpt-4")
        light_model = os.getenv("OPENAI_LIGHT_MODEL", "gpt-3.5-turbo")
        
        api_status = "SET" if api_key != "NOT SET" else "NOT SET"
        print(f"API Key: {api_status}")
        print(f"Heavy Model: {heavy_model}")
        print(f"Light Model: {light_model}")
    
    elif backend == "transformers_gpu":
        try:
            import torch
            print(f"PyTorch: {torch.__version__}")
            print(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU Device: {torch.cuda.get_device_name()}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            heavy_model = os.getenv("GPU_HEAVY_MODEL", "microsoft/DialoGPT-medium")
            light_model = os.getenv("GPU_LIGHT_MODEL", "microsoft/DialoGPT-small")
            print(f"Heavy Model: {heavy_model}")
            print(f"Light Model: {light_model}")
            
        except ImportError:
            print("PyTorch not installed! Install with:")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    else:
        print(f"7B Model: {os.getenv('MODEL_7B_PATH', 'Not set')}")
        print(f"1B Model: {os.getenv('MODEL_1B_PATH', 'Not set')}")
        print(f"GPU Layers: {os.getenv('MODEL_GPU_LAYERS', '20')}")
    
    print()


if __name__ == "__main__":
    get_backend_info()