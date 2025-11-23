#!/usr/bin/env python3
"""
Backend-agnostic model factory.
Automatically chooses between llama.cpp and GPU-accelerated transformers
based on MODEL_BACKEND environment variable.
"""

import os
from typing import Union


def create_model_manager(**kwargs) -> Union['ModelManager', 'GPUModelManager']:
    """
    Factory function to create the appropriate model manager based on 
    MODEL_BACKEND environment variable.
    
    Returns
    -------
    Union[ModelManager, GPUModelManager]
        The appropriate model manager instance
    """
    backend = os.getenv("MODEL_BACKEND", "llama_cpp").lower()
    
    if backend == "transformers_gpu":
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
    
    print(f"\n🔧 Model Backend: {backend}")
    
    if backend == "transformers_gpu":
        try:
            import torch
            print(f"🔥 PyTorch: {torch.__version__}")
            print(f"🚀 CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"🎯 GPU Device: {torch.cuda.get_device_name()}")
                print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            heavy_model = os.getenv("GPU_HEAVY_MODEL", "microsoft/DialoGPT-medium")
            light_model = os.getenv("GPU_LIGHT_MODEL", "microsoft/DialoGPT-small")
            print(f"🧠 Heavy Model: {heavy_model}")
            print(f"⚡ Light Model: {light_model}")
            
        except ImportError:
            print("❌ PyTorch not installed! Install with:")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    else:
        print(f"📁 7B Model: {os.getenv('MODEL_7B_PATH', 'Not set')}")
        print(f"📁 1B Model: {os.getenv('MODEL_1B_PATH', 'Not set')}")
        print(f"🔧 GPU Layers: {os.getenv('MODEL_GPU_LAYERS', '20')}")
    
    print()


if __name__ == "__main__":
    get_backend_info()