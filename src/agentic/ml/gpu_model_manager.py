"""
GPU Model Manager
=================

GPU-accelerated model manager using transformers library with CUDA support.
Supports both local and Hugging Face Hub models.
"""

import torch
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class _ThreadLockWrapper:
    """Wrapper around a real threading.Lock to satisfy isinstance tests."""
    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._lock.__exit__(exc_type, exc_val, exc_tb)


class GPUModelManager:
    """
    GPU-accelerated model manager using transformers library.
    Supports CUDA, automatic mixed precision, and efficient memory usage.
    """

    def __init__(
        self,
        heavy_model_name: str = "microsoft/DialoGPT-medium",
        light_model_name: str = "microsoft/DialoGPT-small", 
        device: str = "auto",
        torch_dtype: str = "float16",
        max_memory_gb: float = 8.0,
        preload: bool = True,
    ):
        """
        Parameters
        ----------
        heavy_model_name : str
            Hugging Face model name or local path for heavy model
        light_model_name : str  
            Hugging Face model name or local path for light model
        device : str
            Device to use: "auto", "cuda", "cpu"
        torch_dtype : str
            Precision: "float16", "bfloat16", "float32"
        max_memory_gb : float
            Maximum GPU memory per model in GB
        preload : bool
            Whether to load models immediately
        """
        
        # Thread safety
        self._lock_heavy = _ThreadLockWrapper()
        self._lock_light = _ThreadLockWrapper()
        
        # Configuration
        self._heavy_model_name = heavy_model_name
        self._light_model_name = light_model_name
        self._device = self._get_device(device)
        self._torch_dtype = getattr(torch, torch_dtype)
        self._max_memory = {0: f"{max_memory_gb}GB"} if "cuda" in str(self._device) else None
        
        # Model instances
        self.heavy_model = None
        self.light_model = None
        self.heavy_tokenizer = None
        self.light_tokenizer = None
        
        print(f"[GPUModelManager] Using device: {self._device}, dtype: {torch_dtype}")
        
        if preload:
            self.preload_all()
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")  # Apple Silicon
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _load_model(self, heavy: bool = False) -> None:
        """Load a model with GPU acceleration."""
        import time
        
        model_name = self._heavy_model_name if heavy else self._light_model_name
        model_type = "heavy" if heavy else "light"
        
        print(f"[GPUModelManager] Loading {model_type} model: {model_name}")
        t0 = time.time()
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with GPU acceleration
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self._torch_dtype,
                device_map=self._device,
                max_memory=self._max_memory,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # Store model and tokenizer
            if heavy:
                self.heavy_model = model
                self.heavy_tokenizer = tokenizer
            else:
                self.light_model = model
                self.light_tokenizer = tokenizer
                
        except Exception as e:
            print(f"[GPUModelManager] Error loading {model_type} model: {e}")
            raise
        
        t1 = time.time()
        print(f"[GPUModelManager] Loaded {model_type} model in {t1-t0:.2f}s")
        
        # Print GPU memory usage if CUDA
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"[GPUModelManager] GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
    
    def get(self, heavy: bool = False) -> Tuple[object, object]:
        """
        Get model and tokenizer with thread lock.
        
        Returns
        -------
        tuple
            (model_wrapper, lock)
        """
        if heavy:
            if self.heavy_model is None:
                self._load_model(heavy=True)
            wrapper = ModelWrapper(self.heavy_model, self.heavy_tokenizer, self._device)
            return wrapper, self._lock_heavy
        else:
            if self.light_model is None:
                self._load_model(heavy=False)
            wrapper = ModelWrapper(self.light_model, self.light_tokenizer, self._device)
            return wrapper, self._lock_light
    
    def preload_all(self):
        """Load both heavy and light models."""
        self._load_model(heavy=True)
        self._load_model(heavy=False)


class ModelWrapper:
    """
    Wrapper to make transformers models compatible with llama.cpp interface.
    """
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def __call__(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, 
                 top_p: float = 0.9, stop: Optional[list] = None, **kwargs) -> str:
        """Generate text using the model."""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode only the new tokens
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Handle stop sequences
            if stop:
                for stop_seq in stop:
                    if stop_seq in response:
                        response = response.split(stop_seq)[0]
                        break
            
            return response.strip()
            
        except Exception as e:
            print(f"[ModelWrapper] Generation error: {e}")
            return f"Error: {e}"