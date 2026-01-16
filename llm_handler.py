"""
Unified LLM Handler with multi-backend support.
Supports vLLM (primary), Ollama (fallback), and direct Transformers.
"""
import os
import configparser
import logging
from typing import Optional, Dict, Any
from enum import Enum
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMBackend(Enum):
    """Available LLM backends."""
    VLLM = "vllm"
    OLLAMA = "ollama"
    TRANSFORMERS = "transformers"


class UnifiedLLMHandler:
    """Unified handler for multiple LLM backends with automatic fallback."""
    
    def __init__(self):
        """Initialize LLM handler with configuration."""
        self.config = self._load_config()
        self.backends = self._initialize_backends()
        self.active_backend = self._select_backend()
        self.summarize_prompt = self.config["llm"]["summarize_prompt"]
        
        logger.info(f"Active LLM backend: {self.active_backend.value if self.active_backend else 'None'}")
    
    def _load_config(self) -> configparser.ConfigParser:
        """Load configuration from config.ini file."""
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), "config.ini")
        config.read(config_path)
        return config
    
    def _initialize_backends(self) -> Dict[LLMBackend, Dict[str, Any]]:
        """Initialize all available backends."""
        backends = {}
        
        # vLLM backend (OpenAI-compatible API)
        if self.config.getboolean("vllm", "enabled", fallback=False):
            vllm_config = {
                "url": self.config["vllm"]["url"],
                "model": self.config["vllm"]["model"],
                "available": self._check_vllm_availability(),
            }
            backends[LLMBackend.VLLM] = vllm_config
            logger.info(f"vLLM backend configured: {vllm_config['available']}")
        
        # Ollama backend
        if self.config.getboolean("ollama", "enabled", fallback=True):
            ollama_config = {
                "url": self.config["ollama"]["url"],
                "model": self.config["ollama"]["default_model"],
                "available": self._check_ollama_availability(),
            }
            backends[LLMBackend.OLLAMA] = ollama_config
            logger.info(f"Ollama backend configured: {ollama_config['available']}")
        
        return backends
    
    def _check_vllm_availability(self) -> bool:
        """Check if vLLM server is available."""
        try:
            url = self.config["vllm"]["url"]
            response = requests.get(f"{url}/v1/models", timeout=5)
            if response.status_code == 200:
                logger.info("vLLM server is available")
                return True
        except Exception as e:
            logger.debug(f"vLLM server not available: {str(e)}")
        return False
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama server is available."""
        try:
            from ollama import Client
            client = Client(host=self.config["ollama"]["url"])
            client.list()
            logger.info("Ollama server is available")
            return True
        except Exception as e:
            logger.debug(f"Ollama server not available: {str(e)}")
        return False
    
    def _select_backend(self) -> Optional[LLMBackend]:
        """Select the best available backend based on priority."""
        # Priority: vLLM > Ollama
        priority_order = [LLMBackend.VLLM, LLMBackend.OLLAMA]
        
        for backend in priority_order:
            if backend in self.backends and self.backends[backend]["available"]:
                logger.info(f"Selected backend: {backend.value}")
                return backend
        
        logger.warning("No LLM backend available")
        return None
    
    def is_available(self) -> bool:
        """Return whether any LLM backend is available."""
        return self.active_backend is not None
    
    def get_active_backend(self) -> Optional[str]:
        """Get the name of the active backend."""
        return self.active_backend.value if self.active_backend else None
    
    def _summarize_vllm(self, text: str, model: str = None) -> Optional[str]:
        """Summarize using vLLM (OpenAI-compatible API)."""
        try:
            config = self.backends[LLMBackend.VLLM]
            model = model or config["model"]
            url = config["url"]
            
            logger.info(f"Generating summary using vLLM model: {model}")
            
            response = requests.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.summarize_prompt},
                        {"role": "user", "content": text}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2048,
                },
                timeout=120,
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result["choices"][0]["message"]["content"]
                logger.info(f"vLLM summary generated. Length: {len(summary)} characters")
                return summary
            else:
                logger.error(f"vLLM API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating summary with vLLM: {str(e)}")
            return None
    
    def _summarize_ollama(self, text: str, model: str = None) -> Optional[str]:
        """Summarize using Ollama."""
        try:
            from ollama import Client
            
            config = self.backends[LLMBackend.OLLAMA]
            model = model or config["model"]
            
            logger.info(f"Generating summary using Ollama model: {model}")
            
            client = Client(host=config["url"])
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": self.summarize_prompt},
                    {"role": "user", "content": text},
                ],
            )
            
            summary = response["message"]["content"]
            logger.info(f"Ollama summary generated. Length: {len(summary)} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary with Ollama: {str(e)}")
            return None
    
    def summarize(self, text: str, model: str = None) -> Optional[str]:
        """
        Summarize text using the best available backend with automatic fallback.
        
        Args:
            text: The text to summarize
            model: Optional model override (uses backend default if None)
            
        Returns:
            Summary text or None if all backends fail
        """
        if not self.is_available():
            logger.warning("No LLM backend available for summarization")
            return None
        
        if not text:
            logger.warning("Cannot summarize: Empty text provided")
            return None
        
        logger.info(f"Input text length: {len(text)} characters")
        
        # Try active backend first
        if self.active_backend == LLMBackend.VLLM:
            summary = self._summarize_vllm(text, model)
            if summary:
                return summary
            # Fallback to Ollama if vLLM fails
            logger.warning("vLLM failed, falling back to Ollama")
            if LLMBackend.OLLAMA in self.backends and self.backends[LLMBackend.OLLAMA]["available"]:
                return self._summarize_ollama(text, model)
        
        elif self.active_backend == LLMBackend.OLLAMA:
            return self._summarize_ollama(text, model)
        
        logger.error("All LLM backends failed")
        return None
    
    def get_available_models(self) -> list:
        """Get list of available models from the active backend."""
        if not self.active_backend:
            return []
        
        try:
            if self.active_backend == LLMBackend.VLLM:
                config = self.backends[LLMBackend.VLLM]
                response = requests.get(f"{config['url']}/v1/models", timeout=5)
                if response.status_code == 200:
                    models = response.json()["data"]
                    return [m["id"] for m in models]
            
            elif self.active_backend == LLMBackend.OLLAMA:
                from ollama import Client
                config = self.backends[LLMBackend.OLLAMA]
                client = Client(host=config["url"])
                models = client.list()
                return [model["model"] for model in models["models"]]
        
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
        
        return []
    
    def get_default_model(self) -> Optional[str]:
        """Get the default model for the active backend."""
        if not self.active_backend:
            return None
        
        config = self.backends.get(self.active_backend)
        return config["model"] if config else None
