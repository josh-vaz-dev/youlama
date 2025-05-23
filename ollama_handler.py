import os
import configparser
import logging
from typing import List, Optional
from ollama import Client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OllamaHandler:
    def __init__(self):
        """Initialize Ollama handler with configuration."""
        self.config = self._load_config()
        self.endpoint = self.config["ollama"]["url"]
        self.default_model = self.config["ollama"]["default_model"]
        self.summarize_prompt = self.config["ollama"]["summarize_prompt"]
        self.client = Client(host=self.endpoint)
        self.available = self._check_availability()
        logger.info(f"Initialized Ollama handler with endpoint: {self.endpoint}")
        logger.info(f"Default model: {self.default_model}")
        logger.info(f"Ollama available: {self.available}")

    def _load_config(self) -> configparser.ConfigParser:
        """Load configuration from config.ini file."""
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), "config.ini")
        config.read(config_path)
        return config

    def _check_availability(self) -> bool:
        """Check if Ollama server is available."""
        try:
            self.client.list()
            logger.info("Ollama server is available")
            return True
        except Exception as e:
            logger.warning(f"Ollama server is not available: {str(e)}")
            return False

    def is_available(self) -> bool:
        """Return whether Ollama is available."""
        return self.available

    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            models = self.client.list()
            # The response structure is different, models are directly in the response
            model_names = [model["model"] for model in models["models"]]
            logger.info(f"Found {len(model_names)} available models")
            return model_names
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []

    def get_default_model(self) -> str:
        """Get the default model, falling back to first available if configured model not found."""
        if not self.available:
            return None

        available_models = self.get_available_models()
        if not available_models:
            return None

        if self.default_model in available_models:
            logger.info(f"Using configured default model: {self.default_model}")
            return self.default_model
        else:
            logger.warning(
                f"Configured model '{self.default_model}' not found, using first available model: {available_models[0]}"
            )
            return available_models[0]

    def summarize(self, text: str, model: str = None) -> Optional[str]:
        """Summarize text using Ollama."""
        if not self.available:
            logger.warning("Cannot summarize: Ollama is not available")
            return None

        if not text:
            logger.warning("Cannot summarize: Empty text provided")
            return None

        model = model or self.default_model
        if not model:
            logger.warning("Cannot summarize: No model specified")
            return None

        try:
            logger.info(f"Generating summary using model: {model}")
            logger.info(f"Input text length: {len(text)} characters")

            # Generate the summary using the prompt from config
            response = self.client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": self.summarize_prompt},
                    {"role": "user", "content": text},
                ],
            )

            summary = response["message"]["content"]
            logger.info(f"Summary generated. Length: {len(summary)} characters")
            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return None
