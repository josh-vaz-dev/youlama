import requests
from typing import Optional
import configparser
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config() -> configparser.ConfigParser:
    """Load configuration from config.ini file."""
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), "config.ini")
    config.read(config_path)
    return config


config = load_config()


class OllamaHandler:
    def __init__(self):
        self.enabled = config["ollama"].getboolean("enabled")
        self.url = config["ollama"]["url"]
        self.default_model = config["ollama"]["default_model"]
        self.prompt = config["ollama"]["summarize_prompt"]
        logger.info(
            f"Initialized Ollama handler with URL: {self.url}, Default model: {self.default_model}"
        )
        logger.info(f"Ollama enabled: {self.enabled}")

    def is_available(self) -> bool:
        """Check if Ollama is available and enabled."""
        if not self.enabled:
            logger.info("Ollama is disabled in config")
            return False
        try:
            logger.info(f"Checking Ollama availability at {self.url}")
            response = requests.get(f"{self.url}/api/tags")
            available = response.status_code == 200
            logger.info(
                f"Ollama server response: {'available' if available else 'unavailable'}"
            )
            return available
        except Exception as e:
            logger.error(f"Error checking Ollama availability: {str(e)}")
            return False

    def get_available_models(self) -> list:
        """Get list of available Ollama models."""
        try:
            logger.info("Fetching available Ollama models")
            response = requests.get(f"{self.url}/api/tags")
            if response.status_code == 200:
                models = [model["name"] for model in response.json()["models"]]
                logger.info(
                    f"Found {len(models)} available models: {', '.join(models)}"
                )
                return models
            logger.warning(
                f"Failed to fetch models. Status code: {response.status_code}"
            )
            return []
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {str(e)}")
            return []

    def validate_model(self, model_name: str) -> tuple[bool, Optional[str]]:
        """Validate if a model exists and return the first available model if not."""
        available_models = self.get_available_models()
        if not available_models:
            return False, None

        if model_name in available_models:
            return True, model_name

        logger.warning(
            f"Model {model_name} not found in available models. Using first available model: {available_models[0]}"
        )
        return True, available_models[0]

    def get_default_model(self) -> Optional[str]:
        """Get the default model, falling back to first available if default is not found."""
        if not self.is_available():
            return None

        available_models = self.get_available_models()
        if not available_models:
            return None

        if self.default_model in available_models:
            logger.info(f"Using configured default model: {self.default_model}")
            return self.default_model

        logger.warning(
            f"Configured model '{self.default_model}' not found in available models. Using first available model: {available_models[0]}"
        )
        return available_models[0]

    def summarize(self, text: str, model: Optional[str] = None) -> Optional[str]:
        """Summarize text using Ollama."""
        if not self.is_available():
            logger.warning("Attempted to summarize with Ollama unavailable")
            return None

        # Validate and get the correct model
        is_valid, valid_model = self.validate_model(model or self.default_model)
        if not is_valid:
            logger.error("No valid Ollama models available")
            return None

        prompt = f"{self.prompt}\n\n{text}"
        logger.info(f"Generating summary using model: {valid_model}")
        logger.info(f"Input text length: {len(text)} characters")

        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={"model": valid_model, "prompt": prompt, "stream": False},
            )

            if response.status_code == 200:
                summary = response.json()["response"]
                logger.info(
                    f"Successfully generated summary of length: {len(summary)} characters"
                )
                return summary
            logger.error(
                f"Failed to generate summary. Status code: {response.status_code}"
            )
            return None
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            return None
