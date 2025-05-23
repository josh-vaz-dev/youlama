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

    def summarize(self, text: str, model: Optional[str] = None) -> Optional[str]:
        """Summarize text using Ollama."""
        if not self.is_available():
            logger.warning("Attempted to summarize with Ollama unavailable")
            return None

        model = model or self.default_model
        prompt = f"{self.prompt}\n\n{text}"
        logger.info(f"Generating summary using model: {model}")
        logger.info(f"Input text length: {len(text)} characters")

        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
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
