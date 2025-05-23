import requests
from typing import Optional
import configparser
import os


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

    def is_available(self) -> bool:
        """Check if Ollama is available and enabled."""
        if not self.enabled:
            return False
        try:
            response = requests.get(f"{self.url}/api/tags")
            return response.status_code == 200
        except:
            return False

    def get_available_models(self) -> list:
        """Get list of available Ollama models."""
        try:
            response = requests.get(f"{self.url}/api/tags")
            if response.status_code == 200:
                return [model["name"] for model in response.json()["models"]]
            return []
        except:
            return []

    def summarize(self, text: str, model: Optional[str] = None) -> Optional[str]:
        """Summarize text using Ollama."""
        if not self.is_available():
            return None

        model = model or self.default_model
        prompt = f"{self.prompt}\n\n{text}"

        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )

            if response.status_code == 200:
                return response.json()["response"]
            return None
        except Exception as e:
            print(f"Error summarizing text: {str(e)}")
            return None
