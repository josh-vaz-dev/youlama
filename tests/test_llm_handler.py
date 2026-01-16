"""Unit tests for llm_handler module."""
import pytest
from unittest.mock import patch, MagicMock, Mock
import configparser
import os
from llm_handler import UnifiedLLMHandler, LLMBackend


class TestLLMHandlerInitialization:
    """Tests for LLM handler initialization."""
    
    @patch('llm_handler.UnifiedLLMHandler._load_config')
    @patch('llm_handler.UnifiedLLMHandler._check_vllm_availability')
    @patch('llm_handler.UnifiedLLMHandler._check_ollama_availability')
    def test_init_with_vllm_available(self, mock_ollama, mock_vllm, mock_config):
        """Test initialization with vLLM available."""
        # Setup config
        config = configparser.ConfigParser()
        config.read_string("""
[llm]
summarize_prompt = Test prompt

[vllm]
enabled = true
url = http://vllm:8000
model = test-model

[ollama]
enabled = true
url = http://localhost:11434
default_model = llama3.1:8b
""")
        mock_config.return_value = config
        mock_vllm.return_value = True
        mock_ollama.return_value = True
        
        handler = UnifiedLLMHandler()
        assert handler.is_available()
        assert handler.active_backend == LLMBackend.VLLM
    
    @patch('llm_handler.UnifiedLLMHandler._load_config')
    @patch('llm_handler.UnifiedLLMHandler._check_vllm_availability')
    @patch('llm_handler.UnifiedLLMHandler._check_ollama_availability')
    def test_init_with_only_ollama(self, mock_ollama, mock_vllm, mock_config):
        """Test initialization with only Ollama available."""
        config = configparser.ConfigParser()
        config.read_string("""
[llm]
summarize_prompt = Test prompt

[vllm]
enabled = true
url = http://vllm:8000
model = test-model

[ollama]
enabled = true
url = http://localhost:11434
default_model = llama3.1:8b
""")
        mock_config.return_value = config
        mock_vllm.return_value = False
        mock_ollama.return_value = True
        
        handler = UnifiedLLMHandler()
        assert handler.is_available()
        assert handler.active_backend == LLMBackend.OLLAMA
    
    @patch('llm_handler.UnifiedLLMHandler._load_config')
    @patch('llm_handler.UnifiedLLMHandler._check_vllm_availability')
    @patch('llm_handler.UnifiedLLMHandler._check_ollama_availability')
    def test_init_with_no_backends(self, mock_ollama, mock_vllm, mock_config):
        """Test initialization with no backends available."""
        config = configparser.ConfigParser()
        config.read_string("""
[llm]
summarize_prompt = Test prompt

[vllm]
enabled = true
url = http://vllm:8000
model = test-model

[ollama]
enabled = true
url = http://localhost:11434
default_model = llama3.1:8b
""")
        mock_config.return_value = config
        mock_vllm.return_value = False
        mock_ollama.return_value = False
        
        handler = UnifiedLLMHandler()
        assert not handler.is_available()
        assert handler.active_backend is None


class TestVLLMSummarization:
    """Tests for vLLM summarization."""
    
    @patch('llm_handler.UnifiedLLMHandler._load_config')
    @patch('llm_handler.UnifiedLLMHandler._check_vllm_availability')
    @patch('llm_handler.UnifiedLLMHandler._check_ollama_availability')
    @patch('llm_handler.requests.post')
    def test_vllm_summarize_success(self, mock_post, mock_ollama, mock_vllm, mock_config, mock_vllm_response):
        """Test successful summarization with vLLM."""
        # Setup config
        config = configparser.ConfigParser()
        config.read_string("""
[llm]
summarize_prompt = Test prompt

[vllm]
enabled = true
url = http://vllm:8000
model = test-model

[ollama]
enabled = false
url = http://localhost:11434
default_model = llama3.1:8b
""")
        mock_config.return_value = config
        mock_vllm.return_value = True
        mock_ollama.return_value = False
        
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_vllm_response
        mock_post.return_value = mock_response
        
        handler = UnifiedLLMHandler()
        summary = handler.summarize("Test text to summarize")
        
        assert summary == "This is a test summary from vLLM."
        mock_post.assert_called_once()
    
    @patch('llm_handler.UnifiedLLMHandler._load_config')
    @patch('llm_handler.UnifiedLLMHandler._check_vllm_availability')
    @patch('llm_handler.UnifiedLLMHandler._check_ollama_availability')
    @patch('llm_handler.requests.post')
    def test_vllm_summarize_failure(self, mock_post, mock_ollama, mock_vllm, mock_config):
        """Test vLLM summarization failure."""
        config = configparser.ConfigParser()
        config.read_string("""
[llm]
summarize_prompt = Test prompt

[vllm]
enabled = true
url = http://vllm:8000
model = test-model

[ollama]
enabled = false
url = http://localhost:11434
default_model = llama3.1:8b
""")
        mock_config.return_value = config
        mock_vllm.return_value = True
        mock_ollama.return_value = False
        
        # Setup mock failure
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        handler = UnifiedLLMHandler()
        summary = handler.summarize("Test text")
        
        assert summary is None


class TestOllamaSummarization:
    """Tests for Ollama summarization."""
    
    @patch('llm_handler.UnifiedLLMHandler._load_config')
    @patch('llm_handler.UnifiedLLMHandler._check_vllm_availability')
    @patch('llm_handler.UnifiedLLMHandler._check_ollama_availability')
    @patch('ollama.Client')
    def test_ollama_summarize_success(self, mock_client_class, mock_ollama, mock_vllm, mock_config, mock_ollama_client):
        """Test successful summarization with Ollama."""
        config = configparser.ConfigParser()
        config.read_string("""
[llm]
summarize_prompt = Test prompt

[vllm]
enabled = false
url = http://vllm:8000
model = test-model

[ollama]
enabled = true
url = http://localhost:11434
default_model = llama3.1:8b
""")
        mock_config.return_value = config
        mock_vllm.return_value = False
        mock_ollama.return_value = True
        mock_client_class.return_value = mock_ollama_client
        
        handler = UnifiedLLMHandler()
        summary = handler.summarize("Test text to summarize")
        
        assert summary == "This is a test summary of the provided text."
        mock_ollama_client.chat.assert_called_once()


class TestFallbackBehavior:
    """Tests for fallback behavior between backends."""
    
    @patch('llm_handler.UnifiedLLMHandler._load_config')
    @patch('llm_handler.UnifiedLLMHandler._check_vllm_availability')
    @patch('llm_handler.UnifiedLLMHandler._check_ollama_availability')
    @patch('llm_handler.requests.post')
    @patch('ollama.Client')
    def test_fallback_vllm_to_ollama(self, mock_client_class, mock_post, mock_ollama, mock_vllm, mock_config, mock_ollama_client):
        """Test fallback from vLLM to Ollama on failure."""
        config = configparser.ConfigParser()
        config.read_string("""
[llm]
summarize_prompt = Test prompt

[vllm]
enabled = true
url = http://vllm:8000
model = test-model

[ollama]
enabled = true
url = http://localhost:11434
default_model = llama3.1:8b
""")
        mock_config.return_value = config
        mock_vllm.return_value = True
        mock_ollama.return_value = True
        mock_client_class.return_value = mock_ollama_client
        
        # vLLM fails
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        handler = UnifiedLLMHandler()
        summary = handler.summarize("Test text")
        
        # Should fallback to Ollama
        assert summary == "This is a test summary of the provided text."
        mock_ollama_client.chat.assert_called_once()


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @patch('llm_handler.UnifiedLLMHandler._load_config')
    @patch('llm_handler.UnifiedLLMHandler._check_vllm_availability')
    @patch('llm_handler.UnifiedLLMHandler._check_ollama_availability')
    def test_summarize_empty_text(self, mock_ollama, mock_vllm, mock_config):
        """Test summarization with empty text."""
        config = configparser.ConfigParser()
        config.read_string("""
[llm]
summarize_prompt = Test prompt

[vllm]
enabled = true
url = http://vllm:8000
model = test-model

[ollama]
enabled = false
url = http://localhost:11434
default_model = llama3.1:8b
""")
        mock_config.return_value = config
        mock_vllm.return_value = True
        mock_ollama.return_value = False
        
        handler = UnifiedLLMHandler()
        summary = handler.summarize("")
        
        assert summary is None
    
    @patch('llm_handler.UnifiedLLMHandler._load_config')
    @patch('llm_handler.UnifiedLLMHandler._check_vllm_availability')
    @patch('llm_handler.UnifiedLLMHandler._check_ollama_availability')
    def test_summarize_no_backend(self, mock_ollama, mock_vllm, mock_config):
        """Test summarization with no backend available."""
        config = configparser.ConfigParser()
        config.read_string("""
[llm]
summarize_prompt = Test prompt

[vllm]
enabled = false
url = http://vllm:8000
model = test-model

[ollama]
enabled = false
url = http://localhost:11434
default_model = llama3.1:8b
""")
        mock_config.return_value = config
        mock_vllm.return_value = False
        mock_ollama.return_value = False
        
        handler = UnifiedLLMHandler()
        summary = handler.summarize("Test text")
        
        assert summary is None


class TestModelListing:
    """Tests for model listing functionality."""
    
    @patch('llm_handler.UnifiedLLMHandler._load_config')
    @patch('llm_handler.UnifiedLLMHandler._check_vllm_availability')
    @patch('llm_handler.UnifiedLLMHandler._check_ollama_availability')
    @patch('llm_handler.requests.get')
    def test_get_vllm_models(self, mock_get, mock_ollama, mock_vllm, mock_config):
        """Test getting available models from vLLM."""
        config = configparser.ConfigParser()
        config.read_string("""
[llm]
summarize_prompt = Test prompt

[vllm]
enabled = true
url = http://vllm:8000
model = test-model

[ollama]
enabled = false
url = http://localhost:11434
default_model = llama3.1:8b
""")
        mock_config.return_value = config
        mock_vllm.return_value = True
        mock_ollama.return_value = False
        
        # Mock vLLM models response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "Qwen/Qwen2.5-14B-Instruct"},
                {"id": "meta-llama/Llama-3.1-8B-Instruct"}
            ]
        }
        mock_get.return_value = mock_response
        
        handler = UnifiedLLMHandler()
        models = handler.get_available_models()
        
        assert len(models) == 2
        assert "Qwen/Qwen2.5-14B-Instruct" in models
    
    @patch('llm_handler.UnifiedLLMHandler._load_config')
    @patch('llm_handler.UnifiedLLMHandler._check_vllm_availability')
    @patch('llm_handler.UnifiedLLMHandler._check_ollama_availability')
    @patch('ollama.Client')
    def test_get_ollama_models(self, mock_client_class, mock_ollama, mock_vllm, mock_config, mock_ollama_client):
        """Test getting available models from Ollama."""
        config = configparser.ConfigParser()
        config.read_string("""
[llm]
summarize_prompt = Test prompt

[vllm]
enabled = false
url = http://vllm:8000
model = test-model

[ollama]
enabled = true
url = http://localhost:11434
default_model = llama3.1:8b
""")
        mock_config.return_value = config
        mock_vllm.return_value = False
        mock_ollama.return_value = True
        mock_client_class.return_value = mock_ollama_client
        
        handler = UnifiedLLMHandler()
        models = handler.get_available_models()
        
        assert len(models) == 2
        assert "llama3.1:8b" in models
        assert "qwen2.5:14b" in models
