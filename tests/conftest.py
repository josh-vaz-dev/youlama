"""Test configuration and fixtures for YouLama test suite."""
import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
from typing import Generator


# Mock heavy dependencies before they are imported by app.py
# This allows tests to run without GPU/CUDA dependencies
def _setup_mock_dependencies():
    """Setup mock modules for heavy dependencies that may not be installed."""
    mock_modules = {}
    
    # Mock torch if not available
    if 'torch' not in sys.modules:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.get_device_name.return_value = "Mock GPU"
        mock_torch.version.cuda = "12.4"
        mock_torch.backends.cudnn.version.return_value = 8900
        mock_modules['torch'] = mock_torch
    
    # Mock faster_whisper if not available
    if 'faster_whisper' not in sys.modules:
        mock_whisper = MagicMock()
        mock_modules['faster_whisper'] = mock_whisper
    
    # Update sys.modules with mocks
    sys.modules.update(mock_modules)


# Setup mocks before any imports
_setup_mock_dependencies()


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_audio_path(temp_dir: str) -> str:
    """Create a sample audio file for testing."""
    audio_path = os.path.join(temp_dir, "test_audio.wav")
    # Create a minimal WAV file (44 bytes header)
    with open(audio_path, "wb") as f:
        # WAV header
        f.write(b"RIFF")
        f.write((36).to_bytes(4, "little"))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write((16).to_bytes(4, "little"))
        f.write((1).to_bytes(2, "little"))  # Audio format (PCM)
        f.write((1).to_bytes(2, "little"))  # Channels
        f.write((16000).to_bytes(4, "little"))  # Sample rate
        f.write((32000).to_bytes(4, "little"))  # Byte rate
        f.write((2).to_bytes(2, "little"))  # Block align
        f.write((16).to_bytes(2, "little"))  # Bits per sample
        f.write(b"data")
        f.write((0).to_bytes(4, "little"))
    return audio_path


@pytest.fixture
def sample_subtitle_content() -> str:
    """Sample VTT subtitle content."""
    return """WEBVTT

00:00:00.000 --> 00:00:05.000
This is the first subtitle line.

00:00:05.000 --> 00:00:10.000
This is the second subtitle line with more content.

00:00:10.000 --> 00:00:15.000
And this is the third line for testing purposes.
"""


@pytest.fixture
def sample_transcript() -> str:
    """Sample transcript text."""
    return "This is a sample transcript for testing summarization features."


@pytest.fixture
def mock_whisper_model():
    """Mock WhisperModel for testing."""
    mock_model = MagicMock()
    
    # Mock segment
    mock_segment = Mock()
    mock_segment.text = "Sample transcription text"
    mock_segment.start = 0.0
    mock_segment.end = 5.0
    
    # Mock info
    mock_info = Mock()
    mock_info.language = "en"
    mock_info.duration = 5.0
    
    # Mock transcribe method
    mock_model.transcribe.return_value = ([mock_segment], mock_info)
    
    return mock_model


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing."""
    mock_client = MagicMock()
    mock_client.list.return_value = {
        "models": [
            {"model": "llama3.1:8b"},
            {"model": "qwen2.5:14b"}
        ]
    }
    mock_client.chat.return_value = {
        "message": {
            "content": "This is a test summary of the provided text."
        }
    }
    return mock_client


@pytest.fixture
def mock_vllm_response():
    """Mock vLLM API response."""
    return {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test summary from vLLM."
                },
                "finish_reason": "stop"
            }
        ]
    }


@pytest.fixture
def sample_youtube_url() -> str:
    """Sample YouTube URL for testing."""
    return "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture
def mock_yt_dlp_info():
    """Mock yt-dlp video info."""
    return {
        "id": "dQw4w9WgXcQ",
        "title": "Test Video Title",
        "duration": 212,
        "subtitles": {
            "en": [{"ext": "vtt", "url": "https://example.com/sub.vtt"}]
        },
        "formats": [
            {
                "format_id": "140",
                "ext": "m4a",
                "acodec": "mp4a.40.2",
            }
        ]
    }


@pytest.fixture
def sample_config_content() -> str:
    """Sample config.ini content for testing."""
    return """[whisper]
default_model = large-v3
device = cuda
compute_type = float16
beam_size = 8
vad_filter = true
best_of = 5
patience = 1.5
temperature = 0
compression_ratio_threshold = 2.4
log_prob_threshold = -1.0
no_speech_threshold = 0.6
condition_on_previous_text = true
word_timestamps = true

[app]
max_duration = 7200
server_name = 127.0.0.1
server_port = 7860
share = false

[models]
available_models = tiny,base,small,medium,large-v1,large-v2,large-v3

[languages]
available_languages = en,es,fr,de,it,pt,nl,ja,ko,zh,ru,ar,hi

[llm]
summarize_prompt = Create a detailed summary.

[vllm]
enabled = true
url = http://vllm:8000
model = Qwen/Qwen2.5-14B-Instruct

[ollama]
enabled = true
url = http://localhost:11434
default_model = llama3.1:8b
"""


@pytest.fixture
def config_file(temp_dir: str, sample_config_content: str) -> str:
    """Create a temporary config.ini file."""
    config_path = os.path.join(temp_dir, "config.ini")
    with open(config_path, "w") as f:
        f.write(sample_config_content)
    return config_path
