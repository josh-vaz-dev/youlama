# LLM Agent Instructions for YouLama

> **Purpose**: This file provides structured context for AI coding assistants (GitHub Copilot, Claude, GPT, etc.) working on this codebase.

---

## Project Overview

| Field | Value |
|-------|-------|
| **Name** | YouLama - YouTube Transcription & Summarization |
| **Language** | Python 3.14.2+ |
| **Framework** | Gradio (Web UI), FastAPI-compatible |
| **AI Stack** | Faster-Whisper (transcription), Ollama/vLLM (summarization) |
| **GPU** | NVIDIA RTX 4090 optimized, CUDA 12.4 |
| **Container** | Docker + Docker Compose |

---

## Repository Structure

```
YoutubeSummarizer/
├── app.py                    # Main Gradio application entry point
├── youtube_handler.py        # YouTube URL parsing, subtitle/audio extraction
├── llm_handler.py            # Unified LLM backend (vLLM + Ollama)
├── ollama_handler.py         # Ollama-specific API integration
├── config.ini                # Application configuration
├── requirements.txt          # Production dependencies
├── requirements-test.txt     # Testing dependencies
├── Dockerfile                # Container build instructions
├── docker-compose.yml        # Multi-service orchestration
├── pytest.ini                # Pytest configuration
├── run_tests.py              # Test runner script
├── TESTING.md                # Human-readable testing guide
└── tests/                    # Test suite
    ├── conftest.py           # Shared fixtures and mocks
    ├── test_youtube_handler.py
    ├── test_llm_handler.py
    ├── test_integration.py
    ├── test_performance.py
    ├── test_security.py
    └── test_container_service.py  # CI/CD container service tests
```

---

## Environment Setup

### Activation Commands (Windows PowerShell)

```powershell
# Navigate to project root
cd c:\Users\Josh\Documents\Code\Containers\YoutubeSummarizer

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Verify Python version
python --version  # Should output: Python 3.14.2

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### Python Executable Path

```
C:/Users/Josh/Documents/Code/Containers/YoutubeSummarizer/.venv/Scripts/python.exe
```

---

## Testing Commands

### Quick Reference

| Command | Purpose |
|---------|---------|
| `pytest` | Run all tests |
| `pytest -m unit` | Unit tests only (fast) |
| `pytest -m integration` | Integration tests only |
| `pytest -m performance` | Performance/load tests |
| `pytest -m container` | Container/Docker tests |
| `pytest -m smoke` | Quick smoke tests |
| `pytest tests/test_security.py` | Security vulnerability tests |
| `pytest tests/test_container.py` | Container config tests |
| `pytest tests/test_container_service.py` | CI/CD container service tests |
| `pytest --cov=. --cov-report=html` | Generate coverage report |
| `python run_tests.py --coverage` | Full test run with coverage |

### Container Service Tests (CI/CD)

```powershell
# Build and start container
docker-compose up -d --build youlama

# Run smoke tests against running container
pytest tests/test_container_service.py -v -m smoke --container-url=http://127.0.0.1:7860

# Run all container service tests
pytest tests/test_container_service.py -v --container-url=http://127.0.0.1:7860

# Run Docker integration tests
pytest tests/test_container_service.py -v -m docker
```

### Test Markers

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Component interaction tests
- `@pytest.mark.performance` - Load and performance tests
- `@pytest.mark.container` - Docker container tests
- `@pytest.mark.docker` - Tests requiring Docker CLI
- `@pytest.mark.smoke` - Quick CI/CD smoke tests
- `@pytest.mark.security` - Security-focused tests
- `@pytest.mark.slow` - Tests taking >5 seconds

### Coverage Thresholds

- **Minimum**: 70% line coverage
- **Target**: 85% line coverage
- **Report Location**: `htmlcov/index.html`

---

## Code Patterns

### Import Order (PEP 8 + isort)

```python
# 1. Standard library
import os
import sys
from typing import Optional, List

# 2. Third-party packages
import gradio as gr
import torch
import pytest

# 3. Local modules
from youtube_handler import extract_video_id
from llm_handler import UnifiedLLMHandler
```

### Error Handling Pattern

```python
try:
    result = some_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    raise
except Exception as e:
    logger.exception("Unexpected error")
    raise RuntimeError(f"Unexpected failure: {e}") from e
```

### Fixture Pattern (tests)

```python
@pytest.fixture
def mock_whisper_model():
    """Mock WhisperModel for isolated testing."""
    mock = MagicMock()
    mock.transcribe.return_value = ([mock_segment], mock_info)
    return mock
```

---

## Key Dependencies

### Production (`requirements.txt`)

| Package | Purpose | Min Version |
|---------|---------|-------------|
| gradio | Web UI framework | 5.9.0 |
| faster-whisper | Audio transcription | 1.1.0 |
| torch | PyTorch (CUDA 12.4) | 2.5.1 |
| yt-dlp | YouTube downloading | 2025.1.15 |
| ollama | LLM API client | 0.4.5 |

### Testing (`requirements-test.txt`)

| Package | Purpose | Min Version |
|---------|---------|-------------|
| pytest | Test framework | 8.3.4 |
| pytest-cov | Coverage reporting | 6.0.0 |
| pytest-mock | Mocking utilities | 3.14.0 |
| pytest-asyncio | Async test support | 0.25.2 |
| pytest-xdist | Parallel execution | 3.5.0 |
| bandit | Security scanning | 1.8.2 |
| faker | Test data generation | 33.1.0 |

---

## Common Tasks

### Adding a New Test

1. Create test in appropriate file under `tests/`
2. Use existing fixtures from `conftest.py`
3. Mark with appropriate marker (`@pytest.mark.unit`)
4. Run to verify: `pytest tests/test_<file>.py -v`

### Modifying Dependencies

1. Update `requirements.txt` or `requirements-test.txt`
2. Pin versions with `>=X.Y.Z` format
3. Test installation: `pip install -r <file>`
4. Verify tests pass: `pytest`

### Security Checks

```powershell
# Run Bandit security scan
bandit -r . -x ./tests,./.venv -f screen

# Run all security tests
pytest tests/test_security.py -v
```

---

## Configuration Files

### pytest.ini Key Settings

```ini
testpaths = tests
asyncio_mode = auto
python_files = test_*.py
addopts = -v --showlocals -ra --strict-markers
```

### config.ini Sections

- `[whisper]` - Transcription model settings
- `[vllm]` - vLLM backend configuration (primary)
- `[ollama]` - Ollama backend configuration (fallback)
- `[app]` - Server/UI settings
- `[models]` - Available model list
- `[languages]` - Supported languages

---

## LLM Model Configuration

### Current Model: Qwen3-8B (Recommended)

| Property | Value |
|----------|-------|
| **Model** | `Qwen/Qwen3-8B` |
| **Parameters** | 8.2B |
| **Context Length** | 32K native, 131K with YaRN |
| **VRAM Usage** | ~17GB (fits RTX 4090 24GB) |
| **Release** | May 2025 |
| **Best For** | Summarization, instruction following |

### Why Qwen3-8B?

1. **Thinking Mode** - Can reason through complex transcripts before summarizing
2. **Non-Thinking Mode** - Fast simple summaries when speed matters
3. **Superior Instruction Following** - Better at following detailed summarization prompts
4. **Long Context** - Handles full YouTube transcripts (32K tokens native)
5. **Latest Architecture** - State-of-the-art performance (May 2025)

### Alternative Models (fits 24GB VRAM)

| Model | Parameters | Context | Notes |
|-------|------------|---------|-------|
| `Qwen/Qwen2.5-7B-Instruct` | 7B | 128K | Stable, proven |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | 128K | Requires HF auth |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | 32K | Fast inference |

### vLLM Configuration (docker-compose.yml)

```yaml
command:
  - --model=Qwen/Qwen3-8B
  - --dtype=half
  - --max-model-len=16384
  - --gpu-memory-utilization=0.85
  - --enable-reasoning
  - --reasoning-parser=deepseek_r1
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Activate venv: `.\.venv\Scripts\Activate.ps1` |
| CUDA not available | Verify NVIDIA drivers, reinstall torch with CUDA |
| Tests timeout | Increase timeout: `pytest --timeout=300` |
| Coverage too low | Add tests for uncovered branches |

---

## Safety Notes for LLMs

1. **Never commit secrets** - `.env` files are gitignored
2. **Test before committing** - Run `pytest` to verify changes
3. **Preserve existing tests** - Don't remove tests without reason
4. **Use type hints** - All new functions should have type annotations
5. **Log appropriately** - Use `logger.info/warning/error`

---

*Last Updated: January 2026 | Python 3.14.2 | Maintained for AI-assisted development*
