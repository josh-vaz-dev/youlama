# YouLama - AI-Powered YouTube Transcription & Summarization

[![Tests](https://github.com/YOUR_USERNAME/YoutubeSummarizer/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/YoutubeSummarizer/actions/workflows/tests.yml)
[![Python 3.11-3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

> Secure, production-ready YouTube video transcription and AI summarization service optimized for NVIDIA RTX 4090.

## Features

- 🎥 **High-Quality Transcription**: Whisper large-v3 with enhanced quality settings
- 🚀 **Multi-Backend AI**: vLLM (primary) with Ollama fallback for summarization
- 🌟 **State-of-Art Models**: Qwen3-8B with thinking mode for best summarization quality
- 📺 **YouTube Support**: Auto-extract subtitles or transcribe audio
- 🔒 **Security Hardened**: Non-root containers, localhost binding, capability dropping
- ⚡ **RTX 4090 Optimized**: CUDA 12.3, Flash Attention 2, optimal memory settings
- 🎨 **Modern Web Interface**: Gradio with real-time progress tracking

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────────┐
│   Gradio Web    │────▶│   Whisper    │────▶│  SHARED vLLM Server │
│   Interface     │     │  large-v3    │     │     Qwen3-8B        │
└─────────────────┘     └──────────────┘     │  (OpenAI-compatible)│
        │                                    └─────────────────────┘
        │                                              │
        │                                              ▼
        │                                    ┌─────────────────────┐
        └────────────────────────────────────│       Ollama        │
                                             │     (Fallback)      │
                                             └─────────────────────┘

Other containers can connect to the shared vLLM server via:
  - Network: llm_network (external: true)
  - API: http://vllm:8000/v1 (OpenAI-compatible)
```

## Requirements

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit
- Ollama installed locally (optional, for summarization)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd youlama
```

2. Install NVIDIA Container Toolkit (if not already installed):
```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2 package
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart the Docker daemon
sudo systemctl restart docker
```

3. Install Ollama locally (optional, for summarization):
```bash
curl https://ollama.ai/install.sh | sh
```

4. Copy the example configuration file:
```bash
cp .env.example .env
```

5. Edit the configuration files:
- `.env`: Set your environment variables
- `config.ini`: Configure Whisper, Ollama, and application settings

## Running the Application

1. Start Ollama locally (if you want to use summarization):
```bash
ollama serve
```

2. Build and start the YouLama container:
```bash
docker-compose up --build
```

3. Open your web browser and navigate to:
```
http://localhost:7860
```

## Configuration

### Environment Variables (.env)

```ini
# Server configuration
SERVER_NAME=0.0.0.0
SERVER_PORT=7860
SHARE=true
```

### Application Settings (config.ini)

```ini
[whisper]
default_model = base
device = cuda
compute_type = float16
beam_size = 5
vad_filter = true

[app]
max_duration = 3600
server_name = 0.0.0.0
server_port = 7860
share = true

[models]
available_models = tiny,base,small,medium,large-v1,large-v2,large-v3

[languages]
available_languages = en,es,fr,de,it,pt,nl,ja,ko,zh

[ollama]
enabled = false
url = http://host.docker.internal:11434
default_model = mistral
summarize_prompt = Please provide a comprehensive yet concise summary of the following text. Focus on the main points, key arguments, and important details while maintaining accuracy and completeness. Here's the text to summarize: 
```

## Features in Detail

### YouTube Video Processing
- Supports youtube.com, youtu.be, and invidious URLs
- Automatically extracts subtitles if available
- Falls back to transcription if no subtitles found
- Optional AI-powered summarization with Ollama

### Local File Transcription
- Supports various audio and video formats
- Automatic language detection
- Multiple Whisper model options
- Optional AI-powered summarization with Ollama

### AI Summarization
- Uses locally running Ollama for text summarization
- Configurable model selection
- Customizable prompt
- Available for both local files and YouTube videos

## Testing

The project includes a comprehensive test suite with 80+ tests covering unit, integration, security, performance, and container testing.

### Quick Start

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m security      # Security tests
pytest -m performance   # Performance tests
```

### Test Categories

| Category | Marker | Description |
|----------|--------|-------------|
| Unit | `@pytest.mark.unit` | Fast, isolated component tests |
| Integration | `@pytest.mark.integration` | Tests involving multiple components |
| Security | `@pytest.mark.security` | Security vulnerability tests |
| Performance | `@pytest.mark.performance` | Benchmarks and load tests |
| Container | `@pytest.mark.container` | CI/CD container service tests |
| Docker | `@pytest.mark.docker` | Docker integration tests |
| Smoke | `@pytest.mark.smoke` | Quick health checks |

### Container Testing (CI/CD)

Container tests validate the built Docker image in a CI/CD environment:

```bash
# Build and start the container
docker build -t youlama:test .
docker run -d --name youlama-app -p 7860:7860 youlama:test

# Run container smoke tests
pytest tests/test_container_service.py -v -m smoke --container-url=http://127.0.0.1:7860

# Run all container tests
pytest tests/test_container_service.py -v --container-url=http://127.0.0.1:7860

# Cleanup
docker stop youlama-app && docker rm youlama-app
```

### Coverage

```bash
# Generate coverage report
pytest --cov=. --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

For detailed testing documentation, see [TESTING.md](TESTING.md).

## CI/CD Pipeline

GitHub Actions workflow runs on every push and pull request:

| Job | Description | Runs After |
|-----|-------------|-----------|
| **test** | Unit & integration tests (Python 3.11-3.14 matrix) | - |
| **security** | Bandit security scan, Safety dependency check | - |
| **lint** | flake8, black, isort code quality | - |
| **container** | Build image, run container tests, vulnerability scan | test, security |

### Security Features

- 🔒 Bandit static analysis for security vulnerabilities
- 🔍 Safety dependency vulnerability scanning
- 🐳 Trivy container image vulnerability scanning
- 🔐 Minimal workflow permissions (`contents: read`)
- 🚫 No credential persistence in checkout
- ⏱️ Job timeouts to prevent runaway builds

## Tips

- For better accuracy, use larger models (medium, large)
- Processing time increases with model size
- GPU is recommended for faster processing
- Maximum audio duration is configurable (default: 60 minutes)
- YouTube videos will first try to use available subtitles
- If no subtitles are available, the video will be transcribed
- Ollama summarization is optional and requires Ollama to be running locally
- The application runs in a Docker container with CUDA support
- Models are downloaded and cached in the `models` directory
- The container connects to the local Ollama instance using host.docker.internal

## Security

### Container Security

The Docker container runs with security hardening:

- **Non-root user**: Application runs as unprivileged user
- **Capability dropping**: `--cap-drop ALL` removes all Linux capabilities
- **No new privileges**: `--security-opt no-new-privileges:true`
- **Read-only filesystem**: Container filesystem is read-only
- **Localhost binding**: Ports bound to `127.0.0.1` only
- **No sensitive data**: No secrets, API keys, or credentials stored in images

### Sensitive Data Protection

The following patterns are excluded from version control (`.gitignore`):

- `.env`, `.env.local`, `.env.*.local` - Environment files
- `secrets/`, `*.key`, `*.pem` - Secret files and keys
- `credentials.json`, `config.ini.local` - Credentials
- `*.log`, `logs/` - Log files that may contain sensitive data

**Note:** The `config.ini` file contains only non-sensitive default configuration. Override sensitive values using environment variables or `config.ini.local` (gitignored).

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 