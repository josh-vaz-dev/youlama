# Testing Guide for YouLama

## Quick Start

### Install Test Dependencies

```powershell
pip install -r requirements-test.txt
```

### Run All Tests

```powershell
pytest
```

### Run Specific Test Categories

```powershell
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Performance tests only
pytest -m performance

# Security tests only
pytest tests/test_security.py
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_youtube_handler.py  # YouTube functionality tests
├── test_llm_handler.py      # LLM backend tests
├── test_integration.py      # End-to-end workflow tests
├── test_performance.py      # Performance and load tests
└── test_security.py         # Security vulnerability tests
```

## Test Categories

### Unit Tests (Fast)
Test individual functions and components in isolation.

```powershell
pytest -m unit -v
```

**Coverage:**
- URL validation
- Video ID extraction
- Config loading
- LLM backend selection
- Error handling

### Integration Tests (Medium)
Test complete workflows and component interactions.

```powershell
pytest -m integration -v
```

**Coverage:**
- YouTube URL → Subtitles → Summary
- YouTube URL → Transcription → Summary
- Local file → Transcription
- Multi-backend LLM fallback

### Performance Tests (Slow)
Test system under load and measure performance metrics.

```powershell
pytest -m performance -v
```

**Coverage:**
- Concurrent request handling
- Memory usage
- Response times
- Scalability limits

### Security Tests
Test for common vulnerabilities and security misconfigurations.

```powershell
pytest tests/test_security.py -v
```

**Coverage:**
- Input validation
- Path traversal prevention
- Code injection
- SSRF prevention
- Docker security
- Secret exposure

### Container Tests
Test Docker container configuration, security hardening, and runtime behavior.

```powershell
# Static analysis (no Docker required)
pytest tests/test_container.py -v

# Container tests only
pytest -m container -v

# With running container (requires docker-compose up)
docker-compose up -d
pytest tests/test_container.py -v
```

**Coverage:**
- Dockerfile security hardening (non-root user, capabilities, etc.)
- docker-compose.yml security (port binding, resource limits)
- Container runtime verification

### Container Service Tests (CI/CD)
Integration tests that run against the actual built container to verify the service works correctly.

```powershell
# Build and start the container
docker-compose up -d --build youlama

# Wait for service to be ready
# The container has a health check that takes ~60 seconds

# Run smoke tests (quick validation)
pytest tests/test_container_service.py -v -m smoke --container-url=http://127.0.0.1:7860

# Run all container service tests
pytest tests/test_container_service.py -v --container-url=http://127.0.0.1:7860

# Run security-focused container tests
pytest tests/test_container_service.py -v -m "container and security"

# Run Docker integration tests (requires Docker CLI)
pytest tests/test_container_service.py -v -m docker

# Skip if container not running
pytest tests/test_container_service.py -v --skip-container-tests
```

**Coverage:**
- Service health and availability
- Gradio interface loading
- Response time validation
- API endpoint accessibility
- Security header checks
- CORS configuration
- Concurrent connection handling
- Docker runtime verification (non-root user, capabilities, memory limits)
- Container log inspection
- Image inspection
- Health check validation

## Coverage Reports

### Generate HTML Coverage Report

```powershell
pytest --cov=. --cov-report=html
# Open htmlcov/index.html in browser
```

### Generate Terminal Coverage Report

```powershell
pytest --cov=. --cov-report=term-missing
```

### Coverage Goals

- **Overall:** > 80%
- **Critical paths:** > 90%
- **Handler modules:** > 85%

## Running Tests in Docker

### Build Test Container

```powershell
docker build -f Dockerfile -t youlama-test .
docker run --rm youlama-test pytest
```

### With Coverage

```powershell
docker run --rm youlama-test pytest --cov=. --cov-report=term
```

## Continuous Integration

Tests run automatically on:
- Every push to main/master/develop
- Every pull request
- Security scans on schedule

See [.github/workflows/tests.yml](.github/workflows/tests.yml)

## Writing New Tests

### Test Template

```python
"""Tests for new_module."""
import pytest
from unittest.mock import patch, MagicMock


class TestNewFeature:
    """Tests for new feature."""
    
    def test_basic_functionality(self):
        """Test basic functionality works."""
        # Arrange
        input_data = "test"
        
        # Act
        result = process(input_data)
        
        # Assert
        assert result is not None
    
    @patch('module.external_call')
    def test_with_mock(self, mock_external):
        """Test with mocked external dependency."""
        # Setup mock
        mock_external.return_value = "mocked"
        
        # Test
        result = call_external()
        
        # Verify
        assert result == "mocked"
        mock_external.assert_called_once()
```

### Fixtures

Use fixtures from `conftest.py`:

```python
def test_with_temp_dir(temp_dir):
    """Test using temp directory fixture."""
    file_path = os.path.join(temp_dir, "test.txt")
    # Use temp_dir
```

### Marks

Mark tests appropriately:

```python
@pytest.mark.unit
def test_unit():
    pass

@pytest.mark.integration
def test_integration():
    pass

@pytest.mark.slow
def test_long_running():
    pass

@pytest.mark.gpu
def test_requires_gpu():
    pass
```

## Debugging Tests

### Run Single Test

```powershell
pytest tests/test_youtube_handler.py::TestYouTubeURLValidation::test_valid_youtube_urls -v
```

### With Debugger

```powershell
pytest --pdb
```

### Print Statements

```powershell
pytest -s  # Show print statements
```

### Verbose Output

```powershell
pytest -vv --showlocals
```

## Performance Benchmarking

```powershell
pytest tests/test_performance.py --benchmark-only
```

## Test Data

Test fixtures are defined in `conftest.py`:
- `sample_audio_path` - Minimal WAV file
- `sample_subtitle_content` - VTT subtitles
- `sample_transcript` - Text transcript
- `mock_whisper_model` - Mocked Whisper
- `mock_ollama_client` - Mocked Ollama

## Troubleshooting

### Import Errors

```powershell
# Add project to PYTHONPATH
$env:PYTHONPATH = "."
pytest
```

### Missing Dependencies

```powershell
pip install -r requirements-test.txt
```

### GPU Tests Failing

Skip GPU tests if no GPU available:

```powershell
pytest -m "not gpu"
```

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Use Fixtures**: Reuse common setup via fixtures
3. **Mock External Services**: Don't call real APIs in tests
4. **Fast Tests**: Keep unit tests under 0.1s each
5. **Descriptive Names**: Use clear test function names
6. **One Assert Per Test**: Focus on single behavior
7. **Arrange-Act-Assert**: Structure tests clearly

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
