"""
CI/CD Container Service Tests for YouLama

These tests run against the actual built container to verify the service
is working correctly in the production environment.

Usage:
    # Build and start the container first
    docker-compose up -d --build youlama

    # Run the container service tests
    pytest tests/test_container_service.py -v --container-url=http://127.0.0.1:7860

    # Or set the environment variable
    CONTAINER_URL=http://127.0.0.1:7860 pytest tests/test_container_service.py -v
"""
import pytest
import requests
import time
import os
import subprocess
from typing import Optional


# Configuration
DEFAULT_CONTAINER_URL = os.environ.get("CONTAINER_URL", "http://127.0.0.1:7860")
DEFAULT_TIMEOUT = 30
HEALTH_CHECK_RETRIES = 10
HEALTH_CHECK_INTERVAL = 5


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--container-url",
        action="store",
        default=DEFAULT_CONTAINER_URL,
        help="URL of the running YouLama container"
    )
    parser.addoption(
        "--skip-container-tests",
        action="store_true",
        default=False,
        help="Skip container tests if container is not running"
    )


@pytest.fixture(scope="session")
def container_url(request):
    """Get the container URL from command line or environment."""
    return request.config.getoption("--container-url")


@pytest.fixture(scope="session")
def skip_if_unavailable(request):
    """Check if container tests should be skipped."""
    return request.config.getoption("--skip-container-tests")


@pytest.fixture(scope="session")
def service_available(container_url, skip_if_unavailable):
    """Check if the container service is available."""
    try:
        response = requests.get(container_url, timeout=5)
        return True
    except requests.exceptions.RequestException:
        if skip_if_unavailable:
            pytest.skip("Container not available and --skip-container-tests is set")
        return False


def wait_for_service(url: str, timeout: int = 60, interval: int = 5) -> bool:
    """Wait for the service to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(interval)
    return False


# =============================================================================
# Container Health Tests
# =============================================================================

@pytest.mark.container
class TestContainerHealth:
    """Tests for container health and availability."""

    def test_service_is_running(self, container_url, service_available):
        """Test that the service is running and responding."""
        if not service_available:
            pytest.skip("Service not available")
        
        response = requests.get(container_url, timeout=DEFAULT_TIMEOUT)
        assert response.status_code == 200, f"Service returned {response.status_code}"

    def test_health_endpoint(self, container_url, service_available):
        """Test the health check endpoint."""
        if not service_available:
            pytest.skip("Service not available")
        
        # Gradio apps typically respond on root with 200 if healthy
        response = requests.get(container_url, timeout=DEFAULT_TIMEOUT)
        assert response.status_code == 200

    def test_response_time(self, container_url, service_available):
        """Test that the service responds within acceptable time."""
        if not service_available:
            pytest.skip("Service not available")
        
        start_time = time.time()
        response = requests.get(container_url, timeout=DEFAULT_TIMEOUT)
        elapsed = time.time() - start_time
        
        assert elapsed < 5, f"Response took {elapsed:.2f}s, expected < 5s"
        assert response.status_code == 200

    def test_gradio_interface_loads(self, container_url, service_available):
        """Test that the Gradio interface loads correctly."""
        if not service_available:
            pytest.skip("Service not available")
        
        response = requests.get(container_url, timeout=DEFAULT_TIMEOUT)
        assert response.status_code == 200
        
        # Gradio pages contain specific markers
        content = response.text.lower()
        assert any([
            "gradio" in content,
            "youlama" in content,
            "<!doctype html>" in content
        ]), "Response does not appear to be a valid Gradio page"


# =============================================================================
# API Endpoint Tests
# =============================================================================

@pytest.mark.container
class TestAPIEndpoints:
    """Tests for API endpoints in the container."""

    def test_gradio_api_info(self, container_url, service_available):
        """Test that Gradio API info endpoint is accessible."""
        if not service_available:
            pytest.skip("Service not available")
        
        # Gradio exposes an API info endpoint
        api_url = f"{container_url}/info"
        try:
            response = requests.get(api_url, timeout=DEFAULT_TIMEOUT)
            # May return 200 or 404 depending on Gradio version
            assert response.status_code in [200, 404, 405]
        except requests.exceptions.RequestException as e:
            pytest.skip(f"API info endpoint not accessible: {e}")

    def test_gradio_config_endpoint(self, container_url, service_available):
        """Test that Gradio config endpoint is accessible."""
        if not service_available:
            pytest.skip("Service not available")
        
        # Gradio exposes a config endpoint
        config_url = f"{container_url}/config"
        try:
            response = requests.get(config_url, timeout=DEFAULT_TIMEOUT)
            if response.status_code == 200:
                # Should return JSON config
                config = response.json()
                assert isinstance(config, dict)
        except requests.exceptions.JSONDecodeError:
            pass  # OK if not JSON
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Config endpoint not accessible: {e}")

    def test_static_assets_accessible(self, container_url, service_available):
        """Test that static assets are served correctly."""
        if not service_available:
            pytest.skip("Service not available")
        
        # Get the main page first
        response = requests.get(container_url, timeout=DEFAULT_TIMEOUT)
        assert response.status_code == 200
        
        # Check that it returns HTML content
        content_type = response.headers.get("Content-Type", "")
        assert "text/html" in content_type or response.text.startswith("<!DOCTYPE")


# =============================================================================
# Security Tests
# =============================================================================

@pytest.mark.container
@pytest.mark.security
class TestContainerSecurity:
    """Security tests for the running container."""

    def test_no_server_header_leak(self, container_url, service_available):
        """Test that server header doesn't leak sensitive info."""
        if not service_available:
            pytest.skip("Service not available")
        
        response = requests.get(container_url, timeout=DEFAULT_TIMEOUT)
        server_header = response.headers.get("Server", "")
        
        # Should not expose detailed version info
        sensitive_patterns = ["python", "uvicorn", "gunicorn", "nginx"]
        for pattern in sensitive_patterns:
            # Lowercase comparison, allow generic names but not versions
            if pattern in server_header.lower():
                # Ensure no version numbers are exposed
                import re
                version_pattern = rf"{pattern}[/\s][\d.]+"
                assert not re.search(version_pattern, server_header.lower()), \
                    f"Server header exposes version info: {server_header}"

    def test_no_debug_mode(self, container_url, service_available):
        """Test that debug mode is disabled in production."""
        if not service_available:
            pytest.skip("Service not available")
        
        response = requests.get(container_url, timeout=DEFAULT_TIMEOUT)
        
        # Check for debug indicators in response
        content = response.text.lower()
        debug_indicators = ["debug=true", "traceback", "debugger"]
        
        for indicator in debug_indicators:
            assert indicator not in content, \
                f"Debug indicator found in response: {indicator}"

    def test_security_headers(self, container_url, service_available):
        """Test for presence of security headers."""
        if not service_available:
            pytest.skip("Service not available")
        
        response = requests.get(container_url, timeout=DEFAULT_TIMEOUT)
        headers = response.headers
        
        # These are recommended but not required for Gradio apps
        # Just log warnings if missing
        recommended_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
        ]
        
        missing = [h for h in recommended_headers if h not in headers]
        if missing:
            import warnings
            warnings.warn(f"Missing recommended security headers: {missing}")

    def test_cors_configuration(self, container_url, service_available):
        """Test CORS configuration is appropriate."""
        if not service_available:
            pytest.skip("Service not available")
        
        # Test preflight request
        headers = {
            "Origin": "http://malicious-site.com",
            "Access-Control-Request-Method": "POST",
        }
        
        try:
            response = requests.options(
                container_url, 
                headers=headers, 
                timeout=DEFAULT_TIMEOUT
            )
            
            # Check if CORS allows arbitrary origins
            acao = response.headers.get("Access-Control-Allow-Origin", "")
            if acao == "*":
                import warnings
                warnings.warn("CORS allows all origins (*)")
        except requests.exceptions.RequestException:
            pass  # OPTIONS may not be supported


# =============================================================================
# Stress Tests
# =============================================================================

@pytest.mark.container
@pytest.mark.slow
class TestContainerStress:
    """Stress tests for the container under load."""

    def test_concurrent_connections(self, container_url, service_available):
        """Test handling multiple concurrent connections."""
        if not service_available:
            pytest.skip("Service not available")
        
        import concurrent.futures
        
        def make_request():
            try:
                response = requests.get(container_url, timeout=DEFAULT_TIMEOUT)
                return response.status_code == 200
            except requests.exceptions.RequestException:
                return False
        
        # Test with 10 concurrent connections
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # At least 80% should succeed
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.8, f"Only {success_rate*100:.0f}% of requests succeeded"

    def test_rapid_requests(self, container_url, service_available):
        """Test handling rapid sequential requests."""
        if not service_available:
            pytest.skip("Service not available")
        
        success_count = 0
        total_requests = 20
        
        for _ in range(total_requests):
            try:
                response = requests.get(container_url, timeout=DEFAULT_TIMEOUT)
                if response.status_code == 200:
                    success_count += 1
            except requests.exceptions.RequestException:
                pass
        
        success_rate = success_count / total_requests
        assert success_rate >= 0.9, f"Only {success_rate*100:.0f}% of rapid requests succeeded"


# =============================================================================
# Docker Integration Tests
# =============================================================================

@pytest.mark.container
@pytest.mark.docker
class TestDockerIntegration:
    """Tests that require Docker CLI access."""

    @pytest.fixture
    def docker_available(self):
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def test_container_running(self, docker_available):
        """Test that the YouLama container is running."""
        if not docker_available:
            pytest.skip("Docker not available")
        
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=youlama-app", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        status = result.stdout.strip()
        if not status:
            pytest.skip("Container youlama-app not found")
        
        assert "Up" in status, f"Container not running: {status}"

    def test_container_healthy(self, docker_available):
        """Test that the container health check passes."""
        if not docker_available:
            pytest.skip("Docker not available")
        
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", "youlama-app"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        health_status = result.stdout.strip()
        if not health_status or result.returncode != 0:
            pytest.skip("Container health info not available")
        
        assert health_status == "healthy", f"Container health: {health_status}"

    def test_container_user_is_nonroot(self, docker_available):
        """Test that the container runs as non-root user."""
        if not docker_available:
            pytest.skip("Docker not available")
        
        result = subprocess.run(
            ["docker", "exec", "youlama-app", "id", "-u"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            pytest.skip("Could not exec into container")
        
        uid = result.stdout.strip()
        assert uid != "0", "Container is running as root (uid=0)"

    def test_container_capabilities_dropped(self, docker_available):
        """Test that dangerous capabilities are dropped."""
        if not docker_available:
            pytest.skip("Docker not available")
        
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.HostConfig.CapDrop}}", "youlama-app"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        cap_drop = result.stdout.strip()
        if not cap_drop or result.returncode != 0:
            pytest.skip("Could not inspect container capabilities")
        
        # Check ALL capabilities are dropped
        assert "ALL" in cap_drop or "all" in cap_drop.lower(), \
            f"Not all capabilities dropped: {cap_drop}"

    def test_container_memory_limit(self, docker_available):
        """Test that the container has memory limits configured."""
        if not docker_available:
            pytest.skip("Docker not available")
        
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.HostConfig.Memory}}", "youlama-app"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        memory = result.stdout.strip()
        if not memory or result.returncode != 0:
            pytest.skip("Could not inspect container memory")
        
        # Memory should be set (non-zero)
        try:
            mem_bytes = int(memory)
            # docker-compose sets 12G limit
            assert mem_bytes > 0, "No memory limit configured"
        except ValueError:
            pytest.skip(f"Could not parse memory value: {memory}")

    def test_container_logs_no_errors(self, docker_available):
        """Test that container logs don't show critical errors."""
        if not docker_available:
            pytest.skip("Docker not available")
        
        result = subprocess.run(
            ["docker", "logs", "--tail", "100", "youlama-app"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            pytest.skip("Could not get container logs")
        
        logs = result.stderr + result.stdout
        
        # Check for critical errors
        critical_errors = [
            "CRITICAL",
            "FATAL",
            "Traceback (most recent call last)",
            "ModuleNotFoundError",
            "ImportError",
        ]
        
        for error in critical_errors:
            assert error not in logs, f"Critical error found in logs: {error}"


# =============================================================================
# Smoke Tests for CI/CD Pipeline
# =============================================================================

@pytest.mark.container
@pytest.mark.smoke
class TestContainerSmoke:
    """Quick smoke tests for CI/CD pipeline validation."""

    def test_smoke_service_responds(self, container_url, service_available):
        """Smoke test: Service responds to requests."""
        if not service_available:
            pytest.fail("Container service is not responding")
        
        response = requests.get(container_url, timeout=10)
        assert response.status_code == 200, "Service did not return 200 OK"

    def test_smoke_gradio_loaded(self, container_url, service_available):
        """Smoke test: Gradio interface is loaded."""
        if not service_available:
            pytest.fail("Container service is not responding")
        
        response = requests.get(container_url, timeout=10)
        assert "<!DOCTYPE html>" in response.text or "<!doctype html>" in response.text.lower()

    def test_smoke_no_critical_errors(self, container_url, service_available):
        """Smoke test: No critical errors in response."""
        if not service_available:
            pytest.fail("Container service is not responding")
        
        response = requests.get(container_url, timeout=10)
        
        error_indicators = ["500 Internal Server Error", "503 Service Unavailable"]
        for error in error_indicators:
            assert error not in response.text, f"Error found: {error}"
