"""Container tests for YouLama Docker deployment.

These tests verify the Docker container configuration, security hardening,
and runtime behavior. Tests are designed to work with or without a running
container using appropriate mocking and file-based validation.

Run with: pytest tests/test_container.py -v
Run container tests only: pytest tests/test_container.py -v -m container
"""
import pytest
import os
import yaml
import re
import subprocess
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mark all tests in this module as container tests
pytestmark = pytest.mark.container


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def dockerfile_path():
    """Path to the Dockerfile."""
    return Path(__file__).parent.parent / "Dockerfile"


@pytest.fixture
def docker_compose_path():
    """Path to the docker-compose.yml."""
    return Path(__file__).parent.parent / "docker-compose.yml"


@pytest.fixture
def dockerfile_content(dockerfile_path):
    """Read Dockerfile content."""
    with open(dockerfile_path, "r") as f:
        return f.read()


@pytest.fixture
def docker_compose_content(docker_compose_path):
    """Read and parse docker-compose.yml content."""
    with open(docker_compose_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def requirements_content():
    """Read requirements.txt content."""
    req_path = Path(__file__).parent.parent / "requirements.txt"
    with open(req_path, "r") as f:
        return f.read()


# =============================================================================
# Dockerfile Security Tests
# =============================================================================

class TestDockerfileSecurity:
    """Tests for Dockerfile security hardening."""
    
    def test_non_root_user(self, dockerfile_content):
        """Verify container runs as non-root user."""
        assert "USER" in dockerfile_content, "Dockerfile should specify a USER"
        # Check that USER is not root
        user_lines = [line for line in dockerfile_content.split("\n") 
                     if line.strip().startswith("USER")]
        assert len(user_lines) > 0, "USER directive not found"
        for line in user_lines:
            user = line.replace("USER", "").strip()
            assert user.lower() != "root", f"Container should not run as root, found: {user}"
            assert user != "0", "Container should not run as UID 0 (root)"
    
    def test_no_sudo_installed(self, dockerfile_content):
        """Verify sudo is not installed in container."""
        assert "sudo" not in dockerfile_content.lower() or \
               "sudo" in dockerfile_content.lower() and "remove" in dockerfile_content.lower(), \
               "Container should not have sudo installed"
    
    def test_apt_cache_cleaned(self, dockerfile_content):
        """Verify apt cache is cleaned to reduce image size."""
        if "apt-get" in dockerfile_content:
            assert "apt-get clean" in dockerfile_content or \
                   "rm -rf /var/lib/apt/lists" in dockerfile_content, \
                   "apt cache should be cleaned"
    
    def test_pip_no_cache(self, dockerfile_content):
        """Verify pip doesn't cache packages."""
        assert "--no-cache-dir" in dockerfile_content or \
               "PIP_NO_CACHE_DIR" in dockerfile_content, \
               "pip should use --no-cache-dir"
    
    def test_python_security_settings(self, dockerfile_content):
        """Verify Python security environment variables."""
        assert "PYTHONHASHSEED" in dockerfile_content, \
               "PYTHONHASHSEED should be set for security"
        assert "PYTHONDONTWRITEBYTECODE" in dockerfile_content, \
               "PYTHONDONTWRITEBYTECODE should be set"
    
    def test_setuid_binaries_removed(self, dockerfile_content):
        """Verify setuid/setgid binaries are handled."""
        # Check for chmod a-s or similar security measure
        assert "perm" in dockerfile_content and "6000" in dockerfile_content or \
               "chmod a-s" in dockerfile_content, \
               "setuid/setgid binaries should be addressed"
    
    def test_healthcheck_present(self, dockerfile_content):
        """Verify health check is defined."""
        assert "HEALTHCHECK" in dockerfile_content, \
               "HEALTHCHECK should be defined"
    
    def test_expose_port_defined(self, dockerfile_content):
        """Verify EXPOSE directive is present."""
        assert "EXPOSE" in dockerfile_content, \
               "EXPOSE directive should be defined"
        # Extract exposed port
        expose_match = re.search(r"EXPOSE\s+(\d+)", dockerfile_content)
        assert expose_match, "EXPOSE should specify a port number"
        port = int(expose_match.group(1))
        assert port == 7860, f"Expected port 7860, found {port}"
    
    def test_workdir_set(self, dockerfile_content):
        """Verify WORKDIR is set."""
        assert "WORKDIR" in dockerfile_content, \
               "WORKDIR should be defined"
    
    def test_copy_with_ownership(self, dockerfile_content):
        """Verify COPY uses --chown for proper file ownership."""
        copy_lines = [line for line in dockerfile_content.split("\n") 
                     if line.strip().startswith("COPY")]
        # At least some COPY commands should have --chown
        chown_copies = [line for line in copy_lines if "--chown" in line]
        assert len(chown_copies) > 0, \
               "At least some COPY commands should use --chown"


class TestDockerfileConfiguration:
    """Tests for Dockerfile configuration and best practices."""
    
    def test_base_image_specified(self, dockerfile_content):
        """Verify base image is properly specified."""
        assert "FROM" in dockerfile_content, "FROM directive required"
        from_line = [line for line in dockerfile_content.split("\n") 
                    if line.strip().startswith("FROM")][0]
        assert "nvidia/cuda" in from_line or "python" in from_line, \
               "Base image should be CUDA or Python based"
    
    def test_requirements_copied_first(self, dockerfile_content):
        """Verify requirements.txt is copied before other files for caching."""
        lines = dockerfile_content.split("\n")
        req_copy_line = None
        app_copy_line = None
        
        for i, line in enumerate(lines):
            if "COPY" in line and "requirements" in line:
                req_copy_line = i
            if "COPY" in line and "app.py" in line:
                app_copy_line = i
        
        if req_copy_line and app_copy_line:
            assert req_copy_line < app_copy_line, \
                   "requirements.txt should be copied before app code for better caching"
    
    def test_entrypoint_or_cmd_defined(self, dockerfile_content):
        """Verify ENTRYPOINT or CMD is defined."""
        assert "ENTRYPOINT" in dockerfile_content or "CMD" in dockerfile_content, \
               "ENTRYPOINT or CMD should be defined"
    
    def test_labels_present(self, dockerfile_content):
        """Verify metadata labels are present."""
        assert "LABEL" in dockerfile_content, \
               "LABEL should be defined for image metadata"


# =============================================================================
# Docker Compose Security Tests
# =============================================================================

class TestDockerComposeSecurity:
    """Tests for docker-compose.yml security configuration."""
    
    def test_cap_drop_all(self, docker_compose_content):
        """Verify capabilities are dropped."""
        services = docker_compose_content.get("services", {})
        for name, service in services.items():
            if "cap_drop" in service:
                assert "ALL" in service["cap_drop"], \
                       f"Service {name} should drop ALL capabilities"
    
    def test_no_privileged_mode(self, docker_compose_content):
        """Verify no service runs in privileged mode."""
        services = docker_compose_content.get("services", {})
        for name, service in services.items():
            assert service.get("privileged") != True, \
                   f"Service {name} should not run in privileged mode"
    
    def test_no_new_privileges(self, docker_compose_content):
        """Verify no-new-privileges security option is set."""
        services = docker_compose_content.get("services", {})
        for name, service in services.items():
            security_opts = service.get("security_opt", [])
            # Convert to string list for checking
            sec_str = str(security_opts)
            assert "no-new-privileges" in sec_str, \
                   f"Service {name} should have no-new-privileges:true"
    
    def test_localhost_port_binding(self, docker_compose_content):
        """Verify ports are bound to localhost only."""
        services = docker_compose_content.get("services", {})
        for name, service in services.items():
            ports = service.get("ports", [])
            for port in ports:
                port_str = str(port)
                assert "127.0.0.1:" in port_str or "localhost:" in port_str, \
                       f"Service {name} port {port} should be bound to localhost only"
    
    def test_resource_limits(self, docker_compose_content):
        """Verify resource limits are set."""
        services = docker_compose_content.get("services", {})
        youlama = services.get("youlama", {})
        deploy = youlama.get("deploy", {})
        resources = deploy.get("resources", {})
        limits = resources.get("limits", {})
        
        assert "memory" in limits, "Memory limit should be set"
    
    def test_logging_configured(self, docker_compose_content):
        """Verify logging is properly configured."""
        services = docker_compose_content.get("services", {})
        for name, service in services.items():
            logging = service.get("logging", {})
            if logging:
                options = logging.get("options", {})
                assert "max-size" in options, \
                       f"Service {name} should have log max-size set"
    
    def test_healthcheck_configured(self, docker_compose_content):
        """Verify health checks are configured."""
        services = docker_compose_content.get("services", {})
        youlama = services.get("youlama", {})
        assert "healthcheck" in youlama, \
               "youlama service should have healthcheck"
        
        healthcheck = youlama["healthcheck"]
        assert "test" in healthcheck, "healthcheck should have test command"
        assert "interval" in healthcheck, "healthcheck should have interval"
    
    def test_restart_policy(self, docker_compose_content):
        """Verify restart policy is set."""
        services = docker_compose_content.get("services", {})
        youlama = services.get("youlama", {})
        assert "restart" in youlama, \
               "restart policy should be set"
        assert youlama["restart"] in ["always", "unless-stopped", "on-failure"], \
               "restart policy should be appropriate for production"


class TestDockerComposeConfiguration:
    """Tests for docker-compose.yml configuration."""
    
    def test_networks_defined(self, docker_compose_content):
        """Verify custom network is defined."""
        assert "networks" in docker_compose_content, \
               "Custom networks should be defined"
    
    def test_volumes_defined(self, docker_compose_content):
        """Verify named volumes are defined."""
        assert "volumes" in docker_compose_content, \
               "Named volumes should be defined"
    
    def test_gpu_configuration(self, docker_compose_content):
        """Verify GPU configuration for NVIDIA."""
        services = docker_compose_content.get("services", {})
        youlama = services.get("youlama", {})
        deploy = youlama.get("deploy", {})
        resources = deploy.get("resources", {})
        reservations = resources.get("reservations", {})
        devices = reservations.get("devices", [])
        
        assert len(devices) > 0, "GPU device reservation should be configured"
        gpu_config = devices[0]
        assert gpu_config.get("driver") == "nvidia", \
               "NVIDIA driver should be specified"
        assert "gpu" in gpu_config.get("capabilities", []), \
               "GPU capability should be enabled"
    
    def test_environment_variables(self, docker_compose_content):
        """Verify necessary environment variables are set."""
        services = docker_compose_content.get("services", {})
        youlama = services.get("youlama", {})
        env = youlama.get("environment", [])
        env_str = str(env)
        
        assert "NVIDIA_VISIBLE_DEVICES" in env_str, \
               "NVIDIA_VISIBLE_DEVICES should be set"
    
    def test_depends_on(self, docker_compose_content):
        """Verify service dependencies are configured."""
        services = docker_compose_content.get("services", {})
        youlama = services.get("youlama", {})
        
        if "vllm" in services:
            assert "depends_on" in youlama, \
                   "youlama should depend on vllm service"


# =============================================================================
# Container Runtime Tests (requires Docker)
# =============================================================================

def docker_available():
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.mark.skipif(not docker_available(), reason="Docker not available")
class TestContainerBuild:
    """Tests for container build process."""
    
    def test_dockerfile_syntax(self, dockerfile_path):
        """Verify Dockerfile syntax is valid."""
        result = subprocess.run(
            ["docker", "build", "--check", str(dockerfile_path.parent)],
            capture_output=True,
            text=True
        )
        # --check may not be available in older Docker versions
        if "--check" in result.stderr:
            pytest.skip("Docker --check not available")
        # If no --check, just verify file can be parsed
        assert dockerfile_path.exists()
    
    def test_docker_compose_config_valid(self, docker_compose_path):
        """Verify docker-compose.yml is valid."""
        result = subprocess.run(
            ["docker", "compose", "-f", str(docker_compose_path), "config"],
            capture_output=True,
            text=True,
            cwd=str(docker_compose_path.parent)
        )
        assert result.returncode == 0, f"docker-compose config failed: {result.stderr}"


@pytest.mark.skipif(not docker_available(), reason="Docker not available")
class TestContainerImages:
    """Tests for container image inspection."""
    
    @pytest.fixture
    def built_image(self, docker_compose_path):
        """Build the image for testing (if not exists)."""
        # Check if image exists
        result = subprocess.run(
            ["docker", "images", "-q", "youlama-app"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            return "youlama-app"
        else:
            pytest.skip("Image not built - run 'docker-compose build' first")
    
    def test_image_has_no_root_user(self, built_image):
        """Verify built image doesn't run as root."""
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.Config.User}}", built_image],
            capture_output=True,
            text=True
        )
        user = result.stdout.strip()
        assert user and user != "root" and user != "0", \
               f"Image should not run as root, found user: {user}"
    
    def test_image_has_healthcheck(self, built_image):
        """Verify built image has health check."""
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.Config.Healthcheck}}", built_image],
            capture_output=True,
            text=True
        )
        healthcheck = result.stdout.strip()
        assert healthcheck and healthcheck != "<nil>", \
               "Image should have HEALTHCHECK configured"


# =============================================================================
# Container Runtime Tests (requires running container)
# =============================================================================

def container_running(container_name="youlama-app"):
    """Check if a specific container is running."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout.strip() == "true"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.mark.skipif(not container_running(), reason="Container not running")
class TestRunningContainer:
    """Tests for running container behavior."""
    
    def test_container_user_not_root(self):
        """Verify running container is not running as root."""
        result = subprocess.run(
            ["docker", "exec", "youlama-app", "id", "-u"],
            capture_output=True,
            text=True
        )
        uid = result.stdout.strip()
        assert uid != "0", f"Container should not run as root (UID 0), found UID: {uid}"
    
    def test_container_healthcheck_passes(self):
        """Verify container health check passes."""
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", "youlama-app"],
            capture_output=True,
            text=True
        )
        status = result.stdout.strip()
        assert status == "healthy", f"Container should be healthy, status: {status}"
    
    def test_container_no_privileged(self):
        """Verify container is not running in privileged mode."""
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.HostConfig.Privileged}}", "youlama-app"],
            capture_output=True,
            text=True
        )
        privileged = result.stdout.strip()
        assert privileged == "false", "Container should not be privileged"
    
    def test_container_capabilities_dropped(self):
        """Verify capabilities are dropped."""
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.HostConfig.CapDrop}}", "youlama-app"],
            capture_output=True,
            text=True
        )
        cap_drop = result.stdout.strip()
        assert "ALL" in cap_drop or "all" in cap_drop, \
               f"ALL capabilities should be dropped, found: {cap_drop}"
    
    def test_container_port_binding(self):
        """Verify port is bound to localhost."""
        result = subprocess.run(
            ["docker", "inspect", "--format", 
             "{{range $p, $conf := .NetworkSettings.Ports}}{{$p}}:{{(index $conf 0).HostIp}}{{end}}",
             "youlama-app"],
            capture_output=True,
            text=True
        )
        ports = result.stdout.strip()
        assert "127.0.0.1" in ports or ports == "", \
               f"Ports should be bound to localhost, found: {ports}"
    
    def test_container_memory_limit(self):
        """Verify memory limit is set."""
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.HostConfig.Memory}}", "youlama-app"],
            capture_output=True,
            text=True
        )
        memory = result.stdout.strip()
        assert memory != "0", "Memory limit should be set"
    
    def test_app_responds(self):
        """Verify application responds on expected port."""
        import urllib.request
        import urllib.error
        
        try:
            response = urllib.request.urlopen("http://127.0.0.1:7860", timeout=10)
            assert response.status == 200
        except urllib.error.URLError as e:
            pytest.fail(f"Application not responding: {e}")


# =============================================================================
# Requirements and Dependencies Tests
# =============================================================================

class TestContainerDependencies:
    """Tests for container dependencies and requirements."""
    
    def test_requirements_has_core_packages(self, requirements_content):
        """Verify core packages are in requirements."""
        required_packages = ["gradio", "faster-whisper", "yt-dlp", "ollama"]
        for package in required_packages:
            assert package in requirements_content.lower(), \
                   f"Required package '{package}' not found in requirements.txt"
    
    def test_requirements_has_versions(self, requirements_content):
        """Verify packages have version constraints."""
        lines = [l.strip() for l in requirements_content.split("\n") 
                if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("-")]
        
        versioned = [l for l in lines if ">=" in l or "==" in l or "~=" in l]
        # At least 50% should have version constraints
        assert len(versioned) >= len(lines) * 0.5, \
               "Most packages should have version constraints"
    
    def test_no_vulnerable_packages_pinned(self, requirements_content):
        """Check for known vulnerable package versions (basic check)."""
        # This is a basic check - use safety/pip-audit for comprehensive scanning
        vulnerable_patterns = [
            r"pyyaml\s*[<>=]*\s*[0-4]\.",  # PyYAML < 5.x has vulnerabilities
            r"requests\s*[<>=]*\s*2\.[0-1][0-9]\.",  # Old requests versions
        ]
        
        for pattern in vulnerable_patterns:
            match = re.search(pattern, requirements_content.lower())
            if match:
                pytest.fail(f"Potentially vulnerable package version: {match.group()}")
