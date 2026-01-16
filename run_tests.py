"""Test runner script for YouLama test suite.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --unit       # Run only unit tests
    python run_tests.py --integration # Run only integration tests
    python run_tests.py --coverage   # Generate coverage report
"""
import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(args):
    """Run pytest with specified arguments."""
    cmd = ["pytest"]
    
    # Add test type filters
    if args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])
    elif args.performance:
        cmd.extend(["-m", "performance"])
    elif args.security:
        cmd.append("tests/test_security.py")
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    
    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    # Run specific test file
    if args.file:
        cmd.append(args.file)
    
    # Add any extra pytest args
    if args.pytest_args:
        cmd.extend(args.pytest_args)
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="YouLama Test Runner")
    
    # Test type selection
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--unit", action="store_true", help="Run unit tests only")
    test_group.add_argument("--integration", action="store_true", help="Run integration tests only")
    test_group.add_argument("--performance", action="store_true", help="Run performance tests only")
    test_group.add_argument("--security", action="store_true", help="Run security tests only")
    test_group.add_argument("--file", type=str, help="Run specific test file")
    
    # Options
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("pytest_args", nargs="*", help="Additional pytest arguments")
    
    args = parser.parse_args()
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("Error: pytest not found. Install with: pip install -r requirements-test.txt")
        return 1
    
    return run_tests(args)


if __name__ == "__main__":
    sys.exit(main())
