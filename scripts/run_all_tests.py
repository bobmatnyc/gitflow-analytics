#!/usr/bin/env python3
"""Comprehensive test runner for GitFlow Analytics.

This script runs all tests including:
- Unit tests
- Integration tests
- TUI tests
- CLI tests
- Configuration tests
- ML/Qualitative tests
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


class TestRunner:
    """Comprehensive test runner for GitFlow Analytics."""

    def __init__(self, verbose: bool = False, coverage: bool = False):
        self.verbose = verbose
        self.coverage = coverage
        self.failed_tests: list[str] = []
        self.passed_tests: list[str] = []

    def run_command(self, cmd: list[str], description: str) -> bool:
        """Run a command and return success status."""
        print(f"\n🧪 {description}")
        print(f"   Command: {' '.join(cmd)}")

        if self.verbose:
            print(f"   Working directory: {os.getcwd()}")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                print(f"   ✅ PASSED ({duration:.1f}s)")
                self.passed_tests.append(description)
                return True
            else:
                print(f"   ❌ FAILED ({duration:.1f}s)")
                if not self.verbose and result.stdout:
                    print(f"   STDOUT: {result.stdout[:500]}...")
                if not self.verbose and result.stderr:
                    print(f"   STDERR: {result.stderr[:500]}...")
                self.failed_tests.append(description)
                return False

        except subprocess.TimeoutExpired:
            print("   ⏰ TIMEOUT after 5 minutes")
            self.failed_tests.append(f"{description} (TIMEOUT)")
            return False
        except Exception as e:
            print(f"   💥 ERROR: {e}")
            self.failed_tests.append(f"{description} (ERROR)")
            return False

    def run_pytest_suite(
        self, test_paths: list[str], description: str, markers: Optional[str] = None
    ) -> bool:
        """Run a pytest test suite."""
        cmd = ["python", "-m", "pytest"]

        if self.coverage:
            cmd.extend(["--cov=gitflow_analytics", "--cov-report=term-missing"])

        if self.verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        if markers:
            cmd.extend(["-m", markers])

        # Add all test paths
        cmd.extend(test_paths)

        return self.run_command(cmd, description)

    def run_all_tests(self, test_categories: Optional[list[str]] = None) -> bool:
        """Run all test categories."""
        print("🚀 GitFlow Analytics - Comprehensive Test Suite")
        print("=" * 60)

        # Default to all categories if none specified
        if not test_categories:
            test_categories = [
                "unit",
                "integration",
                "tui",
                "cli",
                "config",
                "ml",
                "reports",
                "core",
                "extractors",
            ]

        all_passed = True

        # 1. Unit Tests
        if "unit" in test_categories:
            # Run specific unit test files
            unit_tests = [
                "tests/test_atomic_caching.py",
                "tests/test_branch_analysis.py",
                "tests/test_classification_system.py",
                "tests/test_metrics.py",
                "tests/test_reports.py",
            ]
            all_passed &= self.run_pytest_suite(unit_tests, "Unit Tests (Core Functionality)")

        # 2. Core Module Tests
        if "core" in test_categories:
            all_passed &= self.run_pytest_suite(["tests/core/"], "Core Module Tests")

        # 3. Configuration Tests
        if "config" in test_categories:
            all_passed &= self.run_pytest_suite(
                [
                    "tests/test_config.py",
                    "tests/test_config_extends.py",
                    "tests/test_config_profiles.py",
                ],
                "Configuration Tests",
            )

        # 4. CLI Tests
        if "cli" in test_categories:
            all_passed &= self.run_pytest_suite(["tests/test_cli.py"], "CLI Tests")

        # 5. TUI Tests
        if "tui" in test_categories:
            all_passed &= self.run_pytest_suite(
                ["tests/tui/"], "TUI (Terminal User Interface) Tests"
            )

        # 6. Extractor Tests
        if "extractors" in test_categories:
            all_passed &= self.run_pytest_suite(["tests/extractors/"], "Data Extractor Tests")

        # 7. Integration Tests
        if "integration" in test_categories:
            all_passed &= self.run_pytest_suite(["tests/integrations/"], "Integration Tests")

        # 8. ML/Qualitative Tests
        if "ml" in test_categories:
            all_passed &= self.run_pytest_suite(
                [
                    "tests/qualitative/",
                    "tests/test_ml_accuracy.py",
                    "tests/test_ml_components.py",
                    "tests/test_classification_system.py",
                ],
                "ML & Qualitative Analysis Tests",
            )

        # 9. Report Generation Tests
        if "reports" in test_categories:
            all_passed &= self.run_pytest_suite(
                ["tests/reports/", "tests/test_reports.py", "tests/test_report_abstraction.py"],
                "Report Generation Tests",
            )

        # 10. Metrics Tests
        if "metrics" in test_categories:
            all_passed &= self.run_pytest_suite(
                ["tests/metrics/", "tests/test_metrics.py"], "Metrics Calculation Tests"
            )

        return all_passed

    def run_linting_and_formatting(self) -> bool:
        """Run code quality checks."""
        print("\n🔍 Code Quality Checks")
        print("=" * 30)

        all_passed = True

        # Type checking with mypy
        all_passed &= self.run_command(
            ["python", "-m", "mypy", "src/gitflow_analytics", "--ignore-missing-imports"],
            "Type Checking (mypy)",
        )

        # Linting with ruff
        all_passed &= self.run_command(["python", "-m", "ruff", "check", "src/"], "Linting (ruff)")

        # Code formatting check with black
        all_passed &= self.run_command(
            ["python", "-m", "black", "--check", "src/"], "Code Formatting Check (black)"
        )

        return all_passed

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        total_tests = len(self.passed_tests) + len(self.failed_tests)

        print(f"Total Test Suites: {total_tests}")
        print(f"✅ Passed: {len(self.passed_tests)}")
        print(f"❌ Failed: {len(self.failed_tests)}")

        if self.passed_tests:
            print("\n✅ PASSED TESTS:")
            for test in self.passed_tests:
                print(f"   • {test}")

        if self.failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in self.failed_tests:
                print(f"   • {test}")

        success_rate = (len(self.passed_tests) / total_tests * 100) if total_tests > 0 else 0
        print(f"\n📈 Success Rate: {success_rate:.1f}%")

        if len(self.failed_tests) == 0:
            print("\n🎉 ALL TESTS PASSED!")
            return True
        else:
            print(f"\n💥 {len(self.failed_tests)} TEST SUITE(S) FAILED")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run GitFlow Analytics test suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Run with coverage reporting")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=[
            "unit",
            "integration",
            "tui",
            "cli",
            "config",
            "ml",
            "reports",
            "core",
            "extractors",
            "metrics",
        ],
        help="Test categories to run (default: all)",
    )
    parser.add_argument("--no-lint", action="store_true", help="Skip linting and formatting checks")

    args = parser.parse_args()

    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)

    runner = TestRunner(verbose=args.verbose, coverage=args.coverage)

    # Run tests
    tests_passed = runner.run_all_tests(args.categories)

    # Run code quality checks unless disabled
    if not args.no_lint:
        linting_passed = runner.run_linting_and_formatting()
    else:
        linting_passed = True
        print("\n⚠️  Skipping code quality checks (--no-lint)")

    # Print summary
    all_passed = runner.print_summary()

    # Exit with appropriate code
    if tests_passed and linting_passed and all_passed:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
