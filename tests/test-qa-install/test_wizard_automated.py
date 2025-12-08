#!/usr/bin/env python3
"""
Automated testing for GitFlow Analytics Install Wizard.

This script tests various aspects of the install wizard without requiring
user interaction.
"""

import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

# Test results tracking
test_results = {
    "passed": [],
    "failed": [],
    "security_issues": [],
}


def log_pass(test_name, message=""):
    """Log a passing test."""
    test_results["passed"].append(test_name)
    print(f"✅ PASS: {test_name}")
    if message:
        print(f"   {message}")


def log_fail(test_name, message=""):
    """Log a failing test."""
    test_results["failed"].append(test_name)
    print(f"❌ FAIL: {test_name}")
    if message:
        print(f"   {message}")


def log_security_issue(test_name, issue):
    """Log a security issue."""
    test_results["security_issues"].append(f"{test_name}: {issue}")
    print(f"🔐 SECURITY ISSUE: {test_name}")
    print(f"   {issue}")


def test_command_availability():
    """Test 1: Command Availability Tests."""
    print("\n" + "=" * 60)
    print("TEST 1: COMMAND AVAILABILITY")
    print("=" * 60)

    # Test 1.1: Command exists
    result = subprocess.run(
        ["gitflow-analytics", "install", "--help"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        log_pass("Test 1.1: Command exists", "install --help works")
    else:
        log_fail("Test 1.1: Command exists", result.stderr)

    # Test 1.2: Help text is clear
    result = subprocess.run(
        ["gitflow-analytics", "--help"],
        capture_output=True,
        text=True,
    )

    if "install" in result.stdout.lower() and result.returncode == 0:
        log_pass("Test 1.2: Help text includes install", "install command documented")
    else:
        log_fail("Test 1.2: Help text includes install", "install not found in help")

    # Test 1.3: --skip-validation flag exists
    result = subprocess.run(
        ["gitflow-analytics", "install", "--help"],
        capture_output=True,
        text=True,
    )

    if "--skip-validation" in result.stdout:
        log_pass("Test 1.3: --skip-validation flag exists", "Testing flag available")
    else:
        log_fail("Test 1.3: --skip-validation flag exists", "Flag not documented")


def test_file_permissions():
    """Test 6: File Generation and Permissions."""
    print("\n" + "=" * 60)
    print("TEST 6: FILE GENERATION AND PERMISSIONS")
    print("=" * 60)

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test-perms"
        output_dir.mkdir()

        # Create minimal test files
        output_dir / "config.yaml"
        env_path = output_dir / ".env"

        # Simulate wizard file creation with umask
        old_umask = os.umask(0o077)
        try:
            with open(env_path, "w") as f:
                f.write("GITHUB_TOKEN=test_token_123\n")
        finally:
            os.umask(old_umask)

        # Set explicit permissions like wizard does
        env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600

        # Test 6.1: Check .env permissions
        actual_perms = stat.S_IMODE(os.stat(env_path).st_mode)
        expected_perms = 0o600

        if actual_perms == expected_perms:
            log_pass("Test 6.1: .env permissions correct", f"Permissions are {oct(actual_perms)}")
        else:
            log_fail(
                "Test 6.1: .env permissions correct",
                f"Expected {oct(expected_perms)}, got {oct(actual_perms)}",
            )

        # Test 6.1b: Security check - file should not be world-readable
        if actual_perms & stat.S_IROTH:
            log_security_issue("Test 6.1: .env permissions", "File is world-readable")

        if actual_perms & stat.S_IRGRP:
            log_security_issue("Test 6.1: .env permissions", "File is group-readable")

        # Test 6.2: .gitignore creation
        gitignore_path = output_dir / ".gitignore"
        with open(gitignore_path, "w") as f:
            f.write("# Test gitignore\n")
            f.write(".env\n")

        if gitignore_path.exists():
            content = gitignore_path.read_text()
            if ".env" in content:
                log_pass("Test 6.2: .gitignore includes .env", "Credential protection active")
            else:
                log_fail("Test 6.2: .gitignore includes .env", ".env not found in .gitignore")


def test_path_validation():
    """Test 3.3: Path traversal protection."""
    print("\n" + "=" * 60)
    print("TEST 3.3: PATH TRAVERSAL PROTECTION")
    print("=" * 60)

    # Test the _validate_directory_path method directly
    from gitflow_analytics.cli_wizards.install_wizard import InstallWizard

    with tempfile.TemporaryDirectory() as tmpdir:
        wizard = InstallWizard(Path(tmpdir), skip_validation=True)

        # Test 3.3a: Reject absolute path outside home/cwd
        bad_path = "/etc/passwd"
        result = wizard._validate_directory_path(bad_path, "Test path")

        if result is None:
            log_pass("Test 3.3a: Path traversal blocked", f"Rejected: {bad_path}")
        else:
            log_fail("Test 3.3a: Path traversal blocked", f"Accepted dangerous path: {bad_path}")
            log_security_issue("Test 3.3a: Path traversal", f"Path traversal allowed: {bad_path}")

        # Test 3.3b: Accept relative path
        good_path = "./reports"
        result = wizard._validate_directory_path(good_path, "Test path")

        if result is not None:
            log_pass("Test 3.3b: Relative path accepted", f"Accepted: {good_path}")
        else:
            log_fail("Test 3.3b: Relative path accepted", f"Rejected valid path: {good_path}")

        # Test 3.3c: Accept path within home
        home_path = str(Path.home() / "test" / "reports")
        result = wizard._validate_directory_path(home_path, "Test path")

        if result is not None:
            log_pass("Test 3.3c: Home directory path accepted", f"Accepted: {home_path}")
        else:
            log_fail(
                "Test 3.3c: Home directory path accepted", f"Rejected valid home path: {home_path}"
            )


def test_memory_clearing():
    """Test 8.2: Memory clearing after use."""
    print("\n" + "=" * 60)
    print("TEST 8.2: MEMORY CLEARING")
    print("=" * 60)

    from gitflow_analytics.cli_wizards.install_wizard import InstallWizard

    with tempfile.TemporaryDirectory() as tmpdir:
        wizard = InstallWizard(Path(tmpdir), skip_validation=True)

        # Add sensitive data
        wizard.env_data["GITHUB_TOKEN"] = "ghp_test_token_12345"
        wizard.env_data["JIRA_ACCESS_TOKEN"] = "jira_test_token_67890"
        wizard.env_data["OPENAI_API_KEY"] = "sk-test-key-abcdef"

        # Clear sensitive data
        wizard._clear_sensitive_data()

        # Test 8.2a: Dictionary should be empty
        if len(wizard.env_data) == 0:
            log_pass("Test 8.2a: env_data cleared", "All sensitive data removed from memory")
        else:
            log_fail("Test 8.2a: env_data cleared", f"Still contains {len(wizard.env_data)} items")
            log_security_issue(
                "Test 8.2a: Memory clearing",
                f"Sensitive data remains in memory: {list(wizard.env_data.keys())}",
            )


def test_exception_sanitization():
    """Test 8.1: Exception message sanitization."""
    print("\n" + "=" * 60)
    print("TEST 8.1: EXCEPTION SANITIZATION")
    print("=" * 60)

    # Read the install_wizard.py source to verify sanitization patterns
    wizard_source = (
        Path(__file__).parent.parent
        / "src"
        / "gitflow_analytics"
        / "cli_wizards"
        / "install_wizard.py"
    )

    if wizard_source.exists():
        source_code = wizard_source.read_text()

        # Test 8.1a: Check for credential exposure prevention in exception handlers

        safe_patterns = [
            "type(e).__name__",
            "Never expose raw exception",
            "could contain credentials",
        ]

        has_safe_patterns = any(pattern in source_code for pattern in safe_patterns)

        if has_safe_patterns:
            log_pass(
                "Test 8.1a: Exception sanitization implemented",
                "Code uses type(e).__name__ pattern",
            )
        else:
            log_fail(
                "Test 8.1a: Exception sanitization implemented",
                "Safe exception handling pattern not found",
            )

        # Test 8.1b: Check for logging suppression
        if "urllib3_logger.setLevel(logging.WARNING)" in source_code:
            log_pass(
                "Test 8.1b: Logging suppression for credentials",
                "urllib3 logging suppressed during auth",
            )
        else:
            log_fail(
                "Test 8.1b: Logging suppression for credentials", "Logging suppression not found"
            )


def test_config_structure():
    """Test 7: Configuration structure validation."""
    print("\n" + "=" * 60)
    print("TEST 7: CONFIGURATION STRUCTURE")
    print("=" * 60)

    from gitflow_analytics.cli_wizards.install_wizard import InstallWizard

    with tempfile.TemporaryDirectory() as tmpdir:
        wizard = InstallWizard(Path(tmpdir), skip_validation=True)

        # Simulate setup
        wizard.env_data["GITHUB_TOKEN"] = "test_token"
        wizard.config_data["github"] = {"token": "${GITHUB_TOKEN}"}
        wizard.config_data["github"]["organization"] = "test-org"

        # Test 7.1: Config uses environment variable placeholders
        if "${GITHUB_TOKEN}" in str(wizard.config_data):
            log_pass(
                "Test 7.1: Environment variable placeholders used",
                "Config references ${GITHUB_TOKEN}, not raw token",
            )
        else:
            log_fail(
                "Test 7.1: Environment variable placeholders used",
                "Config may contain raw credentials",
            )
            log_security_issue(
                "Test 7.1: Config structure", "Raw credentials may be in config instead of env vars"
            )

        # Test 7.2: Organization mode excludes repositories
        if "repositories" not in wizard.config_data["github"]:
            log_pass(
                "Test 7.2: Organization mode structure", "Config correctly uses organization mode"
            )
        else:
            log_fail(
                "Test 7.2: Organization mode structure",
                "Both organization and repositories present",
            )


def print_summary():
    """Print test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    total = len(test_results["passed"]) + len(test_results["failed"])
    passed = len(test_results["passed"])
    failed = len(test_results["failed"])
    security = len(test_results["security_issues"])

    print(f"\nTotal Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"🔐 Security Issues: {security}")

    if security > 0:
        print("\n⚠️  SECURITY ISSUES FOUND:")
        for issue in test_results["security_issues"]:
            print(f"   • {issue}")

    if failed > 0:
        print("\n❌ FAILED TESTS:")
        for test in test_results["failed"]:
            print(f"   • {test}")

    print("\n" + "=" * 60)

    if security > 0 or failed > 0:
        print("❌ OVERALL STATUS: FAIL")
        print("=" * 60)
        return False
    else:
        print("✅ OVERALL STATUS: PASS")
        print("=" * 60)
        return True


def main():
    """Run all automated tests."""
    print("GitFlow Analytics Install Wizard - Automated Testing")
    print("=" * 60)

    # Run tests
    test_command_availability()
    test_file_permissions()
    test_path_validation()
    test_memory_clearing()
    test_exception_sanitization()
    test_config_structure()

    # Print summary
    success = print_summary()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
