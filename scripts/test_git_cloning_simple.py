#!/usr/bin/env python3
"""
Simplified test suite for Git URL cloning feature - tests logic directly
without requiring full module imports.
"""

import re
import sys
from pathlib import Path
from typing import Optional


class TestResults:
    """Store and display test results."""

    def __init__(self):
        self.tests = []
        self.category = ""

    def set_category(self, category: str):
        """Set current test category."""
        self.category = category
        print(f"\n{'=' * 70}")
        print(f"  {category}")
        print(f"{'=' * 70}\n")

    def add(self, test_name: str, passed: bool, details: str = ""):
        """Add a test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.tests.append((f"{self.category}: {test_name}", passed, details))
        print(f"{status} | {test_name}")
        if details:
            print(f"         {details}")

    def summary(self):
        """Print test summary."""
        passed = sum(1 for _, p, _ in self.tests if p)
        total = len(self.tests)

        print(f"\n{'=' * 70}")
        print("  TEST SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {total - passed} ❌")
        print(f"Success Rate: {(passed / total) * 100:.1f}%")

        if total - passed > 0:
            print(f"\n{'=' * 70}")
            print("  FAILED TESTS")
            print(f"{'=' * 70}")
            for name, passed, details in self.tests:
                if not passed:
                    print(f"❌ {name}")
                    if details:
                        print(f"   {details}")


# Replicate the _detect_git_url logic from install_wizard.py
def detect_git_url(input_str: str) -> Optional[str]:
    """Detect if input is a Git URL and normalize it."""
    # HTTPS URL patterns
    https_pattern = r"^https?://[^/]+/[^/]+/[^/]+(?:\.git)?$"
    # SSH URL pattern
    ssh_pattern = r"^git@[^:]+:[^/]+/[^/]+(?:\.git)?$"

    input_str = input_str.strip()

    if re.match(https_pattern, input_str, re.IGNORECASE) or re.match(ssh_pattern, input_str):
        # Ensure .git extension for consistency
        if not input_str.endswith(".git"):
            input_str = input_str + ".git"
        return input_str

    return None


def normalize_git_url(url: str) -> str:
    """Normalize Git URL for comparison."""
    url = url.lower().strip()
    if not url.endswith(".git"):
        url = url + ".git"
    return url


def extract_repo_name(git_url: str) -> Optional[str]:
    """Extract repository name from Git URL."""
    match = re.search(r"/([^/]+?)(?:\.git)?$", git_url)
    if match:
        return match.group(1)
    return None


def test_url_detection():
    """Test URL detection patterns."""
    results = TestResults()
    results.set_category("1. URL Detection Testing")

    test_cases = [
        # HTTPS URLs - should be detected
        ("https://github.com/owner/repo.git", True, "HTTPS with .git"),
        ("https://github.com/owner/repo", True, "HTTPS without .git"),
        ("HTTPS://GITHUB.COM/owner/repo", True, "HTTPS case insensitive"),
        ("http://github.com/owner/repo", True, "HTTP protocol"),
        ("https://gitlab.com/group/project.git", True, "GitLab HTTPS"),
        ("https://bitbucket.org/team/repo.git", True, "Bitbucket HTTPS"),
        # SSH URLs - should be detected
        ("git@github.com:owner/repo.git", True, "SSH with .git"),
        ("git@github.com:owner/repo", True, "SSH without .git"),
        ("git@gitlab.com:group/project.git", True, "SSH GitLab"),
        ("git@bitbucket.org:team/repo.git", True, "SSH Bitbucket"),
        # Local paths - should NOT be detected
        ("/path/to/local/repo", False, "Absolute local path"),
        ("~/repos/myproject", False, "Home directory path"),
        ("./relative/path", False, "Relative path"),
        ("/Users/masa/Projects/repo", False, "Full local path"),
        ("C:\\Users\\repo", False, "Windows path"),
        # Invalid URLs - should NOT be detected
        ("https://github.com", False, "Incomplete URL - no repo"),
        ("https://github.com/owner", False, "Incomplete URL - no repo name"),
        ("github.com/owner/repo", False, "Missing protocol"),
        ("ftp://github.com/owner/repo", False, "Wrong protocol"),
        ("https://github.com/owner/repo/extra", False, "Extra path components"),
        ("git@github.com/owner/repo", False, "Invalid SSH format (slash instead of colon)"),
    ]

    for url, should_detect, description in test_cases:
        result = detect_git_url(url)
        is_detected = result is not None

        if should_detect:
            # Should be detected as Git URL
            passed = is_detected
            if passed and result:
                # Verify .git extension was added
                if not result.endswith(".git"):
                    passed = False
                    details = f"Expected .git extension, got: {result}"
                else:
                    details = f"Detected: {result}"
            else:
                details = "Expected Git URL detection, got None"
        else:
            # Should NOT be detected (local path or invalid)
            passed = not is_detected
            details = "Correctly rejected" if passed else f"Incorrectly detected as: {result}"

        results.add(description, passed, details)

    return results


def test_repository_name_extraction():
    """Test repository name extraction from URLs."""
    results = TestResults()
    results.set_category("2. Repository Name Extraction")

    test_cases = [
        ("https://github.com/owner/myrepo.git", "myrepo"),
        ("https://github.com/owner/my-repo.git", "my-repo"),
        ("git@github.com:owner/myrepo.git", "myrepo"),
        ("https://github.com/owner/myrepo", "myrepo"),
        ("git@gitlab.com:group/subgroup/project.git", "project"),
        ("https://github.com/owner/repo-with-dashes.git", "repo-with-dashes"),
        ("git@github.com:owner/repo_underscores.git", "repo_underscores"),
    ]

    for url, expected_name in test_cases:
        extracted_name = extract_repo_name(url)
        passed = extracted_name == expected_name
        details = f"URL: {url} -> {extracted_name}"
        results.add(f"Extract '{expected_name}'", passed, details)

    return results


def test_url_normalization():
    """Test Git URL normalization."""
    results = TestResults()
    results.set_category("3. URL Normalization")

    test_cases = [
        ("https://github.com/owner/repo.git", "https://github.com/owner/repo.git"),
        ("https://github.com/owner/repo", "https://github.com/owner/repo.git"),
        ("HTTPS://GITHUB.COM/OWNER/REPO", "https://github.com/owner/repo.git"),
        ("git@github.com:owner/repo", "git@github.com:owner/repo.git"),
        ("GIT@GITHUB.COM:OWNER/REPO", "git@github.com:owner/repo.git"),
    ]

    for url, expected in test_cases:
        normalized = normalize_git_url(url)
        passed = normalized == expected
        details = f"{url} -> {normalized}"
        results.add(f"Normalize: {url[:30]}...", passed, details)

    return results


def test_regex_patterns():
    """Test regex pattern matching in detail."""
    results = TestResults()
    results.set_category("4. Regex Pattern Validation")

    # HTTPS pattern
    https_pattern = r"^https?://[^/]+/[^/]+/[^/]+(?:\.git)?$"

    https_tests = [
        ("https://github.com/owner/repo", True),
        ("http://github.com/owner/repo", True),
        ("https://github.com/owner/repo.git", True),
        ("https://a.b/c/d", True),
        ("https://github.com/owner", False),
        ("https://github.com", False),
        ("github.com/owner/repo", False),
        ("https://github.com/owner/repo/extra", False),
    ]

    for url, should_match in https_tests:
        matches = bool(re.match(https_pattern, url, re.IGNORECASE))
        passed = matches == should_match
        details = f"{url} -> {'matched' if matches else 'not matched'}"
        results.add(f"HTTPS pattern: {url[:40]}...", passed, details)

    # SSH pattern
    ssh_pattern = r"^git@[^:]+:[^/]+/[^/]+(?:\.git)?$"

    ssh_tests = [
        ("git@github.com:owner/repo", True),
        ("git@github.com:owner/repo.git", True),
        ("git@gitlab.com:group/project", True),
        ("git@a.b:c/d", True),
        ("git@github.com/owner/repo", False),  # Wrong separator
        ("github.com:owner/repo", False),  # Missing git@
        ("git@github.com:owner", False),  # Missing repo
    ]

    for url, should_match in ssh_tests:
        matches = bool(re.match(ssh_pattern, url))
        passed = matches == should_match
        details = f"{url} -> {'matched' if matches else 'not matched'}"
        results.add(f"SSH pattern: {url[:40]}...", passed, details)

    return results


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    results = TestResults()
    results.set_category("5. Edge Cases and Boundary Conditions")

    # Empty and whitespace
    empty_result = detect_git_url("")
    results.add("Empty string", empty_result is None, "Should return None")

    whitespace_result = detect_git_url("   ")
    results.add("Whitespace only", whitespace_result is None, "Should return None")

    # URL with leading/trailing whitespace
    padded_url = "  https://github.com/owner/repo  "
    padded_result = detect_git_url(padded_url)
    expected = "https://github.com/owner/repo.git"
    results.add(
        "Whitespace trimming",
        padded_result == expected,
        f"Input: '{padded_url}' -> {padded_result}",
    )

    # Very long URL
    long_url = "https://github.com/owner-with-very-long-name/repo-with-very-long-name.git"
    long_result = detect_git_url(long_url)
    results.add(
        "Long URL",
        long_result is not None and long_result.endswith(".git"),
        "Successfully detected long URL",
    )

    # URL with numbers
    numeric_url = "https://github.com/user123/repo456.git"
    numeric_result = detect_git_url(numeric_url)
    results.add("Numeric characters", numeric_result == numeric_url, f"Detected: {numeric_result}")

    # Mixed case normalization
    mixed_case = "HtTpS://GiThUb.CoM/OwNeR/RePo"
    mixed_result = detect_git_url(mixed_case)
    # Should be detected and normalized
    results.add(
        "Mixed case handling",
        mixed_result is not None,
        f"Detected and will be normalized: {mixed_result}",
    )

    return results


def test_code_quality():
    """Test code quality aspects."""
    results = TestResults()
    results.set_category("6. Code Quality Checks")

    # Check Python syntax by attempting to compile the source file
    install_wizard_path = (
        Path(__file__).parent / "src" / "gitflow_analytics" / "cli_wizards" / "install_wizard.py"
    )

    if install_wizard_path.exists():
        try:
            with open(install_wizard_path) as f:
                source_code = f.read()
            compile(source_code, str(install_wizard_path), "exec")
            results.add("Python syntax valid", True, "Source file compiles successfully")
        except SyntaxError as e:
            results.add("Python syntax valid", False, f"Syntax error at line {e.lineno}: {e.msg}")
    else:
        results.add("Python syntax valid", False, f"Source file not found: {install_wizard_path}")

    # Check that methods exist in the file
    if install_wizard_path.exists():
        with open(install_wizard_path) as f:
            source_code = f.read()

        required_methods = [
            "_detect_git_url",
            "_clone_git_repository",
            "_normalize_git_url",
            "_get_git_progress",
        ]

        for method in required_methods:
            exists = f"def {method}" in source_code
            results.add(
                f"Method '{method}' exists",
                exists,
                "Found in source code" if exists else "Not found",
            )

    return results


def main():
    """Run all tests and generate report."""
    print("=" * 70)
    print("  GITFLOW ANALYTICS - GIT URL CLONING FEATURE TEST SUITE")
    print("  (Simplified - Logic Testing Only)")
    print("=" * 70)
    print("Test Environment:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Working Directory: {Path.cwd()}")

    all_results = []

    # Run test suites
    test_functions = [
        test_url_detection,
        test_repository_name_extraction,
        test_url_normalization,
        test_regex_patterns,
        test_edge_cases,
        test_code_quality,
    ]

    for test_func in test_functions:
        try:
            all_results.append(test_func())
        except Exception as e:
            print(f"\n❌ Test suite '{test_func.__name__}' failed: {e}")
            import traceback

            traceback.print_exc()

    # Combined summary
    print("\n\n")
    print("=" * 70)
    print("  OVERALL TEST RESULTS")
    print("=" * 70)

    total_tests = sum(len(r.tests) for r in all_results)
    total_passed = sum(sum(1 for _, p, _ in r.tests if p) for r in all_results)

    print(f"\nTotal Tests Run: {total_tests}")
    print(f"Total Passed: {total_passed} ✅")
    print(f"Total Failed: {total_tests - total_passed} ❌")
    print(f"Overall Success Rate: {(total_passed / total_tests) * 100:.1f}%")

    # Show detailed summaries for each category
    print("\n" + "=" * 70)
    print("  DETAILED RESULTS BY CATEGORY")
    print("=" * 70)

    for result_set in all_results:
        result_set.summary()

    # Test coverage summary
    print("\n" + "=" * 70)
    print("  TEST COVERAGE SUMMARY")
    print("=" * 70)
    print("\n✅ Tested:")
    print("  • URL detection (HTTPS, SSH, local paths)")
    print("  • Repository name extraction from URLs")
    print("  • URL normalization and case handling")
    print("  • Regex pattern validation")
    print("  • Edge cases (empty strings, whitespace, long URLs)")
    print("  • Code quality (syntax, method existence)")

    print("\n⚠️  Not Tested (requires full environment):")
    print("  • Actual git clone operations")
    print("  • Existing repository handling")
    print("  • Network error handling")
    print("  • File system error handling")
    print("  • Progress indicator functionality")

    print("\n💡 Recommendations:")
    print("  • Run integration tests in staging environment")
    print("  • Test with real GitHub/GitLab repositories")
    print("  • Verify authentication handling manually")
    print("  • Test disk space error scenarios")

    # Final verdict
    print("\n" + "=" * 70)
    if total_passed == total_tests:
        print("  ✅ ALL LOGIC TESTS PASSED - READY FOR INTEGRATION TESTING")
    else:
        print("  ⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
    print("=" * 70)

    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
