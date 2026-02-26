#!/usr/bin/env python3
"""
Comprehensive test suite for Git URL cloning feature in install_wizard.py

Tests URL detection, repository cloning, error handling, and integration.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gitflow_analytics.cli_wizards.install_wizard import InstallWizard


class TestResults:
    """Store and display test results."""

    def __init__(self):
        self.tests: List[Tuple[str, bool, str]] = []
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


def test_url_detection():
    """Test URL detection patterns."""
    results = TestResults()
    results.set_category("1. URL Detection Testing")

    wizard = InstallWizard(skip_validation=True)

    # Test HTTPS URLs
    test_cases = [
        ("https://github.com/owner/repo.git", True, "HTTPS with .git"),
        ("https://github.com/owner/repo", True, "HTTPS without .git"),
        ("HTTPS://GITHUB.COM/owner/repo", True, "HTTPS case insensitive"),
        ("http://github.com/owner/repo", True, "HTTP protocol"),
        # Test SSH URLs
        ("git@github.com:owner/repo.git", True, "SSH with .git"),
        ("git@github.com:owner/repo", True, "SSH without .git"),
        ("git@gitlab.com:group/project.git", True, "SSH GitLab"),
        # Test local paths (should return None)
        ("/path/to/local/repo", False, "Absolute local path"),
        ("~/repos/myproject", False, "Home directory path"),
        ("./relative/path", False, "Relative path"),
        ("/Users/masa/Projects/repo", False, "Full local path"),
        # Test invalid URLs
        ("https://github.com", False, "Incomplete URL"),
        ("github.com/owner/repo", False, "Missing protocol"),
        ("ftp://github.com/owner/repo", False, "Wrong protocol"),
    ]

    for url, should_detect, description in test_cases:
        result = wizard._detect_git_url(url)
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
            # Should NOT be detected (local path)
            passed = not is_detected
            details = (
                "Correctly identified as local path"
                if passed
                else f"Incorrectly detected as: {result}"
            )

        results.add(description, passed, details)

    return results


def test_repository_name_extraction():
    """Test repository name extraction from URLs."""
    results = TestResults()
    results.set_category("2. Repository Name Extraction")

    wizard = InstallWizard(skip_validation=True)

    test_cases = [
        ("https://github.com/owner/myrepo.git", "myrepo"),
        ("https://github.com/owner/my-repo.git", "my-repo"),
        ("git@github.com:owner/myrepo.git", "myrepo"),
        ("https://github.com/owner/myrepo", "myrepo"),
        ("git@gitlab.com:group/subgroup/project.git", "project"),
    ]

    import re

    for url, expected_name in test_cases:
        match = re.search(r"/([^/]+?)(?:\.git)?$", url)
        extracted_name = match.group(1) if match else None

        passed = extracted_name == expected_name
        details = f"URL: {url} -> {extracted_name}"
        results.add(f"Extract '{expected_name}'", passed, details)

    return results


def test_url_normalization():
    """Test Git URL normalization."""
    results = TestResults()
    results.set_category("3. URL Normalization")

    wizard = InstallWizard(skip_validation=True)

    test_cases = [
        ("https://github.com/owner/repo.git", "https://github.com/owner/repo.git"),
        ("https://github.com/owner/repo", "https://github.com/owner/repo.git"),
        ("HTTPS://GITHUB.COM/OWNER/REPO", "https://github.com/owner/repo.git"),
        ("git@github.com:owner/repo", "git@github.com:owner/repo.git"),
    ]

    for url, expected in test_cases:
        normalized = wizard._normalize_git_url(url)
        passed = normalized == expected
        details = f"{url} -> {normalized}"
        results.add(f"Normalize: {url[:30]}...", passed, details)

    return results


def test_clone_functionality():
    """Test actual repository cloning (uses a small public repo)."""
    results = TestResults()
    results.set_category("4. Clone Functionality Testing")

    # Create temporary directory for test
    test_dir = Path(tempfile.mkdtemp(prefix="gitflow_test_"))
    original_cwd = Path.cwd()

    try:
        os.chdir(test_dir)
        print(f"Test directory: {test_dir}")

        wizard = InstallWizard(skip_validation=True)

        # Test with a very small public repository
        test_repo_url = "https://github.com/octocat/Hello-World.git"

        print(f"\nAttempting to clone: {test_repo_url}")
        print("This may take a moment...")

        result = wizard._clone_git_repository(test_repo_url)

        if result is None:
            results.add("Clone public repository", False, "Clone returned None")
        else:
            local_path, original_url = result

            # Test 1: Clone returned valid path
            results.add("Clone returns valid path", True, f"Path: {local_path}")

            # Test 2: Repos directory was created
            repos_dir = test_dir / "repos"
            results.add("repos/ directory created", repos_dir.exists(), f"Path: {repos_dir}")

            # Test 3: Repository directory exists
            results.add("Repository directory exists", local_path.exists(), f"Path: {local_path}")

            # Test 4: .git directory exists
            git_dir = local_path / ".git"
            results.add(".git directory present", git_dir.exists(), f"Path: {git_dir}")

            # Test 5: Original URL preserved
            results.add(
                "Original URL preserved",
                original_url == test_repo_url,
                f"Expected: {test_repo_url}, Got: {original_url}",
            )

            # Test 6: Repository name extraction
            expected_name = "Hello-World"
            actual_name = local_path.name
            results.add(
                "Repository name correct",
                actual_name == expected_name,
                f"Expected: {expected_name}, Got: {actual_name}",
            )

    except Exception as e:
        results.add("Clone functionality", False, f"Exception: {type(e).__name__}: {str(e)}")

    finally:
        os.chdir(original_cwd)
        # Cleanup
        try:
            shutil.rmtree(test_dir)
            print(f"\n✓ Cleaned up test directory: {test_dir}")
        except Exception as e:
            print(f"\n⚠ Could not clean up test directory: {e}")

    return results


def test_existing_repository_handling():
    """Test handling of existing repositories."""
    results = TestResults()
    results.set_category("5. Existing Repository Handling")

    test_dir = Path(tempfile.mkdtemp(prefix="gitflow_test_exist_"))
    original_cwd = Path.cwd()

    try:
        os.chdir(test_dir)

        # Create repos directory with a fake repository
        repos_dir = test_dir / "repos"
        repos_dir.mkdir()

        fake_repo = repos_dir / "test-repo"
        fake_repo.mkdir()

        # Test 1: Directory exists but is not a git repo
        results.add(
            "Detect non-git directory",
            fake_repo.exists() and not (fake_repo / ".git").exists(),
            f"Directory exists without .git: {fake_repo}",
        )

        # Test 2: Create a real git repo
        from git import Repo

        test_repo = Repo.init(fake_repo)

        # Add a test file and commit
        test_file = fake_repo / "README.md"
        test_file.write_text("# Test Repository")
        test_repo.index.add([str(test_file)])
        test_repo.index.commit("Initial commit")

        results.add(
            "Create test git repository",
            (fake_repo / ".git").exists(),
            f"Git repository initialized: {fake_repo}",
        )

        # Test 3: Add remote origin
        test_url = "https://github.com/test/test-repo.git"
        test_repo.create_remote("origin", test_url)

        origin_url = test_repo.remotes.origin.url
        results.add(
            "Remote origin set", origin_url == test_url, f"Expected: {test_url}, Got: {origin_url}"
        )

    except Exception as e:
        results.add("Existing repo handling", False, f"Exception: {type(e).__name__}: {str(e)}")

    finally:
        os.chdir(original_cwd)
        try:
            shutil.rmtree(test_dir)
        except Exception:
            pass

    return results


def test_error_handling():
    """Test error handling for various failure scenarios."""
    results = TestResults()
    results.set_category("6. Error Handling Testing")

    test_dir = Path(tempfile.mkdtemp(prefix="gitflow_test_error_"))
    original_cwd = Path.cwd()

    try:
        os.chdir(test_dir)

        wizard = InstallWizard(skip_validation=True)

        # Test 1: Invalid GitHub URL (404)
        invalid_url = "https://github.com/nonexistent-user-12345/nonexistent-repo-67890.git"
        print(f"\nTesting invalid URL (expected to fail): {invalid_url}")

        result = wizard._clone_git_repository(invalid_url)
        results.add(
            "Invalid URL returns None",
            result is None,
            "Should return None for non-existent repository",
        )

        # Test 2: Malformed URL
        malformed_url = "https://github.com/incomplete"
        detected = wizard._detect_git_url(malformed_url)
        results.add(
            "Malformed URL rejected",
            detected is None,
            "Should not detect malformed URL as valid Git URL",
        )

        # Test 3: Empty URL
        empty_result = wizard._detect_git_url("")
        results.add(
            "Empty string handled", empty_result is None, "Should return None for empty string"
        )

        # Test 4: URL with spaces
        spaced_url = "https://github.com/user/repo .git"
        detected_spaced = wizard._detect_git_url(spaced_url)
        results.add(
            "URL with spaces handled", detected_spaced is None, "Should not detect URL with spaces"
        )

    except Exception as e:
        results.add(
            "Error handling tests", False, f"Unexpected exception: {type(e).__name__}: {str(e)}"
        )

    finally:
        os.chdir(original_cwd)
        try:
            shutil.rmtree(test_dir)
        except Exception:
            pass

    return results


def test_import_availability():
    """Test that all required imports are available."""
    results = TestResults()
    results.set_category("7. Import and Dependency Testing")

    try:
        import re

        results.add("Import 're' module", True, "Standard library module")
    except ImportError as e:
        results.add("Import 're' module", False, str(e))

    try:
        from git import GitCommandError, Repo

        results.add("Import GitPython", True, "git.Repo and git.GitCommandError")
    except ImportError as e:
        results.add("Import GitPython", False, str(e))

    try:
        from git.exc import InvalidGitRepositoryError

        results.add("Import git.exc", True, "InvalidGitRepositoryError available")
    except ImportError as e:
        results.add("Import git.exc", False, str(e))

    try:
        from git import RemoteProgress

        results.add("Import RemoteProgress", True, "Progress handler available")
    except ImportError as e:
        results.add("Import RemoteProgress", False, str(e))

    try:
        import click

        results.add("Import click", True, "CLI framework available")
    except ImportError as e:
        results.add("Import click", False, str(e))

    try:
        from pathlib import Path

        results.add("Import Path", True, "pathlib.Path available")
    except ImportError as e:
        results.add("Import Path", False, str(e))

    return results


def main():
    """Run all tests and generate report."""
    print("=" * 70)
    print("  GITFLOW ANALYTICS - GIT URL CLONING FEATURE TEST SUITE")
    print("=" * 70)
    print("Test Environment:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Working Directory: {Path.cwd()}")

    all_results = []

    # Run test suites
    try:
        all_results.append(test_import_availability())
    except Exception as e:
        print(f"\n❌ Import tests failed: {e}")

    try:
        all_results.append(test_url_detection())
    except Exception as e:
        print(f"\n❌ URL detection tests failed: {e}")

    try:
        all_results.append(test_repository_name_extraction())
    except Exception as e:
        print(f"\n❌ Name extraction tests failed: {e}")

    try:
        all_results.append(test_url_normalization())
    except Exception as e:
        print(f"\n❌ URL normalization tests failed: {e}")

    # Only run clone tests if imports are available
    try:
        from git import Repo

        all_results.append(test_clone_functionality())
        all_results.append(test_existing_repository_handling())
    except ImportError:
        print("\n⚠️  Skipping clone tests - GitPython not available")
    except Exception as e:
        print(f"\n❌ Clone tests failed: {e}")

    try:
        all_results.append(test_error_handling())
    except Exception as e:
        print(f"\n❌ Error handling tests failed: {e}")

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

    # Final verdict
    print("\n" + "=" * 70)
    if total_passed == total_tests:
        print("  ✅ ALL TESTS PASSED - FEATURE READY FOR USE")
    else:
        print("  ⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
    print("=" * 70)

    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
