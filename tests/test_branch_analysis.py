#!/usr/bin/env python3
"""
Direct test of branch analysis to verify all branches are being captured.
Tests GitAnalyzer with different branch strategies.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Set
import git
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gitflow_analytics.core.analyzer import GitAnalyzer
from gitflow_analytics.core.cache import GitAnalysisCache


def count_commits_by_branch(repo_path: str, commits: List[Dict]) -> Dict[str, int]:
    """Count how many commits are on each branch."""
    repo = git.Repo(repo_path)
    branch_counts = {}

    # Count commits that exist on each branch
    for branch in repo.heads:
        branch_name = branch.name
        branch_counts[branch_name] = 0

        try:
            # Get all commit hashes on this branch
            branch_commits = set()
            for commit in repo.iter_commits(branch.name):
                branch_commits.add(commit.hexsha[:7])  # Use short hash like analyzer

            # Count how many of our analyzed commits are on this branch
            for analyzed_commit in commits:
                if analyzed_commit["hash"] in branch_commits:
                    branch_counts[branch_name] += 1
        except Exception as e:
            print(f"  Error counting commits for branch {branch_name}: {e}")

    # Also check remote branches
    for ref in repo.remotes.origin.refs:
        if ref.remote_head != "HEAD":
            branch_name = f"origin/{ref.remote_head}"
            branch_counts[branch_name] = 0

            try:
                branch_commits = set()
                for commit in repo.iter_commits(ref.name):
                    branch_commits.add(commit.hexsha[:7])

                for analyzed_commit in commits:
                    if analyzed_commit["hash"] in branch_commits:
                        branch_counts[branch_name] += 1
            except Exception as e:
                print(f"  Error counting commits for branch {branch_name}: {e}")

    return branch_counts


def test_branch_strategy(repo_path: str, strategy: str, weeks: int = 52):
    """Test a specific branch analysis strategy."""

    print(f"\n{'='*80}")
    print(f"Testing: {Path(repo_path).name}")
    print(f"Strategy: {strategy}")
    print(f"Period: {weeks} weeks")
    print("=" * 80)

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as cache_dir:
        cache = GitAnalysisCache(Path(cache_dir))

        # Create analyzer with specified strategy
        branch_config = {
            "strategy": strategy,
            "active_days_threshold": 30,
            "max_branches_per_repo": 50,  # High limit for testing
        }

        analyzer = GitAnalyzer(cache=cache, batch_size=1000, branch_analysis_config=branch_config)

        # Calculate since date
        since = datetime.now(timezone.utc) - timedelta(weeks=weeks)

        # Analyze repository
        print(f"Analyzing repository...")
        commits = analyzer.analyze_repository(Path(repo_path), since)

        print(f"\nTotal commits found: {len(commits)}")

        # Count unique authors
        authors = set(c.get("author_name", "Unknown") for c in commits)
        print(f"Unique authors: {len(authors)}")

        # Show sample of commits
        if commits:
            print(f"\nFirst 5 commits:")
            for commit in commits[:5]:
                msg = commit.get("message", "")[:60] if commit.get("message") else "No message"
                print(f"  - {commit['hash']}: {msg}...")

        # Count commits by branch
        branch_counts = count_commits_by_branch(repo_path, commits)

        print(f"\nCommits found on each branch:")
        for branch, count in sorted(branch_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {branch}: {count} commits")

        # Show branches with no commits in the period
        empty_branches = [b for b, c in branch_counts.items() if c == 0]
        if empty_branches:
            print(f"\nBranches with no commits in period ({len(empty_branches)} branches):")
            for branch in sorted(empty_branches)[:10]:  # Show first 10
                print(f"  - {branch}")
            if len(empty_branches) > 10:
                print(f"  ... and {len(empty_branches) - 10} more")

        return len(commits), branch_counts


def compare_strategies(repo_path: str):
    """Compare all three branch strategies."""

    if not Path(repo_path).exists():
        print(f"Repository not found: {repo_path}")
        return

    # Get repository info
    repo = git.Repo(repo_path)
    print(f"\n{'#'*80}")
    print(f"Repository: {Path(repo_path).name}")
    print(
        f"Total branches: {len(list(repo.heads))} local, {len(list(repo.remotes.origin.refs))} remote"
    )
    print(f"{'#'*80}")

    results = {}

    # Test each strategy
    for strategy in ["main_only", "smart", "all"]:
        count, branches = test_branch_strategy(
            repo_path, strategy, weeks=52
        )  # Use 52 weeks for more data
        results[strategy] = {
            "total": count,
            "branches_with_commits": sum(1 for c in branches.values() if c > 0),
        }

    # Summary comparison
    print(f"\n{'='*80}")
    print(f"STRATEGY COMPARISON SUMMARY")
    print(f"{'='*80}")

    for strategy, data in results.items():
        print(f"\n{strategy.upper()} strategy:")
        print(f"  Total commits: {data['total']}")
        print(f"  Branches with commits: {data['branches_with_commits']}")

    # Calculate differences
    if "all" in results and "main_only" in results:
        missed_by_main = results["all"]["total"] - results["main_only"]["total"]
        print(f"\nCommits missed by 'main_only' strategy: {missed_by_main}")

    if "all" in results and "smart" in results:
        missed_by_smart = results["all"]["total"] - results["smart"]["total"]
        print(f"Commits missed by 'smart' strategy: {missed_by_smart}")


def main():
    """Run tests on EWTN repositories."""

    # Test repositories - these should have multiple branches
    test_repos = [
        "/Users/masa/Projects/managed/gitflow-analytics/EWTN-test/repos/hosanna-ui",
        "/Users/masa/Projects/managed/gitflow-analytics/EWTN-test/repos/ewtn-com",
        "/Users/masa/Projects/managed/gitflow-analytics/EWTN-test/repos/aciprensa-rebuild",
    ]

    for repo_path in test_repos:
        if Path(repo_path).exists():
            compare_strategies(repo_path)
        else:
            print(f"\nRepository not found: {repo_path}")
            print("Please ensure the repository is cloned.")


if __name__ == "__main__":
    main()
