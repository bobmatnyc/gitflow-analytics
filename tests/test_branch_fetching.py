#!/usr/bin/env python3
"""
Direct test of git fetching functionality to verify branch analysis.
Tests DataFetcher with different branch configurations.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Set
import git

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gitflow_analytics.config import Config
from gitflow_analytics.core.data_fetcher import DataFetcher


def test_branch_fetching(repo_path: str, branch_patterns: List[str] = None, weeks: int = 8):
    """Test fetching commits from a repository with different branch patterns."""

    print(f"\n{'='*80}")
    print(f"Testing repository: {repo_path}")
    print(f"Branch patterns: {branch_patterns if branch_patterns else 'ALL BRANCHES'}")
    print(f"Time period: {weeks} weeks")
    print('='*80)

    # Create a minimal config
    config = Config({
        'repositories': {
            'test_repo': {
                'path': repo_path,
                'branch_patterns': branch_patterns or []
            }
        },
        'analysis': {
            'period_weeks': weeks,
            'identity': {
                'fuzzy_threshold': 0.85,
                'auto_analysis': False
            }
        },
        'output': {
            'directory': 'test_output'
        }
    })

    # Initialize DataFetcher
    fetcher = DataFetcher(config)

    # Open the repository to get branch information
    repo = git.Repo(repo_path)

    # Get all branches (both local and remote)
    all_branches = set()
    for ref in repo.references:
        if 'origin/' in ref.name:
            branch_name = ref.name.replace('origin/', '')
            if branch_name != 'HEAD':
                all_branches.add(branch_name)

    print(f"\nAvailable branches in repository:")
    for branch in sorted(all_branches):
        print(f"  - {branch}")

    # Fetch commits
    print(f"\nFetching commits...")
    commits = fetcher.fetch_commits()

    # Analyze commits by branch
    branch_commits: Dict[str, Set[str]] = {}
    total_commits = 0

    for project_key, project_commits in commits.items():
        print(f"\nProject: {project_key}")
        print(f"Total commits: {len(project_commits)}")
        total_commits += len(project_commits)

        # Group commits by branch
        for commit in project_commits:
            # Get branches containing this commit
            commit_branches = []
            try:
                # Get the actual git commit
                git_commit = repo.commit(commit['hash'])

                # Find which branches contain this commit
                for branch in repo.heads:
                    if repo.is_ancestor(git_commit, branch.commit):
                        branch_name = branch.name
                        if branch_name not in branch_commits:
                            branch_commits[branch_name] = set()
                        branch_commits[branch_name].add(commit['hash'])
                        commit_branches.append(branch_name)

                # Also check remote branches
                for ref in repo.remotes.origin.refs:
                    if ref.remote_head != 'HEAD':
                        try:
                            if repo.is_ancestor(git_commit, ref.commit):
                                branch_name = ref.remote_head
                                if branch_name not in branch_commits:
                                    branch_commits[branch_name] = set()
                                branch_commits[branch_name].add(commit['hash'])
                                if branch_name not in commit_branches:
                                    commit_branches.append(branch_name)
                        except:
                            pass
            except Exception as e:
                # Commit might not exist in current state
                pass

    # Print branch analysis
    print(f"\n{'-'*40}")
    print("Commits by branch:")
    print(f"{'-'*40}")

    for branch in sorted(branch_commits.keys()):
        count = len(branch_commits[branch])
        print(f"  {branch}: {count} commits")

    print(f"\n{'-'*40}")
    print(f"Total unique commits: {total_commits}")
    print(f"Branches analyzed: {len(branch_commits)}")

    # Check which branches were actually fetched
    if branch_patterns:
        matched_branches = []
        for pattern in branch_patterns:
            for branch in all_branches:
                if pattern in branch or branch == pattern:
                    matched_branches.append(branch)

        print(f"\nBranches matching patterns {branch_patterns}:")
        for branch in matched_branches:
            print(f"  - {branch}")

        missed_branches = all_branches - set(matched_branches)
        if missed_branches:
            print(f"\nBranches NOT analyzed (filtered out):")
            for branch in sorted(missed_branches):
                print(f"  - {branch}")

    return total_commits, branch_commits


def main():
    """Run tests on EWTN repositories."""

    # Test repositories - update these paths as needed
    test_repos = [
        "/Users/masa/Projects/managed/gitflow-analytics/repos/ewtn-plus-foundation",
        "/Users/masa/Projects/managed/gitflow-analytics/repos/hosanna-ui",
    ]

    for repo_path in test_repos:
        if not Path(repo_path).exists():
            print(f"\nSkipping {repo_path} - not found")
            continue

        # Test 1: Without branch patterns (should get ALL branches)
        print(f"\n{'#'*80}")
        print(f"TEST 1: Fetching ALL branches")
        print(f"{'#'*80}")
        total1, branches1 = test_branch_fetching(repo_path, branch_patterns=None, weeks=8)

        # Test 2: With specific branch patterns
        print(f"\n{'#'*80}")
        print(f"TEST 2: Fetching only main/master branches")
        print(f"{'#'*80}")
        total2, branches2 = test_branch_fetching(repo_path, branch_patterns=['main', 'master'], weeks=8)

        # Test 3: With develop pattern
        print(f"\n{'#'*80}")
        print(f"TEST 3: Fetching develop branches")
        print(f"{'#'*80}")
        total3, branches3 = test_branch_fetching(repo_path, branch_patterns=['develop'], weeks=8)

        # Summary
        print(f"\n{'='*80}")
        print(f"SUMMARY for {Path(repo_path).name}")
        print(f"{'='*80}")
        print(f"All branches: {total1} commits across {len(branches1)} branches")
        print(f"Main/master only: {total2} commits across {len(branches2)} branches")
        print(f"Develop only: {total3} commits across {len(branches3)} branches")
        print(f"Difference: {total1 - total2} commits on non-main branches")


if __name__ == "__main__":
    main()