#!/usr/bin/env python3
"""
Final verification test to show how different branch strategies affect commit analysis.
This test demonstrates the importance of the 'all' strategy for complete coverage.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import tempfile
import git

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gitflow_analytics.core.analyzer import GitAnalyzer
from gitflow_analytics.core.cache import GitAnalysisCache


def count_unique_commits_in_repo(repo_path: str, since: datetime) -> int:
    """Count total unique commits in repository across all branches."""
    repo = git.Repo(repo_path)
    all_commits = set()
    since_str = since.strftime("%Y-%m-%d")

    # Check all references
    for ref in repo.references:
        try:
            for commit in repo.iter_commits(ref, since=since_str):
                all_commits.add(commit.hexsha)
        except:
            pass

    return len(all_commits)


def test_repository(repo_path: str, repo_name: str):
    """Test a single repository with all strategies."""

    print(f"\n{'='*80}")
    print(f"TESTING: {repo_name}")
    print(f"Path: {repo_path}")
    print('='*80)

    if not Path(repo_path).exists():
        print(f"Repository not found!")
        return

    # Time range
    since = datetime.now(timezone.utc) - timedelta(weeks=52)

    # First, count actual commits in repo
    actual_commits = count_unique_commits_in_repo(repo_path, since)
    print(f"\nActual unique commits in repository: {actual_commits}")

    # Test each strategy
    strategies = {
        'main_only': 'Analyzes only main/master branch',
        'smart': 'Analyzes active branches with filtering',
        'all': 'Analyzes all branches comprehensively'
    }

    results = {}

    for strategy, description in strategies.items():
        print(f"\n{'-'*60}")
        print(f"Strategy: {strategy.upper()}")
        print(f"Description: {description}")
        print(f"{'-'*60}")

        with tempfile.TemporaryDirectory() as cache_dir:
            # Create cache
            cache = GitAnalysisCache(Path(cache_dir))

            # Configure analyzer
            branch_config = {
                'strategy': strategy,
                'active_days_threshold': 30,
                'max_branches_per_repo': 100,
                'update_repo': False  # Don't try to fetch
            }

            # Suppress update attempts
            analyzer = GitAnalyzer(
                cache=cache,
                batch_size=1000,
                branch_analysis_config=branch_config
            )

            # Override the _update_repository method to prevent fetch attempts
            analyzer._update_repository = lambda repo: None

            try:
                # Analyze repository
                commits = analyzer.analyze_repository(Path(repo_path), since)
                results[strategy] = len(commits)

                print(f"Commits found: {len(commits)}")

                # Get unique branches represented
                branches = set()
                for commit in commits:
                    if 'branch' in commit:
                        branches.add(commit['branch'])

                print(f"Branches analyzed: {len(branches)}")
                if branches and len(branches) <= 5:
                    for branch in sorted(branches):
                        print(f"  - {branch}")

            except Exception as e:
                print(f"Error: {e}")
                results[strategy] = 0

    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print('='*60)
    print(f"Repository: {repo_name}")
    print(f"Actual unique commits: {actual_commits}")
    print()

    for strategy in ['main_only', 'smart', 'all']:
        if strategy in results:
            count = results[strategy]
            coverage = (count / actual_commits * 100) if actual_commits > 0 else 0
            missed = actual_commits - count

            print(f"{strategy.upper():15} {count:4} commits ({coverage:5.1f}% coverage)")
            if missed > 0:
                print(f"{'':15} ⚠️  Missed {missed} commits!")

    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print('='*60)

    if 'all' in results and 'main_only' in results:
        missed_by_main = actual_commits - results['main_only']
        if missed_by_main > 0:
            print(f"⚠️  Using 'main_only' strategy would miss {missed_by_main} commits")
            print(f"   This could exclude work from {missed_by_main} commits on feature branches")

    if 'all' in results:
        all_coverage = (results['all'] / actual_commits * 100) if actual_commits > 0 else 0
        if all_coverage >= 95:
            print(f"✅ The 'all' strategy provides {all_coverage:.1f}% coverage")
            print(f"   This ensures comprehensive analysis of all development work")
        else:
            print(f"⚠️  Even 'all' strategy only achieved {all_coverage:.1f}% coverage")
            print(f"   There may be configuration issues or branch access problems")


def main():
    """Test multiple repositories to demonstrate branch strategy importance."""

    test_repos = [
        ("/Users/masa/Projects/managed/gitflow-analytics/EWTN-test/repos/acidigital-admin", "acidigital-admin"),
        ("/Users/masa/Projects/managed/gitflow-analytics/EWTN-test/repos/ewtn-com", "ewtn-com"),
    ]

    print("BRANCH STRATEGY VERIFICATION TEST")
    print("This test demonstrates how different branch strategies affect commit coverage")

    for repo_path, repo_name in test_repos:
        test_repository(repo_path, repo_name)

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print('='*80)
    print("The 'all' strategy is recommended as the default because:")
    print("1. It ensures complete coverage of all development work")
    print("2. It doesn't miss commits on feature/hotfix branches")
    print("3. It provides the most accurate view of team productivity")
    print("\nThe 'main_only' and 'smart' strategies can be used when:")
    print("- Performance is critical and accuracy can be sacrificed")
    print("- You only care about production-ready code on main branch")
    print("- You have a strict gitflow where all work goes through main")


if __name__ == "__main__":
    main()