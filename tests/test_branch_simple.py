#!/usr/bin/env python3
"""
Simple test to verify branch analysis is working correctly.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gitflow_analytics.core.analyzer import GitAnalyzer
from gitflow_analytics.core.cache import GitAnalysisCache


def test_branch_analysis():
    """Test branch analysis on hosanna-ui repository."""

    repo_path = "/Users/masa/Projects/managed/gitflow-analytics/EWTN-test/repos/hosanna-ui"

    if not Path(repo_path).exists():
        print(f"Repository not found: {repo_path}")
        return

    print(f"Testing repository: {repo_path}")
    print("="*60)

    # Test with different strategies
    strategies = ['main_only', 'smart', 'all']
    results = {}

    for strategy in strategies:
        print(f"\nTesting {strategy} strategy...")

        # Create temporary cache
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = GitAnalysisCache(Path(cache_dir))

            # Create analyzer
            branch_config = {
                'strategy': strategy,
                'active_days_threshold': 30,
                'max_branches_per_repo': 50
            }

            analyzer = GitAnalyzer(
                cache=cache,
                batch_size=1000,
                branch_analysis_config=branch_config
            )

            # Analyze last 52 weeks
            since = datetime.now(timezone.utc) - timedelta(weeks=52)

            # Analyze repository
            commits = analyzer.analyze_repository(Path(repo_path), since)

            results[strategy] = len(commits)
            print(f"  Found {len(commits)} commits")

            # Show sample commits
            if commits and len(commits) > 0:
                print(f"  Sample commits:")
                for commit in commits[:3]:
                    msg = commit.get('message', '')[:50] if commit.get('message') else 'No message'
                    print(f"    - {commit['hash'][:8]}: {msg}...")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    for strategy, count in results.items():
        print(f"  {strategy}: {count} commits")

    # Calculate differences
    if 'all' in results and 'main_only' in results:
        print(f"\nCommits on non-main branches: {results['all'] - results['main_only']}")

    if 'all' in results and 'smart' in results:
        diff = results['all'] - results['smart']
        if diff > 0:
            print(f"Commits missed by smart strategy: {diff}")
        elif diff < 0:
            print(f"Smart strategy found {-diff} more commits than 'all' (possible duplicates)")
        else:
            print("Smart and all strategies found the same number of commits")


if __name__ == "__main__":
    test_branch_analysis()