#!/usr/bin/env python3
"""Debug git statistics extraction to investigate zero lines changed issue."""

import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import git
from git import Repo


def test_git_stats_on_current_repo():
    """Test git statistics on the current repo to verify the extraction logic."""
    print("🔍 Testing Git Statistics Extraction")
    print("=" * 50)
    
    # Use the current gitflow-analytics repo
    repo_path = Path("/Users/masa/Projects/managed/gitflow-analytics")
    
    try:
        repo = Repo(repo_path)
        print(f"✅ Successfully opened repository: {repo_path}")
        
        # Get recent commits (last 10)
        commits = list(repo.iter_commits('HEAD', max_count=10))
        print(f"📊 Found {len(commits)} recent commits")
        
        print("\n🧪 Testing commit statistics extraction:")
        print("-" * 60)
        
        for i, commit in enumerate(commits[:5], 1):  # Test first 5 commits
            print(f"\n#{i} Commit: {commit.hexsha[:8]}")
            print(f"   Message: {commit.message.strip()[:80]}...")
            print(f"   Author: {commit.author.name} <{commit.author.email}>")
            
            # Test the same extraction logic used in analyzer.py
            stats = commit.stats.total
            print(f"   Raw stats type: {type(stats)}")
            print(f"   Raw stats: {stats}")
            
            # Test both dict and object access patterns
            if isinstance(stats, dict):
                files_count = stats.get("files", 0)
                insertions = stats.get("insertions", 0)
                deletions = stats.get("deletions", 0)
                print(f"   Dict access - Files: {files_count}, +{insertions}, -{deletions}")
            else:
                files_count = getattr(stats, 'files', 0)
                insertions = getattr(stats, 'insertions', 0)
                deletions = getattr(stats, 'deletions', 0)
                print(f"   Attr access - Files: {files_count}, +{insertions}, -{deletions}")
                
            # Also test manual diff calculation
            parent = commit.parents[0] if commit.parents else None
            manual_files = 0
            manual_insertions = 0
            manual_deletions = 0
            
            try:
                for diff in commit.diff(parent):
                    manual_files += 1
                    if diff.diff:
                        diff_text = (
                            diff.diff
                            if isinstance(diff.diff, str)
                            else diff.diff.decode("utf-8", errors="ignore")
                        )
                        for line in diff_text.split("\n"):
                            if line.startswith("+") and not line.startswith("+++"):
                                manual_insertions += 1
                            elif line.startswith("-") and not line.startswith("---"):
                                manual_deletions += 1
                
                print(f"   Manual calc - Files: {manual_files}, +{manual_insertions}, -{manual_deletions}")
                
                # Check if they match
                stats_total = (insertions + deletions) if isinstance(stats, dict) else (getattr(stats, 'insertions', 0) + getattr(stats, 'deletions', 0))
                manual_total = manual_insertions + manual_deletions
                match_status = "✅ MATCH" if stats_total == manual_total else "❌ MISMATCH"
                print(f"   Status: {match_status} (stats total: {stats_total}, manual: {manual_total})")
                
            except Exception as e:
                print(f"   ⚠️  Manual calculation failed: {e}")
                
        return True
        
    except Exception as e:
        print(f"❌ Failed to test git stats: {e}")
        return False


def test_date_range_calculation():
    """Test date range calculation for activity gaps."""
    print("\n\n📅 Testing Date Range Calculation")
    print("=" * 50)
    
    # Test the date range for 20-week analysis
    end_date = datetime(2025, 8, 8, tzinfo=timezone.utc)
    weeks_back = 20
    start_date = end_date - timedelta(weeks=weeks_back)
    
    print(f"Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Human readable: {start_date.strftime('%b %d')} to {end_date.strftime('%b %d')}")
    
    # Calculate weekly buckets
    print(f"\nWeekly buckets:")
    current_date = start_date
    week_num = 1
    
    while current_date <= end_date:
        week_end = min(current_date + timedelta(days=6), end_date)
        print(f"  Week {week_num:2d}: {current_date.strftime('%b %d')} - {week_end.strftime('%b %d')}")
        current_date += timedelta(days=7)
        week_num += 1
        if week_num > 21:  # Safety check
            break
    
    return True


def test_activity_gaps():
    """Test detection of activity gaps in commit history."""
    print("\n\n⏰ Testing Activity Gap Detection")
    print("=" * 50)
    
    repo_path = Path("/Users/masa/Projects/managed/gitflow-analytics")
    
    try:
        repo = Repo(repo_path)
        
        # Get commits from the last 20 weeks
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(weeks=20)
        
        commits = list(repo.iter_commits('HEAD', since=start_date))
        print(f"Found {len(commits)} commits in the last 20 weeks")
        
        if not commits:
            print("⚠️  No commits found in date range")
            return False
            
        # Group commits by week
        weekly_commits = {}
        
        for commit in commits:
            commit_date = commit.committed_datetime
            if commit_date.tzinfo is None:
                commit_date = commit_date.replace(tzinfo=timezone.utc)
            elif commit_date.tzinfo != timezone.utc:
                commit_date = commit_date.astimezone(timezone.utc)
                
            # Calculate week number from start date
            days_diff = (commit_date - start_date).days
            week_num = days_diff // 7 + 1
            
            if week_num not in weekly_commits:
                weekly_commits[week_num] = []
            weekly_commits[week_num].append(commit)
        
        # Find gaps
        print(f"\nWeekly activity distribution:")
        total_weeks = 20
        inactive_weeks = 0
        
        for week in range(1, total_weeks + 1):
            if week in weekly_commits:
                count = len(weekly_commits[week])
                print(f"  Week {week:2d}: {count:2d} commits")
            else:
                print(f"  Week {week:2d}:  0 commits ❌")
                inactive_weeks += 1
        
        print(f"\n📊 Activity Summary:")
        print(f"   Active weeks: {total_weeks - inactive_weeks}/{total_weeks}")
        print(f"   Inactive weeks: {inactive_weeks}")
        print(f"   Activity rate: {((total_weeks - inactive_weeks) / total_weeks * 100):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to analyze activity gaps: {e}")
        return False


if __name__ == "__main__":
    print("🚀 GitFlow Analytics - Git Statistics Diagnostics")
    print("=" * 60)
    
    success = True
    success &= test_git_stats_on_current_repo()
    success &= test_date_range_calculation()
    success &= test_activity_gaps()
    
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")