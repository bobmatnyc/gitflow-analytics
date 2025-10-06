#!/usr/bin/env python3
"""
Direct test of branch fetching logic without repository updates.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import git

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_branch_commits():
    """Test branch analysis by directly examining repository."""

    repo_path = "/Users/masa/Projects/managed/gitflow-analytics/EWTN-test/repos/acidigital-admin"
    repo = git.Repo(repo_path)

    print(f"Repository: {repo_path}")
    print("=" * 60)

    # Get date range
    since = datetime.now(timezone.utc) - timedelta(weeks=52)
    since_str = since.strftime("%Y-%m-%d")

    print(f"Analyzing commits since: {since_str}")
    print()

    # Track all unique commits
    all_commits = set()
    branch_commits = {}

    # Analyze each branch
    for ref in repo.references:
        if isinstance(ref, git.RemoteReference) and ref.remote_head != "HEAD":
            branch_name = f"{ref.remote_name}/{ref.remote_head}"

            # Get commits on this branch
            try:
                commits = list(repo.iter_commits(ref, since=since_str))
                branch_commits[branch_name] = len(commits)

                # Add to all commits set
                for commit in commits:
                    all_commits.add(commit.hexsha)

                print(f"Branch {branch_name}: {len(commits)} commits")

                # Show sample commits
                if commits:
                    print(f"  Latest 3 commits:")
                    for commit in commits[:3]:
                        msg = commit.message.split("\n")[0][:50]
                        print(f"    - {commit.hexsha[:8]}: {msg}")

            except Exception as e:
                print(f"Branch {branch_name}: Error - {e}")

        elif isinstance(ref, git.Head):
            branch_name = ref.name

            try:
                commits = list(repo.iter_commits(ref, since=since_str))
                branch_commits[branch_name] = len(commits)

                for commit in commits:
                    all_commits.add(commit.hexsha)

                print(f"Branch {branch_name}: {len(commits)} commits")

                if commits:
                    print(f"  Latest 3 commits:")
                    for commit in commits[:3]:
                        msg = commit.message.split("\n")[0][:50]
                        print(f"    - {commit.hexsha[:8]}: {msg}")

            except Exception as e:
                print(f"Branch {branch_name}: Error - {e}")

    print()
    print("=" * 60)
    print("SUMMARY:")
    print(f"Total unique commits across all branches: {len(all_commits)}")
    print(f"Branches analyzed: {len(branch_commits)}")

    # Check for commits unique to non-main branches
    main_commits = set()
    for ref_name in ["main", "origin/main", "master", "origin/master"]:
        if ref_name in branch_commits:
            try:
                ref = (
                    repo.refs[ref_name]
                    if "/" not in ref_name
                    else repo.remotes.origin.refs[ref_name.split("/")[-1]]
                )
                for commit in repo.iter_commits(ref, since=since_str):
                    main_commits.add(commit.hexsha)
            except:
                pass

    non_main_commits = all_commits - main_commits
    print(f"Commits only on non-main branches: {len(non_main_commits)}")

    # Show which branches have unique commits
    if non_main_commits:
        print("\nBranches with unique commits:")
        for branch_name in branch_commits:
            if "main" not in branch_name and "master" not in branch_name:
                try:
                    if "/" in branch_name:
                        ref = repo.remotes.origin.refs[branch_name.split("/")[-1]]
                    else:
                        ref = repo.refs[branch_name]

                    branch_unique = 0
                    for commit in repo.iter_commits(ref, since=since_str):
                        if commit.hexsha in non_main_commits:
                            branch_unique += 1

                    if branch_unique > 0:
                        print(f"  {branch_name}: {branch_unique} unique commits")
                except:
                    pass


if __name__ == "__main__":
    test_branch_commits()
