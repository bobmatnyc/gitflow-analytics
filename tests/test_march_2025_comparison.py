#!/usr/bin/env python3
"""
Test script to pull all commit data for March 2025 and compare DB vs Git.

This script will:
1. Use the existing config.yaml to pull March 2025 data
2. Query the database for stored commits
3. Query Git directly for the same period
4. Generate a detailed comparison report
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add the gitflow-analytics package to the path
sys.path.insert(0, "/Users/masa/Projects/managed/gitflow-analytics/src")

import git

from gitflow_analytics.config import Config
from gitflow_analytics.core.analyzer import GitAnalyzer
from gitflow_analytics.core.cache import GitAnalysisCache


class March2025Tester:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = Config.from_file(config_path)
        self.cache_dir = Path("/Users/masa/Clients/EWTN/gfa/.gitflow-cache")
        self.cache = GitAnalysisCache(self.cache_dir)

        # March 2025 date range
        self.start_date = datetime(2025, 3, 1, tzinfo=timezone.utc)
        self.end_date = datetime(2025, 3, 31, 23, 59, 59, tzinfo=timezone.utc)

        # Test repository (pick one that exists)
        self.test_repo_path = Path("/Users/masa/Clients/EWTN/gfa/repos/ewtn-cms")
        if not self.test_repo_path.exists():
            # Fallback to first available repo
            repos_dir = Path("/Users/masa/Clients/EWTN/gfa/repos")
            available_repos = [
                d for d in repos_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
            ]
            if available_repos:
                self.test_repo_path = available_repos[0]
                print(f"Using fallback repository: {self.test_repo_path.name}")
            else:
                raise ValueError("No repositories found in /Users/masa/Clients/EWTN/gfa/repos")

        self.results = {"db_commits": [], "git_commits": [], "comparison": {}, "discrepancies": []}

    def run_gitflow_analysis(self):
        """Run GitFlow Analytics to pull March 2025 data."""
        print("🔄 Running GitFlow Analytics for March 2025...")

        # Create a temporary config for March 2025 analysis
        {
            "repositories": [
                {
                    "name": self.test_repo_path.name,
                    "path": str(self.test_repo_path),
                    "project_key": self.test_repo_path.name.upper().replace("-", "_"),
                }
            ],
            "analysis": {
                "start_date": "2025-03-01",
                "end_date": "2025-03-31",
                "include_merges": True,
                "branch_strategy": "all",
            },
            "cache": {"directory": str(self.cache_dir)},
        }

        # Run the analysis using GitAnalyzer directly
        try:
            analyzer = GitAnalyzer(
                cache_dir=self.cache_dir,
                identity_resolver=None,
                story_point_extractor=None,
                ticket_extractor=None,
            )

            # Analyze the repository for March 2025
            commits = analyzer.analyze_repository(
                repo_path=self.test_repo_path,
                weeks_back=None,  # We'll filter by date instead
                branch_strategy="all",
            )

            # Filter commits to March 2025
            march_commits = []
            for commit in commits:
                commit_date = commit.get("timestamp")
                if isinstance(commit_date, str):
                    commit_date = datetime.fromisoformat(commit_date.replace("Z", "+00:00"))
                elif not isinstance(commit_date, datetime):
                    continue

                if self.start_date <= commit_date <= self.end_date:
                    march_commits.append(commit)

            print(f"✅ GitFlow analysis complete. Found {len(march_commits)} commits in March 2025")
            return march_commits

        except Exception as e:
            print(f"❌ Error running GitFlow analysis: {e}")
            return []

    def query_database_commits(self):
        """Query the database for March 2025 commits."""
        print("🔍 Querying database for March 2025 commits...")

        db_path = self.cache_dir / "gitflow_cache.db"
        if not db_path.exists():
            print(f"❌ Database not found at {db_path}")
            return []

        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            query = """
            SELECT 
                commit_hash,
                author_name,
                author_email,
                message,
                timestamp,
                branch,
                is_merge,
                files_changed,
                insertions,
                deletions,
                complexity_delta,
                story_points,
                ticket_references,
                repo_path
            FROM cached_commits 
            WHERE repo_path = ? 
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
            """

            cursor = conn.execute(
                query,
                (str(self.test_repo_path), self.start_date.isoformat(), self.end_date.isoformat()),
            )

            db_commits = []
            for row in cursor.fetchall():
                commit_dict = dict(row)
                # Parse JSON fields
                if commit_dict["ticket_references"]:
                    try:
                        commit_dict["ticket_references"] = json.loads(
                            commit_dict["ticket_references"]
                        )
                    except:
                        commit_dict["ticket_references"] = []
                db_commits.append(commit_dict)

            conn.close()
            print(f"✅ Found {len(db_commits)} commits in database for March 2025")
            self.results["db_commits"] = db_commits
            return db_commits

        except Exception as e:
            print(f"❌ Error querying database: {e}")
            return []

    def query_git_commits(self):
        """Query Git directly for March 2025 commits."""
        print("🔍 Querying Git directly for March 2025 commits...")

        try:
            repo = git.Repo(self.test_repo_path)

            # Get all commits in March 2025
            git_commits = []

            # Use git log with date filtering
            commits = list(repo.iter_commits(all=True, since=self.start_date, until=self.end_date))

            for commit in commits:
                try:
                    # Get file changes
                    files_changed = []
                    insertions = 0
                    deletions = 0

                    if commit.parents:
                        diffs = commit.diff(commit.parents[0])
                        for diff in diffs:
                            if diff.b_path:
                                files_changed.append(diff.b_path)
                            elif diff.a_path:
                                files_changed.append(diff.a_path)

                    # Get stats
                    try:
                        stats = commit.stats.total
                        insertions = stats.get("insertions", 0)
                        deletions = stats.get("deletions", 0)
                    except:
                        pass

                    git_commit = {
                        "commit_hash": commit.hexsha,
                        "author_name": commit.author.name,
                        "author_email": commit.author.email,
                        "message": commit.message.strip(),
                        "timestamp": commit.committed_datetime,
                        "is_merge": len(commit.parents) > 1,
                        "files_changed": len(files_changed),
                        "files_changed_list": files_changed,
                        "insertions": insertions,
                        "deletions": deletions,
                        "parents": [p.hexsha for p in commit.parents],
                    }

                    git_commits.append(git_commit)

                except Exception as e:
                    print(f"⚠️  Error processing commit {commit.hexsha[:8]}: {e}")
                    continue

            print(f"✅ Found {len(git_commits)} commits in Git for March 2025")
            self.results["git_commits"] = git_commits
            return git_commits

        except Exception as e:
            print(f"❌ Error querying Git: {e}")
            return []

    def compare_commits(self):
        """Compare database commits with Git commits."""
        print("🔍 Comparing database vs Git commits...")

        db_commits = self.results["db_commits"]
        git_commits = self.results["git_commits"]

        # Create lookup dictionaries
        db_by_hash = {c["commit_hash"]: c for c in db_commits}
        git_by_hash = {c["commit_hash"]: c for c in git_commits}

        # Find discrepancies
        db_only = set(db_by_hash.keys()) - set(git_by_hash.keys())
        git_only = set(git_by_hash.keys()) - set(db_by_hash.keys())
        common = set(db_by_hash.keys()) & set(git_by_hash.keys())

        discrepancies = []

        # Check commits only in DB
        for hash in db_only:
            discrepancies.append(
                {
                    "type": "db_only",
                    "commit_hash": hash,
                    "message": "Commit exists in database but not in Git",
                    "db_commit": db_by_hash[hash],
                }
            )

        # Check commits only in Git
        for hash in git_only:
            discrepancies.append(
                {
                    "type": "git_only",
                    "commit_hash": hash,
                    "message": "Commit exists in Git but not in database",
                    "git_commit": git_by_hash[hash],
                }
            )

        # Check common commits for data differences
        for hash in common:
            db_commit = db_by_hash[hash]
            git_commit = git_by_hash[hash]

            differences = []

            # Compare key fields
            if db_commit["author_name"] != git_commit["author_name"]:
                differences.append(
                    f"Author name: DB='{db_commit['author_name']}' vs Git='{git_commit['author_name']}'"
                )

            if db_commit["author_email"] != git_commit["author_email"]:
                differences.append(
                    f"Author email: DB='{db_commit['author_email']}' vs Git='{git_commit['author_email']}'"
                )

            if db_commit["message"].strip() != git_commit["message"].strip():
                differences.append("Message differs")

            if db_commit["is_merge"] != git_commit["is_merge"]:
                differences.append(
                    f"Merge flag: DB={db_commit['is_merge']} vs Git={git_commit['is_merge']}"
                )

            if db_commit["files_changed"] != git_commit["files_changed"]:
                differences.append(
                    f"Files changed count: DB={db_commit['files_changed']} vs Git={git_commit['files_changed']}"
                )

            if db_commit["insertions"] != git_commit["insertions"]:
                differences.append(
                    f"Insertions: DB={db_commit['insertions']} vs Git={git_commit['insertions']}"
                )

            if db_commit["deletions"] != git_commit["deletions"]:
                differences.append(
                    f"Deletions: DB={db_commit['deletions']} vs Git={git_commit['deletions']}"
                )

            if differences:
                discrepancies.append(
                    {
                        "type": "data_mismatch",
                        "commit_hash": hash,
                        "message": "Data differences found",
                        "differences": differences,
                        "db_commit": db_commit,
                        "git_commit": git_commit,
                    }
                )

        self.results["comparison"] = {
            "total_db_commits": len(db_commits),
            "total_git_commits": len(git_commits),
            "db_only_count": len(db_only),
            "git_only_count": len(git_only),
            "common_count": len(common),
            "discrepancies_count": len(discrepancies),
        }

        self.results["discrepancies"] = discrepancies

        print("✅ Comparison complete:")
        print(f"   📊 DB commits: {len(db_commits)}")
        print(f"   📊 Git commits: {len(git_commits)}")
        print(f"   📊 Common: {len(common)}")
        print(f"   📊 DB only: {len(db_only)}")
        print(f"   📊 Git only: {len(git_only)}")
        print(
            f"   📊 Data mismatches: {len([d for d in discrepancies if d['type'] == 'data_mismatch'])}"
        )

    def generate_report(self):
        """Generate a detailed comparison report."""
        print("📝 Generating comparison report...")

        report_path = Path("march_2025_comparison_report.md")

        with open(report_path, "w") as f:
            f.write("# March 2025 Commit Data Comparison Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Repository**: {self.test_repo_path.name}\n")
            f.write("**Period**: March 1-31, 2025\n\n")

            # Summary
            comp = self.results["comparison"]
            f.write("## Summary\n\n")
            f.write(f"- **Database commits**: {comp['total_db_commits']}\n")
            f.write(f"- **Git commits**: {comp['total_git_commits']}\n")
            f.write(f"- **Common commits**: {comp['common_count']}\n")
            f.write(f"- **DB-only commits**: {comp['db_only_count']}\n")
            f.write(f"- **Git-only commits**: {comp['git_only_count']}\n")
            f.write(f"- **Data discrepancies**: {comp['discrepancies_count']}\n\n")

            # Accuracy calculation
            if comp["total_git_commits"] > 0:
                accuracy = (comp["common_count"] / comp["total_git_commits"]) * 100
                f.write(f"**Data Accuracy**: {accuracy:.1f}% (commits in both DB and Git)\n\n")

            # Discrepancies
            if self.results["discrepancies"]:
                f.write("## Discrepancies\n\n")

                for i, disc in enumerate(
                    self.results["discrepancies"][:20], 1
                ):  # Limit to first 20
                    f.write(f"### {i}. {disc['type'].replace('_', ' ').title()}\n")
                    f.write(f"**Commit**: `{disc['commit_hash'][:8]}`\n")
                    f.write(f"**Issue**: {disc['message']}\n")

                    if disc["type"] == "data_mismatch":
                        f.write("**Differences**:\n")
                        for diff in disc["differences"]:
                            f.write(f"- {diff}\n")

                    f.write("\n")

                if len(self.results["discrepancies"]) > 20:
                    f.write(
                        f"... and {len(self.results['discrepancies']) - 20} more discrepancies\n\n"
                    )

            # Sample commits
            if self.results["db_commits"]:
                f.write("## Sample Database Commits\n\n")
                for commit in self.results["db_commits"][:5]:
                    f.write(
                        f"- `{commit['commit_hash'][:8]}` by {commit['author_name']}: {commit['message'][:60]}...\n"
                    )
                f.write("\n")

            if self.results["git_commits"]:
                f.write("## Sample Git Commits\n\n")
                for commit in self.results["git_commits"][:5]:
                    f.write(
                        f"- `{commit['commit_hash'][:8]}` by {commit['author_name']}: {commit['message'][:60]}...\n"
                    )

        print(f"✅ Report generated: {report_path}")

    def run_test(self):
        """Run the complete test."""
        print("🚀 Starting March 2025 DB vs Git comparison test...")
        print(f"📁 Repository: {self.test_repo_path}")
        print(f"📅 Period: {self.start_date.date()} to {self.end_date.date()}")
        print()

        # Step 1: Run GitFlow analysis to ensure data is in DB
        self.run_gitflow_analysis()

        # Step 2: Query database
        self.query_database_commits()

        # Step 3: Query Git directly
        self.query_git_commits()

        # Step 4: Compare
        self.compare_commits()

        # Step 5: Generate report
        self.generate_report()

        print("\n🎉 Test complete!")


if __name__ == "__main__":
    config_path = "/Users/masa/Clients/EWTN/gfa/config.yaml"

    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    tester = March2025Tester(config_path)
    tester.run_test()
