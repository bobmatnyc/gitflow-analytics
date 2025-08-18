"""
Checksum validation for GitFlow Analytics.

This module provides validation functionality to ensure data integrity
by comparing processed commits against actual git history.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from git import Repo

logger = logging.getLogger(__name__)


class ChecksumValidator:
    """Validates processed data against actual git repository state."""

    def __init__(self):
        """Initialize the checksum validator."""
        pass

    def validate_repository_commits(
        self,
        repo_path: Path,
        processed_commits: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        branch_patterns: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """
        Validate processed commits against actual git history.
        
        Args:
            repo_path: Path to the git repository
            processed_commits: Number of commits processed by GitFlow Analytics
            start_date: Start date for analysis period
            end_date: End date for analysis period
            branch_patterns: Branch patterns to include in validation
            
        Returns:
            Dictionary with validation results
        """
        try:
            repo = Repo(repo_path)
            
            # Get all commits in the repository
            total_commits_all_branches = self._count_all_commits(repo)
            
            # Get commits in date range if specified
            commits_in_range = None
            if start_date and end_date:
                commits_in_range = self._count_commits_in_range(
                    repo, start_date, end_date, branch_patterns
                )
            
            # Get commits by branch
            branch_breakdown = self._get_branch_commit_breakdown(
                repo, start_date, end_date
            )
            
            # Calculate validation results
            validation_result = {
                "repository_path": str(repo_path),
                "validation_timestamp": datetime.now().isoformat(),
                "total_commits_all_branches": total_commits_all_branches,
                "processed_commits": processed_commits,
                "commits_in_date_range": commits_in_range,
                "branch_breakdown": branch_breakdown,
                "validation_status": self._determine_validation_status(
                    processed_commits, commits_in_range, total_commits_all_branches
                ),
                "analysis_period": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                },
            }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to validate repository {repo_path}: {e}")
            return {
                "repository_path": str(repo_path),
                "validation_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "validation_status": "ERROR",
            }

    def _count_all_commits(self, repo: Repo) -> int:
        """Count all commits across all branches."""
        try:
            # Use git log --all to count all commits
            commits = list(repo.iter_commits("--all"))
            return len(commits)
        except Exception as e:
            logger.warning(f"Failed to count all commits: {e}")
            return 0

    def _count_commits_in_range(
        self,
        repo: Repo,
        start_date: datetime,
        end_date: datetime,
        branch_patterns: Optional[List[str]] = None,
    ) -> int:
        """Count commits in the specified date range."""
        try:
            # Build git log command for date range
            if branch_patterns:
                # Use specific branches
                refs = []
                for pattern in branch_patterns:
                    if pattern == "main" or pattern == "master":
                        refs.append(pattern)
                    else:
                        # For now, just use main branches
                        refs.append("main")
                ref_spec = " ".join(refs) if refs else "--all"
            else:
                ref_spec = "--all"
            
            commits = list(
                repo.iter_commits(
                    ref_spec,
                    since=start_date,
                    until=end_date,
                )
            )
            return len(commits)
        except Exception as e:
            logger.warning(f"Failed to count commits in range: {e}")
            return 0

    def _get_branch_commit_breakdown(
        self,
        repo: Repo,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """Get commit count breakdown by branch."""
        branch_breakdown = {}
        
        try:
            # Get all branches (local and remote)
            all_refs = list(repo.refs)
            
            for ref in all_refs:
                try:
                    if start_date and end_date:
                        commits = list(
                            repo.iter_commits(
                                ref.name,
                                since=start_date,
                                until=end_date,
                            )
                        )
                    else:
                        commits = list(repo.iter_commits(ref.name))
                    
                    branch_breakdown[ref.name] = len(commits)
                except Exception as e:
                    logger.debug(f"Failed to count commits for {ref.name}: {e}")
                    branch_breakdown[ref.name] = 0
                    
        except Exception as e:
            logger.warning(f"Failed to get branch breakdown: {e}")
            
        return branch_breakdown

    def _determine_validation_status(
        self,
        processed_commits: int,
        commits_in_range: Optional[int],
        total_commits: int,
    ) -> str:
        """Determine the validation status based on commit counts."""
        if commits_in_range is not None:
            # We have a date range, so compare against that
            if processed_commits == commits_in_range:
                return "VALID"
            elif processed_commits < commits_in_range:
                return "UNDER_PROCESSED"
            else:
                return "OVER_PROCESSED"
        else:
            # No date range, compare against total
            if processed_commits == total_commits:
                return "VALID"
            elif processed_commits < total_commits:
                return "UNDER_PROCESSED"
            else:
                return "OVER_PROCESSED"

    def generate_validation_report(
        self, validation_results: List[Dict[str, any]]
    ) -> str:
        """Generate a human-readable validation report."""
        report_lines = []
        report_lines.append("# GitFlow Analytics Validation Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        for result in validation_results:
            repo_name = Path(result["repository_path"]).name
            report_lines.append(f"## Repository: {repo_name}")
            report_lines.append(f"Path: {result['repository_path']}")
            
            if "error" in result:
                report_lines.append(f"❌ **Status**: ERROR - {result['error']}")
            else:
                status = result["validation_status"]
                status_emoji = {
                    "VALID": "✅",
                    "UNDER_PROCESSED": "⚠️",
                    "OVER_PROCESSED": "⚠️",
                    "ERROR": "❌",
                }.get(status, "❓")
                
                report_lines.append(f"{status_emoji} **Status**: {status}")
                report_lines.append(f"- Processed commits: {result['processed_commits']}")
                report_lines.append(f"- Total commits (all branches): {result['total_commits_all_branches']}")
                
                if result.get("commits_in_date_range") is not None:
                    report_lines.append(f"- Commits in analysis period: {result['commits_in_date_range']}")
                
                if result.get("analysis_period", {}).get("start_date"):
                    period = result["analysis_period"]
                    report_lines.append(f"- Analysis period: {period['start_date']} to {period['end_date']}")
                
                # Branch breakdown (top 5 branches)
                if result.get("branch_breakdown"):
                    sorted_branches = sorted(
                        result["branch_breakdown"].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    report_lines.append("- Top branches by commit count:")
                    for branch, count in sorted_branches:
                        report_lines.append(f"  - {branch}: {count} commits")
            
            report_lines.append("")
        
        return "\n".join(report_lines)
