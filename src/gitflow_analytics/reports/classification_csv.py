"""CSV report methods for ClassificationReportGenerator.

Extracted from classification_writer.py to keep file sizes manageable.
Contains detailed CSV, developer breakdown, repository analysis,
confidence analysis, temporal patterns, and classification matrix reports.
"""

import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ClassificationCsvMixin:
    """Mixin providing CSV report generation methods for ClassificationReportGenerator.

    Attributes (expected from host class):
        output_directory: Path  -- where to write reports
        confidence_threshold: float  -- threshold for high-confidence classification
        min_commits_for_analysis: int  -- minimum commits to include a developer
    """

    # These are declared on the host class; typing stubs to satisfy mypy / IDEs.
    output_directory: "Path"
    confidence_threshold: float
    min_commits_for_analysis: int

    def _get_timestamp(self) -> str:  # provided by host; stub to avoid import loops
        raise NotImplementedError

    def generate_detailed_csv_report(
        self, classified_commits: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate detailed CSV report with all commit information.

        Args:
            classified_commits: List of classified commits
            metadata: Optional analysis metadata

        Returns:
            Path to generated detailed CSV file
        """
        output_path = self.output_directory / f"classification_detailed_{self._get_timestamp()}.csv"

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            headers = [
                "commit_hash",
                "date",
                "author",
                "canonical_author",
                "repository",
                "predicted_class",
                "confidence",
                "is_reliable",
                "message_preview",
                "files_changed",
                "insertions",
                "deletions",
                "lines_changed",
                "primary_language",
                "primary_activity",
                "is_multilingual",
                "branch",
                "project_key",
                "ticket_references",
            ]
            writer.writerow(headers)

            for commit in classified_commits:
                file_analysis = commit.get("file_analysis_summary", {})

                row = [
                    commit.get("hash", "")[:12],
                    commit.get("timestamp", "").strftime("%Y-%m-%d %H:%M:%S")
                    if commit.get("timestamp")
                    else "",
                    commit.get("author_name", ""),
                    commit.get("canonical_author_name", commit.get("author_name", "")),
                    commit.get("repository", ""),
                    commit.get("predicted_class", ""),
                    f"{commit.get('classification_confidence', 0):.3f}",
                    commit.get("is_reliable_prediction", False),
                    commit.get("message", "")[:100].replace("\n", " "),
                    commit.get("files_changed", 0),
                    commit.get("insertions", 0),
                    commit.get("deletions", 0),
                    commit.get("insertions", 0) + commit.get("deletions", 0),
                    file_analysis.get("primary_language", ""),
                    file_analysis.get("primary_activity", ""),
                    file_analysis.get("is_multilingual", False),
                    commit.get("branch", ""),
                    commit.get("project_key", ""),
                    len(commit.get("ticket_references", [])),
                ]
                writer.writerow(row)

        logger.info(f"Detailed CSV report generated: {output_path}")
        return str(output_path)

    def generate_developer_breakdown_report(
        self, classified_commits: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate per-developer classification breakdown CSV."""
        output_path = (
            self.output_directory / f"classification_by_developer_{self._get_timestamp()}.csv"
        )

        developer_stats = defaultdict(
            lambda: {
                "total_commits": 0,
                "classifications": Counter(),
                "confidence_scores": [],
                "repositories": set(),
                "total_lines_changed": 0,
                "avg_files_per_commit": 0,
                "commit_dates": [],
            }
        )

        for commit in classified_commits:
            author = commit.get("canonical_author_name", commit.get("author_name", "unknown"))
            stats = developer_stats[author]

            stats["total_commits"] += 1
            stats["classifications"][commit.get("predicted_class", "unknown")] += 1

            if "classification_confidence" in commit:
                stats["confidence_scores"].append(commit["classification_confidence"])

            stats["repositories"].add(commit.get("repository", "unknown"))
            stats["total_lines_changed"] += commit.get("insertions", 0) + commit.get("deletions", 0)
            stats["avg_files_per_commit"] += commit.get("files_changed", 0)

            if commit.get("timestamp"):
                stats["commit_dates"].append(commit["timestamp"])

        for _author, stats in developer_stats.items():
            if stats["total_commits"] > 0:
                stats["avg_confidence"] = (
                    sum(stats["confidence_scores"]) / len(stats["confidence_scores"])
                    if stats["confidence_scores"]
                    else 0
                )
                stats["avg_files_per_commit"] = (
                    stats["avg_files_per_commit"] / stats["total_commits"]
                )
                stats["avg_lines_per_commit"] = (
                    stats["total_lines_changed"] / stats["total_commits"]
                )
                stats["primary_classification"] = (
                    stats["classifications"].most_common(1)[0][0]
                    if stats["classifications"]
                    else "unknown"
                )
                stats["classification_diversity"] = len(stats["classifications"])
                stats["repository_count"] = len(stats["repositories"])

                if stats["commit_dates"]:
                    date_range = max(stats["commit_dates"]) - min(stats["commit_dates"])
                    stats["activity_span_days"] = date_range.days
                else:
                    stats["activity_span_days"] = 0

        filtered_developers = {
            k: v
            for k, v in developer_stats.items()
            if v["total_commits"] >= self.min_commits_for_analysis
        }

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(["Developer Classification Analysis"])
            writer.writerow(["Total Developers:", len(developer_stats)])
            writer.writerow(
                [f"Developers with \u2265{self.min_commits_for_analysis} commits:", len(filtered_developers)]
            )
            writer.writerow([])

            headers = [
                "developer",
                "total_commits",
                "primary_classification",
                "classification_diversity",
                "avg_confidence",
                "high_confidence_ratio",
                "repository_count",
                "repositories",
                "avg_files_per_commit",
                "avg_lines_per_commit",
                "activity_span_days",
            ]

            all_classifications: set = set()
            for stats in filtered_developers.values():
                all_classifications.update(stats["classifications"].keys())

            classification_headers = [f"{cls}_count" for cls in sorted(all_classifications)]
            headers.extend(classification_headers)
            writer.writerow(headers)

            sorted_developers = sorted(
                filtered_developers.items(), key=lambda x: x[1]["total_commits"], reverse=True
            )

            for author, stats in sorted_developers:
                high_confidence_count = sum(
                    1 for score in stats["confidence_scores"] if score >= self.confidence_threshold
                )
                high_confidence_ratio = (
                    high_confidence_count / len(stats["confidence_scores"])
                    if stats["confidence_scores"]
                    else 0
                )

                row = [
                    author,
                    stats["total_commits"],
                    stats["primary_classification"],
                    stats["classification_diversity"],
                    f"{stats['avg_confidence']:.3f}",
                    f"{high_confidence_ratio:.3f}",
                    stats["repository_count"],
                    "; ".join(sorted(stats["repositories"])),
                    f"{stats['avg_files_per_commit']:.1f}",
                    f"{stats['avg_lines_per_commit']:.0f}",
                    stats["activity_span_days"],
                ]

                for cls in sorted(all_classifications):
                    row.append(stats["classifications"].get(cls, 0))

                writer.writerow(row)

        logger.info(f"Developer breakdown report generated: {output_path}")
        return str(output_path)

    def generate_repository_analysis_report(
        self, classified_commits: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate per-repository classification analysis CSV."""
        output_path = (
            self.output_directory / f"classification_by_repository_{self._get_timestamp()}.csv"
        )

        repo_stats = defaultdict(
            lambda: {
                "total_commits": 0,
                "classifications": Counter(),
                "developers": set(),
                "confidence_scores": [],
                "total_lines_changed": 0,
                "languages": Counter(),
                "activities": Counter(),
            }
        )

        for commit in classified_commits:
            repo = commit.get("repository", "unknown")
            stats = repo_stats[repo]

            stats["total_commits"] += 1
            stats["classifications"][commit.get("predicted_class", "unknown")] += 1
            stats["developers"].add(
                commit.get("canonical_author_name", commit.get("author_name", "unknown"))
            )

            if "classification_confidence" in commit:
                stats["confidence_scores"].append(commit["classification_confidence"])

            stats["total_lines_changed"] += commit.get("insertions", 0) + commit.get("deletions", 0)

            file_analysis = commit.get("file_analysis_summary", {})
            if file_analysis.get("primary_language"):
                stats["languages"][file_analysis["primary_language"]] += 1
            if file_analysis.get("primary_activity"):
                stats["activities"][file_analysis["primary_activity"]] += 1

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(["Repository Classification Analysis"])
            writer.writerow(["Total Repositories:", len(repo_stats)])
            writer.writerow([])

            headers = [
                "repository",
                "total_commits",
                "developer_count",
                "primary_classification",
                "avg_confidence",
                "avg_lines_per_commit",
                "primary_language",
                "primary_activity",
                "classification_diversity",
                "language_diversity",
            ]
            writer.writerow(headers)

            sorted_repos = sorted(
                repo_stats.items(), key=lambda x: x[1]["total_commits"], reverse=True
            )

            for repo, stats in sorted_repos:
                avg_confidence = (
                    sum(stats["confidence_scores"]) / len(stats["confidence_scores"])
                    if stats["confidence_scores"]
                    else 0
                )
                avg_lines = (
                    stats["total_lines_changed"] / stats["total_commits"]
                    if stats["total_commits"] > 0
                    else 0
                )

                primary_class = (
                    stats["classifications"].most_common(1)[0][0]
                    if stats["classifications"]
                    else "unknown"
                )
                primary_lang = (
                    stats["languages"].most_common(1)[0][0] if stats["languages"] else "unknown"
                )
                primary_activity = (
                    stats["activities"].most_common(1)[0][0] if stats["activities"] else "unknown"
                )

                row = [
                    repo,
                    stats["total_commits"],
                    len(stats["developers"]),
                    primary_class,
                    f"{avg_confidence:.3f}",
                    f"{avg_lines:.0f}",
                    primary_lang,
                    primary_activity,
                    len(stats["classifications"]),
                    len(stats["languages"]),
                ]
                writer.writerow(row)

        logger.info(f"Repository analysis report generated: {output_path}")
        return str(output_path)

    def generate_confidence_analysis_report(
        self, classified_commits: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate confidence score analysis CSV."""
        output_path = (
            self.output_directory
            / f"classification_confidence_analysis_{self._get_timestamp()}.csv"
        )

        confidence_scores = [c.get("classification_confidence", 0) for c in classified_commits]

        confidence_by_class: Dict[str, list] = defaultdict(list)
        for commit in classified_commits:
            class_type = commit.get("predicted_class", "unknown")
            confidence = commit.get("classification_confidence", 0)
            confidence_by_class[class_type].append(confidence)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(["Classification Confidence Analysis"])
            writer.writerow([])

            if confidence_scores:
                writer.writerow(["Overall Confidence Statistics"])
                writer.writerow(["Metric", "Value"])
                writer.writerow(["Total Predictions", len(confidence_scores)])
                writer.writerow(
                    ["Average Confidence", f"{sum(confidence_scores) / len(confidence_scores):.3f}"]
                )
                writer.writerow(["Minimum Confidence", f"{min(confidence_scores):.3f}"])
                writer.writerow(["Maximum Confidence", f"{max(confidence_scores):.3f}"])

                very_high = sum(1 for s in confidence_scores if s >= 0.9)
                high = sum(1 for s in confidence_scores if 0.8 <= s < 0.9)
                medium = sum(1 for s in confidence_scores if 0.6 <= s < 0.8)
                low = sum(1 for s in confidence_scores if 0.4 <= s < 0.6)
                very_low = sum(1 for s in confidence_scores if s < 0.4)
                n = len(confidence_scores)

                writer.writerow(["Very High (\u22650.9)", f"{very_high} ({(very_high/n)*100:.1f}%)"])
                writer.writerow(["High (0.8-0.9)", f"{high} ({(high/n)*100:.1f}%)"])
                writer.writerow(["Medium (0.6-0.8)", f"{medium} ({(medium/n)*100:.1f}%)"])
                writer.writerow(["Low (0.4-0.6)", f"{low} ({(low/n)*100:.1f}%)"])
                writer.writerow(["Very Low (<0.4)", f"{very_low} ({(very_low/n)*100:.1f}%)"])
                writer.writerow([])

            writer.writerow(["Confidence by Classification Type"])
            writer.writerow(
                ["Classification", "Count", "Avg Confidence", "Min", "Max", "High Confidence Count"]
            )

            for class_type, scores in sorted(confidence_by_class.items()):
                if scores:
                    avg_conf = sum(scores) / len(scores)
                    high_conf_count = sum(1 for s in scores if s >= self.confidence_threshold)

                    writer.writerow(
                        [
                            class_type,
                            len(scores),
                            f"{avg_conf:.3f}",
                            f"{min(scores):.3f}",
                            f"{max(scores):.3f}",
                            f"{high_conf_count} ({(high_conf_count/len(scores))*100:.1f}%)",
                        ]
                    )

        logger.info(f"Confidence analysis report generated: {output_path}")
        return str(output_path)

    def generate_temporal_patterns_report(
        self, classified_commits: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate temporal patterns analysis CSV."""
        output_path = (
            self.output_directory / f"classification_temporal_patterns_{self._get_timestamp()}.csv"
        )

        daily_stats = defaultdict(
            lambda: {
                "total_commits": 0,
                "classifications": Counter(),
                "developers": set(),
                "confidence_scores": [],
            }
        )

        for commit in classified_commits:
            if commit.get("timestamp"):
                date_key = commit["timestamp"].date()
                stats = daily_stats[date_key]

                stats["total_commits"] += 1
                stats["classifications"][commit.get("predicted_class", "unknown")] += 1
                stats["developers"].add(
                    commit.get("canonical_author_name", commit.get("author_name", "unknown"))
                )

                if "classification_confidence" in commit:
                    stats["confidence_scores"].append(commit["classification_confidence"])

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(["Temporal Classification Patterns"])
            writer.writerow([])

            all_classifications: set = set()
            for stats in daily_stats.values():
                all_classifications.update(stats["classifications"].keys())

            headers = ["date", "total_commits", "developer_count", "avg_confidence"]
            headers.extend([f"{cls}_count" for cls in sorted(all_classifications)])
            writer.writerow(headers)

            for date, stats in sorted(daily_stats.items()):
                avg_confidence = (
                    sum(stats["confidence_scores"]) / len(stats["confidence_scores"])
                    if stats["confidence_scores"]
                    else 0
                )

                row = [
                    date.isoformat(),
                    stats["total_commits"],
                    len(stats["developers"]),
                    f"{avg_confidence:.3f}",
                ]

                for cls in sorted(all_classifications):
                    row.append(stats["classifications"].get(cls, 0))

                writer.writerow(row)

        logger.info(f"Temporal patterns report generated: {output_path}")
        return str(output_path)

    def generate_classification_matrix_report(
        self, classified_commits: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate classification distribution matrix CSV."""
        output_path = self.output_directory / f"classification_matrix_{self._get_timestamp()}.csv"

        class_counts = Counter(c.get("predicted_class", "unknown") for c in classified_commits)

        dev_class_matrix: Dict[str, Counter] = defaultdict(Counter)
        repo_class_matrix: Dict[str, Counter] = defaultdict(Counter)
        lang_class_matrix: Dict[str, Counter] = defaultdict(Counter)

        for commit in classified_commits:
            class_type = commit.get("predicted_class", "unknown")
            developer = commit.get("canonical_author_name", commit.get("author_name", "unknown"))
            repository = commit.get("repository", "unknown")

            dev_class_matrix[developer][class_type] += 1
            repo_class_matrix[repository][class_type] += 1

            file_analysis = commit.get("file_analysis_summary", {})
            language = file_analysis.get("primary_language", "unknown")
            lang_class_matrix[language][class_type] += 1

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(["Classification Distribution Matrix"])
            writer.writerow([])

            writer.writerow(["Overall Classification Distribution"])
            writer.writerow(["Classification", "Count", "Percentage"])
            total_commits = len(classified_commits)

            for class_type, count in class_counts.most_common():
                percentage = (count / total_commits) * 100
                writer.writerow([class_type, count, f"{percentage:.1f}%"])

            writer.writerow([])

            writer.writerow(["Top Developers by Classification Diversity"])
            writer.writerow(
                ["Developer", "Total Commits", "Classifications Used", "Primary Classification"]
            )

            dev_diversity = []
            for dev, classifications in dev_class_matrix.items():
                total_dev_commits = sum(classifications.values())
                if total_dev_commits >= self.min_commits_for_analysis:
                    diversity = len(classifications)
                    primary = classifications.most_common(1)[0][0]
                    dev_diversity.append((dev, total_dev_commits, diversity, primary))

            for dev, total, diversity, primary in sorted(
                dev_diversity, key=lambda x: (x[2], x[1]), reverse=True
            )[:10]:
                writer.writerow([dev, total, diversity, primary])

            writer.writerow([])

            writer.writerow(["Language vs Classification Matrix"])
            all_classes = sorted(class_counts.keys())
            header = ["Language"] + all_classes + ["Total"]
            writer.writerow(header)

            for language, classifications in sorted(
                lang_class_matrix.items(), key=lambda x: sum(x[1].values()), reverse=True
            ):
                row = [language]
                total_lang_commits = sum(classifications.values())

                for class_type in all_classes:
                    count = classifications.get(class_type, 0)
                    percentage = (count / total_lang_commits) * 100 if total_lang_commits > 0 else 0
                    row.append(f"{count} ({percentage:.1f}%)")

                row.append(total_lang_commits)
                writer.writerow(row)

        logger.info(f"Classification matrix report generated: {output_path}")
        return str(output_path)
