"""Classification report generator for GitFlow Analytics.

This module provides comprehensive reporting capabilities for commit classification
results, including aggregate statistics, developer breakdowns, confidence analysis,
and temporal patterns. Designed to integrate with existing GitFlow Analytics
reporting infrastructure.
"""

import csv
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ClassificationReportGenerator:
    """Generator for comprehensive commit classification reports.
    
    This class creates detailed reports from commit classification results,
    providing insights into development patterns, team productivity, and
    code quality metrics through the lens of commit categorization.
    
    Key capabilities:
    - Aggregate classification statistics
    - Per-developer activity breakdowns
    - Per-repository analysis
    - Confidence score analysis
    - Temporal pattern identification
    - Export to multiple formats (CSV, JSON, Markdown)
    """
    
    def __init__(self, output_directory: Path, config: Optional[Dict[str, Any]] = None):
        """Initialize the classification report generator.
        
        Args:
            output_directory: Directory where reports will be saved
            config: Optional configuration for report generation
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {}
        self.include_low_confidence = self.config.get('include_low_confidence', True)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.min_commits_for_analysis = self.config.get('min_commits_for_analysis', 5)
        
        # Report metadata
        self.generated_at = datetime.now()
        self.reports_generated = []
        
        logger.info(f"Classification report generator initialized - output: {self.output_directory}")
    
    def generate_comprehensive_report(self, classified_commits: List[Dict[str, Any]], 
                                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Generate all available classification reports.
        
        Args:
            classified_commits: List of commits with classification results
            metadata: Optional metadata about the analysis (date range, repos, etc.)
            
        Returns:
            Dictionary mapping report types to file paths
        """
        if not classified_commits:
            logger.warning("No classified commits provided - skipping report generation")
            return {}
        
        # Filter classified commits
        classified_only = [c for c in classified_commits if 'predicted_class' in c]
        
        if not classified_only:
            logger.warning("No commits with classification results found")
            return {}
        
        logger.info(f"Generating comprehensive classification reports for {len(classified_only)} commits")
        
        report_paths = {}
        
        try:
            # Generate individual reports
            report_paths['summary'] = self.generate_summary_report(classified_only, metadata)
            report_paths['detailed_csv'] = self.generate_detailed_csv_report(classified_only, metadata)
            report_paths['developer_breakdown'] = self.generate_developer_breakdown_report(classified_only, metadata)
            report_paths['repository_analysis'] = self.generate_repository_analysis_report(classified_only, metadata)
            report_paths['confidence_analysis'] = self.generate_confidence_analysis_report(classified_only, metadata)
            report_paths['temporal_patterns'] = self.generate_temporal_patterns_report(classified_only, metadata)
            report_paths['classification_matrix'] = self.generate_classification_matrix_report(classified_only, metadata)
            report_paths['executive_summary'] = self.generate_executive_summary_report(classified_only, metadata)
            
            # Generate comprehensive JSON export
            report_paths['comprehensive_json'] = self.generate_json_export(classified_only, metadata)
            
            # Generate markdown summary
            report_paths['markdown_summary'] = self.generate_markdown_summary(classified_only, metadata)
            
            self.reports_generated = list(report_paths.keys())
            logger.info(f"Generated {len(report_paths)} classification reports")
            
            return report_paths
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive reports: {e}")
            return report_paths
    
    def generate_summary_report(self, classified_commits: List[Dict[str, Any]], 
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate high-level summary report.
        
        Args:
            classified_commits: List of classified commits
            metadata: Optional analysis metadata
            
        Returns:
            Path to generated summary CSV file
        """
        output_path = self.output_directory / f'classification_summary_{self._get_timestamp()}.csv'
        
        # Calculate summary statistics
        total_commits = len(classified_commits)
        classification_counts = Counter(c['predicted_class'] for c in classified_commits)
        confidence_scores = [c.get('classification_confidence', 0) for c in classified_commits]
        
        high_confidence_count = sum(1 for score in confidence_scores if score >= self.confidence_threshold)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        unique_developers = len(set(c.get('canonical_author_name', c.get('author_name', 'unknown')) 
                                  for c in classified_commits))
        unique_repositories = len(set(c.get('repository', 'unknown') for c in classified_commits))
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header information
            writer.writerow(['Classification Analysis Summary'])
            writer.writerow(['Generated:', self.generated_at.isoformat()])
            
            if metadata:
                writer.writerow(['Analysis Period:', f"{metadata.get('start_date', 'N/A')} to {metadata.get('end_date', 'N/A')}"])
                writer.writerow(['Configuration:', metadata.get('config_path', 'N/A')])
            
            writer.writerow([])
            
            # Overall statistics
            writer.writerow(['Overall Statistics'])
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Commits Analyzed', total_commits])
            writer.writerow(['Unique Developers', unique_developers])
            writer.writerow(['Unique Repositories', unique_repositories])
            writer.writerow(['Average Confidence Score', f'{avg_confidence:.3f}'])
            writer.writerow(['High Confidence Predictions', f'{high_confidence_count} ({(high_confidence_count/total_commits)*100:.1f}%)'])
            writer.writerow([])
            
            # Classification distribution
            writer.writerow(['Classification Distribution'])
            writer.writerow(['Classification Type', 'Count', 'Percentage'])
            
            for class_type, count in classification_counts.most_common():
                percentage = (count / total_commits) * 100
                writer.writerow([class_type, count, f'{percentage:.1f}%'])
        
        logger.info(f"Summary report generated: {output_path}")
        return str(output_path)
    
    def generate_detailed_csv_report(self, classified_commits: List[Dict[str, Any]], 
                                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate detailed CSV report with all commit information.
        
        Args:
            classified_commits: List of classified commits
            metadata: Optional analysis metadata
            
        Returns:
            Path to generated detailed CSV file
        """
        output_path = self.output_directory / f'classification_detailed_{self._get_timestamp()}.csv'
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            headers = [
                'commit_hash', 'date', 'author', 'canonical_author', 'repository',
                'predicted_class', 'confidence', 'is_reliable', 'message_preview',
                'files_changed', 'insertions', 'deletions', 'lines_changed',
                'primary_language', 'primary_activity', 'is_multilingual',
                'branch', 'project_key', 'ticket_references'
            ]
            writer.writerow(headers)
            
            # Write commit details
            for commit in classified_commits:
                file_analysis = commit.get('file_analysis_summary', {})
                
                row = [
                    commit.get('hash', '')[:12],  # Shortened hash
                    commit.get('timestamp', '').strftime('%Y-%m-%d %H:%M:%S') if commit.get('timestamp') else '',
                    commit.get('author_name', ''),
                    commit.get('canonical_author_name', commit.get('author_name', '')),
                    commit.get('repository', ''),
                    commit.get('predicted_class', ''),
                    f"{commit.get('classification_confidence', 0):.3f}",
                    commit.get('is_reliable_prediction', False),
                    commit.get('message', '')[:100].replace('\n', ' '),  # Preview of message
                    commit.get('files_changed', 0),
                    commit.get('insertions', 0),
                    commit.get('deletions', 0),
                    commit.get('insertions', 0) + commit.get('deletions', 0),
                    file_analysis.get('primary_language', ''),
                    file_analysis.get('primary_activity', ''),
                    file_analysis.get('is_multilingual', False),
                    commit.get('branch', ''),
                    commit.get('project_key', ''),
                    len(commit.get('ticket_references', []))
                ]
                writer.writerow(row)
        
        logger.info(f"Detailed CSV report generated: {output_path}")
        return str(output_path)
    
    def generate_developer_breakdown_report(self, classified_commits: List[Dict[str, Any]], 
                                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate per-developer classification breakdown.
        
        Args:
            classified_commits: List of classified commits
            metadata: Optional analysis metadata
            
        Returns:
            Path to generated developer breakdown CSV file
        """
        output_path = self.output_directory / f'classification_by_developer_{self._get_timestamp()}.csv'
        
        # Aggregate developer statistics
        developer_stats = defaultdict(lambda: {
            'total_commits': 0,
            'classifications': Counter(),
            'confidence_scores': [],
            'repositories': set(),
            'total_lines_changed': 0,
            'avg_files_per_commit': 0,
            'commit_dates': []
        })
        
        for commit in classified_commits:
            author = commit.get('canonical_author_name', commit.get('author_name', 'unknown'))
            stats = developer_stats[author]
            
            stats['total_commits'] += 1
            stats['classifications'][commit.get('predicted_class', 'unknown')] += 1
            
            if 'classification_confidence' in commit:
                stats['confidence_scores'].append(commit['classification_confidence'])
            
            stats['repositories'].add(commit.get('repository', 'unknown'))
            stats['total_lines_changed'] += commit.get('insertions', 0) + commit.get('deletions', 0)
            stats['avg_files_per_commit'] += commit.get('files_changed', 0)
            
            if commit.get('timestamp'):
                stats['commit_dates'].append(commit['timestamp'])
        
        # Calculate derived metrics
        for author, stats in developer_stats.items():
            if stats['total_commits'] > 0:
                stats['avg_confidence'] = sum(stats['confidence_scores']) / len(stats['confidence_scores']) if stats['confidence_scores'] else 0
                stats['avg_files_per_commit'] = stats['avg_files_per_commit'] / stats['total_commits']
                stats['avg_lines_per_commit'] = stats['total_lines_changed'] / stats['total_commits']
                stats['primary_classification'] = stats['classifications'].most_common(1)[0][0] if stats['classifications'] else 'unknown'
                stats['classification_diversity'] = len(stats['classifications'])
                stats['repository_count'] = len(stats['repositories'])
                
                # Calculate activity span
                if stats['commit_dates']:
                    date_range = max(stats['commit_dates']) - min(stats['commit_dates'])
                    stats['activity_span_days'] = date_range.days
                else:
                    stats['activity_span_days'] = 0
        
        # Filter developers with minimum commits
        filtered_developers = {k: v for k, v in developer_stats.items() 
                             if v['total_commits'] >= self.min_commits_for_analysis}
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write summary section
            writer.writerow(['Developer Classification Analysis'])
            writer.writerow(['Total Developers:', len(developer_stats)])
            writer.writerow(['Developers with ≥{} commits:'.format(self.min_commits_for_analysis), len(filtered_developers)])
            writer.writerow([])
            
            # Write detailed breakdown
            headers = [
                'developer', 'total_commits', 'primary_classification', 'classification_diversity',
                'avg_confidence', 'high_confidence_ratio', 'repository_count', 'repositories',
                'avg_files_per_commit', 'avg_lines_per_commit', 'activity_span_days'
            ]
            
            # Add classification type columns
            all_classifications = set()
            for stats in filtered_developers.values():
                all_classifications.update(stats['classifications'].keys())
            
            classification_headers = [f'{cls}_count' for cls in sorted(all_classifications)]
            headers.extend(classification_headers)
            
            writer.writerow(headers)
            
            # Sort developers by total commits (descending)
            sorted_developers = sorted(filtered_developers.items(), 
                                     key=lambda x: x[1]['total_commits'], reverse=True)
            
            for author, stats in sorted_developers:
                high_confidence_count = sum(1 for score in stats['confidence_scores'] 
                                          if score >= self.confidence_threshold)
                high_confidence_ratio = high_confidence_count / len(stats['confidence_scores']) if stats['confidence_scores'] else 0
                
                row = [
                    author,
                    stats['total_commits'],
                    stats['primary_classification'],
                    stats['classification_diversity'],
                    f"{stats['avg_confidence']:.3f}",
                    f"{high_confidence_ratio:.3f}",
                    stats['repository_count'],
                    '; '.join(sorted(stats['repositories'])),
                    f"{stats['avg_files_per_commit']:.1f}",
                    f"{stats['avg_lines_per_commit']:.0f}",
                    stats['activity_span_days']
                ]
                
                # Add classification counts
                for cls in sorted(all_classifications):
                    row.append(stats['classifications'].get(cls, 0))
                
                writer.writerow(row)
        
        logger.info(f"Developer breakdown report generated: {output_path}")
        return str(output_path)
    
    def generate_repository_analysis_report(self, classified_commits: List[Dict[str, Any]], 
                                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate per-repository classification analysis.
        
        Args:
            classified_commits: List of classified commits
            metadata: Optional analysis metadata
            
        Returns:
            Path to generated repository analysis CSV file
        """
        output_path = self.output_directory / f'classification_by_repository_{self._get_timestamp()}.csv'
        
        # Aggregate repository statistics
        repo_stats = defaultdict(lambda: {
            'total_commits': 0,
            'classifications': Counter(),
            'developers': set(),
            'confidence_scores': [],
            'total_lines_changed': 0,
            'languages': Counter(),
            'activities': Counter()
        })
        
        for commit in classified_commits:
            repo = commit.get('repository', 'unknown')
            stats = repo_stats[repo]
            
            stats['total_commits'] += 1
            stats['classifications'][commit.get('predicted_class', 'unknown')] += 1
            stats['developers'].add(commit.get('canonical_author_name', commit.get('author_name', 'unknown')))
            
            if 'classification_confidence' in commit:
                stats['confidence_scores'].append(commit['classification_confidence'])
            
            stats['total_lines_changed'] += commit.get('insertions', 0) + commit.get('deletions', 0)
            
            # File analysis information
            file_analysis = commit.get('file_analysis_summary', {})
            if file_analysis.get('primary_language'):
                stats['languages'][file_analysis['primary_language']] += 1
            if file_analysis.get('primary_activity'):
                stats['activities'][file_analysis['primary_activity']] += 1
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            writer.writerow(['Repository Classification Analysis'])
            writer.writerow(['Total Repositories:', len(repo_stats)])
            writer.writerow([])
            
            headers = [
                'repository', 'total_commits', 'developer_count', 'primary_classification',
                'avg_confidence', 'avg_lines_per_commit', 'primary_language', 'primary_activity',
                'classification_diversity', 'language_diversity'
            ]
            writer.writerow(headers)
            
            # Sort repositories by commit count
            sorted_repos = sorted(repo_stats.items(), key=lambda x: x[1]['total_commits'], reverse=True)
            
            for repo, stats in sorted_repos:
                avg_confidence = sum(stats['confidence_scores']) / len(stats['confidence_scores']) if stats['confidence_scores'] else 0
                avg_lines = stats['total_lines_changed'] / stats['total_commits'] if stats['total_commits'] > 0 else 0
                
                primary_class = stats['classifications'].most_common(1)[0][0] if stats['classifications'] else 'unknown'
                primary_lang = stats['languages'].most_common(1)[0][0] if stats['languages'] else 'unknown'
                primary_activity = stats['activities'].most_common(1)[0][0] if stats['activities'] else 'unknown'
                
                row = [
                    repo,
                    stats['total_commits'],
                    len(stats['developers']),
                    primary_class,
                    f"{avg_confidence:.3f}",
                    f"{avg_lines:.0f}",
                    primary_lang,
                    primary_activity,
                    len(stats['classifications']),
                    len(stats['languages'])
                ]
                writer.writerow(row)
        
        logger.info(f"Repository analysis report generated: {output_path}")
        return str(output_path)
    
    def generate_confidence_analysis_report(self, classified_commits: List[Dict[str, Any]], 
                                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate confidence score analysis report.
        
        Args:
            classified_commits: List of classified commits
            metadata: Optional analysis metadata
            
        Returns:
            Path to generated confidence analysis CSV file
        """
        output_path = self.output_directory / f'classification_confidence_analysis_{self._get_timestamp()}.csv'
        
        confidence_scores = [c.get('classification_confidence', 0) for c in classified_commits]
        
        # Calculate confidence statistics by classification type
        confidence_by_class = defaultdict(list)
        for commit in classified_commits:
            class_type = commit.get('predicted_class', 'unknown')
            confidence = commit.get('classification_confidence', 0)
            confidence_by_class[class_type].append(confidence)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            writer.writerow(['Classification Confidence Analysis'])
            writer.writerow([])
            
            # Overall confidence statistics
            if confidence_scores:
                writer.writerow(['Overall Confidence Statistics'])
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Total Predictions', len(confidence_scores)])
                writer.writerow(['Average Confidence', f"{sum(confidence_scores) / len(confidence_scores):.3f}"])
                writer.writerow(['Minimum Confidence', f"{min(confidence_scores):.3f}"])
                writer.writerow(['Maximum Confidence', f"{max(confidence_scores):.3f}"])
                
                # Confidence distribution
                very_high = sum(1 for s in confidence_scores if s >= 0.9)
                high = sum(1 for s in confidence_scores if 0.8 <= s < 0.9)
                medium = sum(1 for s in confidence_scores if 0.6 <= s < 0.8)
                low = sum(1 for s in confidence_scores if 0.4 <= s < 0.6)
                very_low = sum(1 for s in confidence_scores if s < 0.4)
                
                writer.writerow(['Very High (≥0.9)', f"{very_high} ({(very_high/len(confidence_scores))*100:.1f}%)"])
                writer.writerow(['High (0.8-0.9)', f"{high} ({(high/len(confidence_scores))*100:.1f}%)"])
                writer.writerow(['Medium (0.6-0.8)', f"{medium} ({(medium/len(confidence_scores))*100:.1f}%)"])
                writer.writerow(['Low (0.4-0.6)', f"{low} ({(low/len(confidence_scores))*100:.1f}%)"])
                writer.writerow(['Very Low (<0.4)', f"{very_low} ({(very_low/len(confidence_scores))*100:.1f}%)"])
                writer.writerow([])
            
            # Confidence by classification type
            writer.writerow(['Confidence by Classification Type'])
            writer.writerow(['Classification', 'Count', 'Avg Confidence', 'Min', 'Max', 'High Confidence Count'])
            
            for class_type, scores in sorted(confidence_by_class.items()):
                if scores:
                    avg_conf = sum(scores) / len(scores)
                    high_conf_count = sum(1 for s in scores if s >= self.confidence_threshold)
                    
                    writer.writerow([
                        class_type,
                        len(scores),
                        f"{avg_conf:.3f}",
                        f"{min(scores):.3f}",
                        f"{max(scores):.3f}",
                        f"{high_conf_count} ({(high_conf_count/len(scores))*100:.1f}%)"
                    ])
        
        logger.info(f"Confidence analysis report generated: {output_path}")
        return str(output_path)
    
    def generate_temporal_patterns_report(self, classified_commits: List[Dict[str, Any]], 
                                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate temporal patterns analysis report.
        
        Args:
            classified_commits: List of classified commits
            metadata: Optional analysis metadata
            
        Returns:
            Path to generated temporal patterns CSV file
        """
        output_path = self.output_directory / f'classification_temporal_patterns_{self._get_timestamp()}.csv'
        
        # Group commits by date
        daily_stats = defaultdict(lambda: {
            'total_commits': 0,
            'classifications': Counter(),
            'developers': set(),
            'confidence_scores': []
        })
        
        for commit in classified_commits:
            if commit.get('timestamp'):
                date_key = commit['timestamp'].date()
                stats = daily_stats[date_key]
                
                stats['total_commits'] += 1
                stats['classifications'][commit.get('predicted_class', 'unknown')] += 1
                stats['developers'].add(commit.get('canonical_author_name', commit.get('author_name', 'unknown')))
                
                if 'classification_confidence' in commit:
                    stats['confidence_scores'].append(commit['classification_confidence'])
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            writer.writerow(['Temporal Classification Patterns'])
            writer.writerow([])
            
            # Get all classification types for column headers
            all_classifications = set()
            for stats in daily_stats.values():
                all_classifications.update(stats['classifications'].keys())
            
            headers = ['date', 'total_commits', 'developer_count', 'avg_confidence']
            headers.extend([f'{cls}_count' for cls in sorted(all_classifications)])
            writer.writerow(headers)
            
            # Sort by date
            for date, stats in sorted(daily_stats.items()):
                avg_confidence = sum(stats['confidence_scores']) / len(stats['confidence_scores']) if stats['confidence_scores'] else 0
                
                row = [
                    date.isoformat(),
                    stats['total_commits'],
                    len(stats['developers']),
                    f"{avg_confidence:.3f}"
                ]
                
                # Add classification counts for this date
                for cls in sorted(all_classifications):
                    row.append(stats['classifications'].get(cls, 0))
                
                writer.writerow(row)
        
        logger.info(f"Temporal patterns report generated: {output_path}")
        return str(output_path)
    
    def generate_classification_matrix_report(self, classified_commits: List[Dict[str, Any]], 
                                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate classification distribution matrix report.
        
        Args:
            classified_commits: List of classified commits
            metadata: Optional analysis metadata
            
        Returns:
            Path to generated classification matrix CSV file
        """
        output_path = self.output_directory / f'classification_matrix_{self._get_timestamp()}.csv'
        
        # Create cross-tabulation of classifications vs other dimensions
        class_counts = Counter(c.get('predicted_class', 'unknown') for c in classified_commits)
        
        # Developer vs Classification matrix
        dev_class_matrix = defaultdict(Counter)
        repo_class_matrix = defaultdict(Counter)
        lang_class_matrix = defaultdict(Counter)
        
        for commit in classified_commits:
            class_type = commit.get('predicted_class', 'unknown')
            developer = commit.get('canonical_author_name', commit.get('author_name', 'unknown'))
            repository = commit.get('repository', 'unknown')
            
            dev_class_matrix[developer][class_type] += 1
            repo_class_matrix[repository][class_type] += 1
            
            # Language information
            file_analysis = commit.get('file_analysis_summary', {})
            language = file_analysis.get('primary_language', 'unknown')
            lang_class_matrix[language][class_type] += 1
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            writer.writerow(['Classification Distribution Matrix'])
            writer.writerow([])
            
            # Overall classification distribution
            writer.writerow(['Overall Classification Distribution'])
            writer.writerow(['Classification', 'Count', 'Percentage'])
            total_commits = len(classified_commits)
            
            for class_type, count in class_counts.most_common():
                percentage = (count / total_commits) * 100
                writer.writerow([class_type, count, f'{percentage:.1f}%'])
            
            writer.writerow([])
            
            # Top developers by classification diversity
            writer.writerow(['Top Developers by Classification Diversity'])
            writer.writerow(['Developer', 'Total Commits', 'Classifications Used', 'Primary Classification'])
            
            dev_diversity = []
            for dev, classifications in dev_class_matrix.items():
                total_dev_commits = sum(classifications.values())
                if total_dev_commits >= self.min_commits_for_analysis:
                    diversity = len(classifications)
                    primary = classifications.most_common(1)[0][0]
                    dev_diversity.append((dev, total_dev_commits, diversity, primary))
            
            # Sort by diversity, then by total commits
            for dev, total, diversity, primary in sorted(dev_diversity, key=lambda x: (x[2], x[1]), reverse=True)[:10]:
                writer.writerow([dev, total, diversity, primary])
            
            writer.writerow([])
            
            # Language vs Classification matrix
            writer.writerow(['Language vs Classification Matrix'])
            all_classes = sorted(class_counts.keys())
            header = ['Language'] + all_classes + ['Total']
            writer.writerow(header)
            
            for language, classifications in sorted(lang_class_matrix.items(), 
                                                  key=lambda x: sum(x[1].values()), reverse=True):
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
    
    def generate_executive_summary_report(self, classified_commits: List[Dict[str, Any]], 
                                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate executive summary report for leadership.
        
        Args:
            classified_commits: List of classified commits
            metadata: Optional analysis metadata
            
        Returns:
            Path to generated executive summary CSV file
        """
        output_path = self.output_directory / f'classification_executive_summary_{self._get_timestamp()}.csv'
        
        # Calculate key metrics
        total_commits = len(classified_commits)
        unique_developers = len(set(c.get('canonical_author_name', c.get('author_name', 'unknown')) 
                                  for c in classified_commits))
        unique_repositories = len(set(c.get('repository', 'unknown') for c in classified_commits))
        
        classification_counts = Counter(c.get('predicted_class', 'unknown') for c in classified_commits)
        confidence_scores = [c.get('classification_confidence', 0) for c in classified_commits]
        
        # Productivity metrics
        total_lines_changed = sum(c.get('insertions', 0) + c.get('deletions', 0) for c in classified_commits)
        avg_lines_per_commit = total_lines_changed / total_commits if total_commits > 0 else 0
        
        # Time span analysis
        commit_dates = [c['timestamp'] for c in classified_commits if c.get('timestamp')]
        if commit_dates:
            analysis_span = (max(commit_dates) - min(commit_dates)).days
        else:
            analysis_span = 0
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            writer.writerow(['Executive Summary - Commit Classification Analysis'])
            writer.writerow(['Generated:', self.generated_at.strftime('%Y-%m-%d %H:%M:%S')])
            
            if metadata:
                writer.writerow(['Analysis Period:', f"{metadata.get('start_date', 'N/A')} to {metadata.get('end_date', 'N/A')}"])
            
            writer.writerow([])
            
            # Key metrics
            writer.writerow(['KEY METRICS'])
            writer.writerow(['Total Development Activity', f'{total_commits:,} commits'])
            writer.writerow(['Team Size', f'{unique_developers} active developers'])
            writer.writerow(['Codebase Scope', f'{unique_repositories} repositories'])
            writer.writerow(['Analysis Timespan', f'{analysis_span} days'])
            writer.writerow(['Average Code Changes per Commit', f'{avg_lines_per_commit:.0f} lines'])
            
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                high_confidence_pct = (sum(1 for s in confidence_scores if s >= self.confidence_threshold) / len(confidence_scores)) * 100
                writer.writerow(['Classification Confidence', f'{avg_confidence:.1%} average'])
                writer.writerow(['High Confidence Predictions', f'{high_confidence_pct:.1f}%'])
            
            writer.writerow([])
            
            # Development focus areas
            writer.writerow(['DEVELOPMENT FOCUS AREAS'])
            writer.writerow(['Activity Type', 'Commits', '% of Total', 'Strategic Insight'])
            
            # Define strategic insights for each classification type
            strategic_insights = {
                'feature': 'New capability development',
                'bugfix': 'Quality maintenance and stability',
                'refactor': 'Technical debt management',
                'docs': 'Knowledge management and documentation',
                'test': 'Quality assurance and testing',
                'config': 'Infrastructure and configuration',
                'chore': 'Maintenance and operational tasks',
                'security': 'Security and compliance',
                'hotfix': 'Critical issue resolution',
                'style': 'Code quality and standards',
                'build': 'Build system and deployment',
                'ci': 'Automation and continuous integration'
            }
            
            for class_type, count in classification_counts.most_common():
                percentage = (count / total_commits) * 100
                insight = strategic_insights.get(class_type, 'Unclassified development activity')
                writer.writerow([class_type.title(), f'{count:,}', f'{percentage:.1f}%', insight])
            
            writer.writerow([])
            
            # Recommendations
            writer.writerow(['STRATEGIC RECOMMENDATIONS'])
            
            # Generate recommendations based on the data
            recommendations = []
            
            # Feature vs maintenance balance
            feature_pct = (classification_counts.get('feature', 0) / total_commits) * 100
            maintenance_pct = ((classification_counts.get('bugfix', 0) + 
                              classification_counts.get('refactor', 0) + 
                              classification_counts.get('chore', 0)) / total_commits) * 100
            
            if feature_pct > 60:
                recommendations.append("High feature development velocity - consider increasing quality assurance")
            elif feature_pct < 20:
                recommendations.append("Low feature development - may indicate focus on maintenance or technical debt")
            
            if maintenance_pct > 40:
                recommendations.append("High maintenance overhead - consider technical debt reduction initiatives")
            
            # Documentation analysis
            docs_pct = (classification_counts.get('docs', 0) / total_commits) * 100
            if docs_pct < 5:
                recommendations.append("Low documentation activity - consider improving documentation practices")
            
            # Testing analysis
            test_pct = (classification_counts.get('test', 0) / total_commits) * 100
            if test_pct < 10:
                recommendations.append("Limited testing activity - consider strengthening testing practices")
            
            # Security analysis
            security_pct = (classification_counts.get('security', 0) / total_commits) * 100
            if security_pct > 0:
                recommendations.append(f"Active security focus ({security_pct:.1f}% of commits) - positive security posture")
            
            # Confidence analysis
            if confidence_scores:
                low_confidence_pct = (sum(1 for s in confidence_scores if s < 0.6) / len(confidence_scores)) * 100
                if low_confidence_pct > 20:
                    recommendations.append("Consider improving commit message clarity for better classification")
            
            for i, recommendation in enumerate(recommendations, 1):
                writer.writerow([f'Recommendation {i}', recommendation])
        
        logger.info(f"Executive summary report generated: {output_path}")
        return str(output_path)
    
    def generate_json_export(self, classified_commits: List[Dict[str, Any]], 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive JSON export of all classification data.
        
        Args:
            classified_commits: List of classified commits
            metadata: Optional analysis metadata
            
        Returns:
            Path to generated JSON file
        """
        output_path = self.output_directory / f'classification_comprehensive_{self._get_timestamp()}.json'
        
        # Create comprehensive data structure
        export_data = {
            'metadata': {
                'generated_at': self.generated_at.isoformat(),
                'total_commits': len(classified_commits),
                'generator_version': '1.0',
                'config': self.config
            },
            'summary_statistics': self._calculate_summary_statistics(classified_commits),
            'commits': classified_commits,
            'analysis_metadata': metadata or {}
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"JSON export generated: {output_path}")
        return str(output_path)
    
    def generate_markdown_summary(self, classified_commits: List[Dict[str, Any]], 
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate markdown summary report.
        
        Args:
            classified_commits: List of classified commits
            metadata: Optional analysis metadata
            
        Returns:
            Path to generated markdown file
        """
        output_path = self.output_directory / f'classification_summary_{self._get_timestamp()}.md'
        
        # Calculate statistics
        total_commits = len(classified_commits)
        classification_counts = Counter(c.get('predicted_class', 'unknown') for c in classified_commits)
        confidence_scores = [c.get('classification_confidence', 0) for c in classified_commits]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Commit Classification Analysis Report\n\n")
            f.write(f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if metadata:
                f.write(f"**Analysis Period:** {metadata.get('start_date', 'N/A')} to {metadata.get('end_date', 'N/A')}\n\n")
            
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Total Commits Analyzed:** {total_commits:,}\n")
            f.write(f"- **Unique Developers:** {len(set(c.get('canonical_author_name', c.get('author_name', 'unknown')) for c in classified_commits))}\n")
            f.write(f"- **Unique Repositories:** {len(set(c.get('repository', 'unknown') for c in classified_commits))}\n")
            
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                high_confidence_count = sum(1 for s in confidence_scores if s >= self.confidence_threshold)
                f.write(f"- **Average Confidence:** {avg_confidence:.1%}\n")
                f.write(f"- **High Confidence Predictions:** {high_confidence_count:,} ({(high_confidence_count/total_commits)*100:.1f}%)\n")
            
            f.write("\n## Classification Distribution\n\n")
            f.write("| Classification Type | Count | Percentage |\n")
            f.write("|-------------------|--------|------------|\n")
            
            for class_type, count in classification_counts.most_common():
                percentage = (count / total_commits) * 100
                f.write(f"| {class_type.title()} | {count:,} | {percentage:.1f}% |\n")
            
            f.write(f"\n## Analysis Details\n\n")
            f.write(f"This report was generated using GitFlow Analytics commit classification system.\n")
            f.write(f"Classification confidence threshold: {self.confidence_threshold}\n\n")
            
            f.write("For detailed analysis, see the accompanying CSV reports:\n")
            for report_type in self.reports_generated:
                f.write(f"- {report_type.replace('_', ' ').title()}\n")
        
        logger.info(f"Markdown summary generated: {output_path}")
        return str(output_path)
    
    def _calculate_summary_statistics(self, classified_commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics.
        
        Args:
            classified_commits: List of classified commits
            
        Returns:
            Dictionary containing summary statistics
        """
        total_commits = len(classified_commits)
        
        classification_counts = Counter(c.get('predicted_class', 'unknown') for c in classified_commits)
        confidence_scores = [c.get('classification_confidence', 0) for c in classified_commits]
        
        developers = set(c.get('canonical_author_name', c.get('author_name', 'unknown')) for c in classified_commits)
        repositories = set(c.get('repository', 'unknown') for c in classified_commits)
        
        return {
            'total_commits': total_commits,
            'unique_developers': len(developers),
            'unique_repositories': len(repositories),
            'classification_distribution': dict(classification_counts),
            'confidence_statistics': {
                'average': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                'minimum': min(confidence_scores) if confidence_scores else 0,
                'maximum': max(confidence_scores) if confidence_scores else 0,
                'high_confidence_count': sum(1 for s in confidence_scores if s >= self.confidence_threshold),
                'high_confidence_percentage': (sum(1 for s in confidence_scores if s >= self.confidence_threshold) / len(confidence_scores)) * 100 if confidence_scores else 0
            },
            'productivity_metrics': {
                'total_lines_changed': sum(c.get('insertions', 0) + c.get('deletions', 0) for c in classified_commits),
                'average_lines_per_commit': sum(c.get('insertions', 0) + c.get('deletions', 0) for c in classified_commits) / total_commits if total_commits > 0 else 0,
                'average_files_per_commit': sum(c.get('files_changed', 0) for c in classified_commits) / total_commits if total_commits > 0 else 0
            }
        }
    
    def _get_timestamp(self) -> str:
        """Get timestamp string for file naming.
        
        Returns:
            Timestamp string in YYYYMMDD_HHMMSS format
        """
        return self.generated_at.strftime('%Y%m%d_%H%M%S')