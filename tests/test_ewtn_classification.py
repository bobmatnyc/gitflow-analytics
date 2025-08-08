#!/usr/bin/env python3
"""
EWTN Repository Classification Test Script

This script tests the new commit classification system with real EWTN data.
It analyzes commits from June 1-7, 2024, applies the classification pipeline,
uses developer normalization from the EWTN config, and generates comprehensive
classification reports.

Features:
- Loads EWTN configuration from ~/Clients/EWTN/gfa/config.yaml
- Analyzes commits from a specific week in June 2024
- Applies commit classification using the new system
- Uses EWTN's developer identity normalization
- Generates aggregate classification results
- Produces detailed reports with confidence scores
- Handles organization-based repository discovery

Usage:
    python test_ewtn_classification.py [--dry-run] [--debug] [--force-retrain]
"""

import argparse
import logging
import sys
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Add the src directory to Python path for local development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gitflow_analytics.config import ConfigLoader
from gitflow_analytics.core.analyzer import GitAnalyzer
from gitflow_analytics.core.cache import GitAnalysisCache
from gitflow_analytics.core.identity import DeveloperIdentityResolver
from gitflow_analytics.classification.classifier import CommitClassifier
from gitflow_analytics.integrations.github_integration import GitHubIntegration
from gitflow_analytics.reports.classification_writer import ClassificationReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EWTNClassificationTester:
    """Test harness for EWTN repository classification analysis."""
    
    def __init__(self, config_path: Path, dry_run: bool = False, debug: bool = False):
        """Initialize the EWTN classification tester.
        
        Args:
            config_path: Path to EWTN configuration file
            dry_run: If True, don't make API calls or write files
            debug: Enable debug logging
        """
        self.dry_run = dry_run
        self.debug = debug
        
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        # Load configuration
        logger.info(f"Loading EWTN configuration from: {config_path}")
        self.config_path = Path(config_path)  # Ensure it's a Path object
        self.config = ConfigLoader.load(self.config_path)
        
        # Set up test date range (June 1-7, 2024)
        self.test_start = datetime(2024, 6, 1, tzinfo=timezone.utc)
        self.test_end = datetime(2024, 6, 7, 23, 59, 59, tzinfo=timezone.utc)
        
        logger.info(f"Test period: {self.test_start.date()} to {self.test_end.date()}")
        
        # Initialize components
        self._setup_components()
        
        # Results storage
        self.analysis_results = {
            'repositories_analyzed': 0,
            'total_commits': 0,
            'classified_commits': 0,
            'developers': set(),
            'classification_distribution': Counter(),
            'confidence_scores': [],
            'processing_time': 0.0,
            'errors': []
        }
    
    def _setup_components(self):
        """Initialize GitFlow Analytics components."""
        try:
            # Setup cache
            cache_dir_str = self.config.cache.directory
            logger.debug(f"Cache directory string: {cache_dir_str}")
            logger.debug(f"Config path: {self.config_path}")
            cache_dir = Path(cache_dir_str)
            logger.debug(f"Cache dir Path object: {cache_dir}")
            if not cache_dir.is_absolute():
                cache_dir = self.config_path.parent / cache_dir
                logger.debug(f"Resolved cache dir: {cache_dir}")
            
            logger.info(f"Using cache directory: {cache_dir}")
            self.cache = GitAnalysisCache(cache_dir)
            
            # Setup identity resolver
            logger.info("Initializing identity resolver with EWTN mappings")
            cache_db_path = cache_dir / 'identities.db'
            
            # Get identity configuration safely
            identity_config = getattr(self.config.analysis, 'identity', None)
            manual_mappings = getattr(identity_config, 'manual_mappings', []) if identity_config else []
            similarity_threshold = getattr(identity_config, 'similarity_threshold', 0.85) if identity_config else 0.85
            
            logger.debug(f"Manual mappings: {len(manual_mappings)} entries")
            logger.debug(f"Similarity threshold: {similarity_threshold}")
            logger.debug(f"Cache DB path: {cache_db_path}")
            
            self.identity_resolver = DeveloperIdentityResolver(
                db_path=cache_db_path,
                manual_mappings=manual_mappings,
                similarity_threshold=similarity_threshold
            )
            
            logger.debug("Identity resolver initialized successfully")
            
            # Setup classification system
            classification_config = {
                'enabled': True,
                'confidence_threshold': 0.7,  # Higher threshold for production
                'batch_size': 50,  # Smaller batches for memory efficiency
                'auto_retrain': True,
                'model': {
                    'algorithm': 'random_forest',
                    'n_estimators': 200,
                    'max_depth': 20,
                    'min_samples_split': 5
                }
            }
            
            logger.info("Initializing commit classification system")
            self.classifier = CommitClassifier(
                config=classification_config,
                cache_dir=cache_dir / 'classification'
            )
            
            # Setup Git analyzer with classification enabled
            branch_mapping_rules = getattr(self.config.analysis, 'branch_mapping_rules', {})
            ticket_platforms = getattr(self.config.analysis, 'ticket_platforms', ['jira'])
            exclude_config = getattr(self.config.analysis, 'exclude', None)
            exclude_paths = getattr(exclude_config, 'paths', []) if exclude_config else []
            
            self.analyzer = GitAnalyzer(
                cache=self.cache,
                batch_size=100,
                branch_mapping_rules=branch_mapping_rules,
                allowed_ticket_platforms=ticket_platforms,
                exclude_paths=exclude_paths,
                classification_config=classification_config
            )
            
            # Setup GitHub integration for organization discovery
            if not self.dry_run:
                github_config = self.config.github
                self.github_integration = GitHubIntegration(
                    token=github_config.token,
                    cache=self.cache
                )
            else:
                self.github_integration = None
                logger.info("Dry run mode: GitHub integration disabled")
                
        except Exception as e:
            logger.error(f"Failed to setup components: {e}")
            raise
    
    def discover_repositories(self) -> List[Dict[str, Any]]:
        """Discover repositories to analyze.
        
        Returns:
            List of repository information dictionaries
        """
        repositories = []
        
        # If repositories are explicitly configured, use those
        if self.config.repositories:
            logger.info(f"Using {len(self.config.repositories)} configured repositories")
            for repo_config in self.config.repositories:
                repositories.append({
                    'name': repo_config.path.name,
                    'path': repo_config.path,
                    'branch': getattr(repo_config, 'branch', None),
                    'project_key': getattr(repo_config, 'project_key', repo_config.path.name.upper())
                })
        
        # Otherwise discover from GitHub organization (simplified approach)
        elif self.github_integration and self.config.github.organization:
            logger.info(f"Discovering repositories from organization: {self.config.github.organization}")
            try:
                # Use GitHub API to get organization repositories
                github_client = self.github_integration.github
                org = github_client.get_organization(self.config.github.organization)
                
                # Get first 10 repositories for testing
                repo_count = 0
                for github_repo in org.get_repos():
                    if repo_count >= 10:  # Limit for testing
                        break
                    
                    if github_repo.archived:
                        continue  # Skip archived repos
                    
                    repo_path = Path.cwd() / 'repos' / github_repo.name
                    repositories.append({
                        'name': github_repo.name,
                        'path': repo_path,
                        'branch': github_repo.default_branch,
                        'project_key': github_repo.name.upper().replace('-', '_'),
                        'clone_url': github_repo.clone_url,
                        'github_repo': github_repo
                    })
                    
                    # Note: For this test, we assume repositories are already cloned
                    # In production, you'd implement cloning here
                    if not repo_path.exists():
                        logger.warning(f"Repository path does not exist: {repo_path}")
                        logger.info(f"To clone: git clone {github_repo.clone_url} {repo_path}")
                    
                    repo_count += 1
                        
            except Exception as e:
                logger.error(f"Failed to discover repositories: {e}")
                self.analysis_results['errors'].append(f"Repository discovery failed: {e}")
                
        else:
            logger.warning("No repositories configured and no organization specified")
        
        logger.info(f"Found {len(repositories)} repositories to analyze")
        return repositories
    
    def analyze_repository_commits(self, repo_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze commits from a single repository.
        
        Args:
            repo_info: Repository information dictionary
            
        Returns:
            List of analyzed commit data
        """
        repo_path = repo_info['path']
        repo_name = repo_info['name']
        
        logger.info(f"Analyzing repository: {repo_name}")
        
        if not repo_path.exists():
            error_msg = f"Repository path does not exist: {repo_path}"
            logger.error(error_msg)
            self.analysis_results['errors'].append(error_msg)
            return []
        
        try:
            # Analyze commits in the test date range
            commits = self.analyzer.analyze_repository(
                repo_path=repo_path,
                since=self.test_start,
                branch=repo_info.get('branch')
            )
            
            # Filter commits to exact date range
            filtered_commits = []
            for commit in commits:
                commit_time = commit['timestamp']
                if self.test_start <= commit_time <= self.test_end:
                    # Add repository context
                    commit['repository'] = repo_name
                    commit['project_key'] = repo_info['project_key']
                    filtered_commits.append(commit)
            
            logger.info(f"Found {len(filtered_commits)} commits in test period for {repo_name}")
            return filtered_commits
            
        except Exception as e:
            error_msg = f"Failed to analyze repository {repo_name}: {e}"
            logger.error(error_msg)
            self.analysis_results['errors'].append(error_msg)
            return []
    
    def classify_commits(self, commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply classification to commits.
        
        Args:
            commits: List of commit data dictionaries
            
        Returns:
            List of commits with classification results
        """
        if not commits:
            return []
        
        logger.info(f"Classifying {len(commits)} commits")
        
        try:
            # Check if we need to train the model first
            model_status = self.classifier.get_model_status()
            if not model_status['model_trained']:
                logger.info("Model not trained, using initial rule-based training")
                self._bootstrap_model_training(commits)
            
            # Classify commits
            classification_results = self.classifier.classify_commits(commits)
            
            # Merge classification results back into commits
            classified_commits = []
            for commit, classification in zip(commits, classification_results):
                commit_with_classification = commit.copy()
                commit_with_classification.update({
                    'predicted_class': classification['predicted_class'],
                    'classification_confidence': classification['confidence'],
                    'is_reliable_prediction': classification['is_reliable_prediction'],
                    'class_probabilities': classification['class_probabilities'],
                    'file_analysis': classification['file_analysis'],
                    'classification_metadata': classification['classification_metadata']
                })
                classified_commits.append(commit_with_classification)
            
            logger.info(f"Successfully classified {len(classified_commits)} commits")
            return classified_commits
            
        except Exception as e:
            error_msg = f"Classification failed: {e}"
            logger.error(error_msg)
            self.analysis_results['errors'].append(error_msg)
            return commits  # Return original commits without classification
    
    def _bootstrap_model_training(self, commits: List[Dict[str, Any]]):
        """Bootstrap model training with rule-based initial labels.
        
        Args:
            commits: Commits to use for initial training
        """
        logger.info("Bootstrapping model training with rule-based labels")
        
        training_data = []
        for commit in commits[:1000]:  # Use up to 1000 commits for training
            # Simple rule-based labeling for bootstrap
            message = commit.get('message', '').lower()
            
            if any(word in message for word in ['fix', 'bug', 'error', 'issue']):
                label = 'bugfix'
            elif any(word in message for word in ['feat', 'feature', 'add', 'implement']):
                label = 'feature'  
            elif any(word in message for word in ['doc', 'docs', 'readme', 'comment']):
                label = 'docs'
            elif any(word in message for word in ['test', 'spec', 'coverage']):
                label = 'test'
            elif any(word in message for word in ['refactor', 'cleanup', 'optimize']):
                label = 'refactor'
            elif any(word in message for word in ['config', 'setting', 'env']):
                label = 'config'
            elif any(word in message for word in ['merge', 'Merge']):
                label = 'merge'
            else:
                label = 'chore'
            
            training_data.append((commit, label))
        
        if training_data:
            try:
                self.classifier.train_model(training_data, validation_split=0.2)
                logger.info(f"Bootstrap training completed with {len(training_data)} samples")
            except Exception as e:
                logger.warning(f"Bootstrap training failed: {e}")
    
    def apply_identity_normalization(self, commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply EWTN developer identity normalization.
        
        Args:
            commits: List of commits with author information
            
        Returns:
            List of commits with normalized author identities
        """
        logger.info("Applying EWTN developer identity normalization")
        
        normalized_commits = []
        for commit in commits:
            # Get canonical identity (resolve_developer returns canonical_id)
            canonical_id = self.identity_resolver.resolve_developer(
                commit['author_name'],
                commit['author_email']
            )
            
            # Update commit with canonical identity
            normalized_commit = commit.copy()
            normalized_commit.update({
                'canonical_author_name': canonical_id,  # Using canonical_id as name for now
                'canonical_author_email': commit['author_email'],  # Keep original email
                'canonical_author_id': canonical_id,
                'identity_confidence': 1.0  # Assume high confidence for resolved identities
            })
            
            normalized_commits.append(normalized_commit)
            
            # Track developers using canonical ID
            self.analysis_results['developers'].add(canonical_id)
        
        logger.info(f"Normalized identities for {len(normalized_commits)} commits")
        logger.info(f"Found {len(self.analysis_results['developers'])} unique developers")
        
        return normalized_commits
    
    def generate_classification_reports(self, classified_commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive classification reports using the professional report generator.
        
        Args:
            classified_commits: List of classified commits
            
        Returns:
            Dictionary containing all report data
        """
        logger.info("Generating classification reports using ClassificationReportGenerator")
        
        # Create output directory for reports
        reports_output_dir = Path('ewtn_classification_results') / 'detailed_reports'
        
        # Setup report generator
        report_generator = ClassificationReportGenerator(
            output_directory=reports_output_dir,
            config={
                'confidence_threshold': 0.7,
                'min_commits_for_analysis': 3,
                'include_low_confidence': True
            }
        )
        
        # Prepare metadata for reports
        metadata = {
            'start_date': self.test_start.date().isoformat(),
            'end_date': self.test_end.date().isoformat(),
            'config_path': str(self.config_path),
            'organization': self.config.github.organization,
            'analysis_type': 'EWTN Commit Classification Test',
            'total_repositories_analyzed': self.analysis_results['repositories_analyzed']
        }
        
        # Generate comprehensive reports
        report_paths = report_generator.generate_comprehensive_report(
            classified_commits=classified_commits,
            metadata=metadata
        )
        
        # Also generate our custom summary statistics for backward compatibility
        custom_reports = {
            'summary': self._generate_summary_report(classified_commits),
            'by_developer': self._generate_developer_report(classified_commits),
            'by_repository': self._generate_repository_report(classified_commits),
            'by_classification': self._generate_classification_breakdown(classified_commits),
            'confidence_analysis': self._generate_confidence_analysis(classified_commits),
            'temporal_patterns': self._generate_temporal_patterns(classified_commits)
        }
        
        # Combine professional reports and custom analysis
        return {
            'professional_reports': report_paths,
            'custom_analysis': custom_reports,
            'output_directory': str(reports_output_dir)
        }
    
    def _generate_summary_report(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate high-level summary report."""
        total_commits = len(commits)
        classified_commits = [c for c in commits if 'predicted_class' in c]
        
        return {
            'total_commits': total_commits,
            'classified_commits': len(classified_commits),
            'classification_coverage': len(classified_commits) / total_commits if total_commits > 0 else 0,
            'unique_developers': len(self.analysis_results['developers']),
            'unique_repositories': len(set(c.get('repository', 'unknown') for c in commits)),
            'date_range': {
                'start': self.test_start.isoformat(),
                'end': self.test_end.isoformat()
            },
            'processing_errors': len(self.analysis_results['errors'])
        }
    
    def _generate_developer_report(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate per-developer classification breakdown."""
        developer_stats = defaultdict(lambda: {
            'total_commits': 0,
            'classifications': Counter(),
            'confidence_scores': [],
            'repositories': set(),
            'avg_files_changed': 0,
            'total_lines_changed': 0
        })
        
        for commit in commits:
            if 'canonical_author_name' not in commit:
                continue
                
            dev_name = commit['canonical_author_name']
            stats = developer_stats[dev_name]
            
            stats['total_commits'] += 1
            stats['repositories'].add(commit.get('repository', 'unknown'))
            stats['avg_files_changed'] += commit.get('files_changed', 0)
            stats['total_lines_changed'] += commit.get('insertions', 0) + commit.get('deletions', 0)
            
            if 'predicted_class' in commit:
                stats['classifications'][commit['predicted_class']] += 1
                stats['confidence_scores'].append(commit.get('classification_confidence', 0))
        
        # Finalize statistics
        for dev_name, stats in developer_stats.items():
            if stats['total_commits'] > 0:
                stats['avg_files_changed'] = stats['avg_files_changed'] / stats['total_commits']
                stats['avg_confidence'] = sum(stats['confidence_scores']) / len(stats['confidence_scores']) if stats['confidence_scores'] else 0
                stats['repositories'] = list(stats['repositories'])
                stats['primary_classification'] = stats['classifications'].most_common(1)[0][0] if stats['classifications'] else 'unknown'
        
        return dict(developer_stats)
    
    def _generate_repository_report(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate per-repository classification breakdown."""
        repo_stats = defaultdict(lambda: {
            'total_commits': 0,
            'classifications': Counter(),
            'developers': set(),
            'confidence_scores': [],
            'avg_commit_size': 0
        })
        
        for commit in commits:
            repo_name = commit.get('repository', 'unknown')
            stats = repo_stats[repo_name]
            
            stats['total_commits'] += 1
            stats['developers'].add(commit.get('canonical_author_name', commit.get('author_name', 'unknown')))
            
            commit_size = commit.get('insertions', 0) + commit.get('deletions', 0)
            stats['avg_commit_size'] += commit_size
            
            if 'predicted_class' in commit:
                stats['classifications'][commit['predicted_class']] += 1
                stats['confidence_scores'].append(commit.get('classification_confidence', 0))
        
        # Finalize statistics
        for repo_name, stats in repo_stats.items():
            if stats['total_commits'] > 0:
                stats['avg_commit_size'] = stats['avg_commit_size'] / stats['total_commits']
                stats['avg_confidence'] = sum(stats['confidence_scores']) / len(stats['confidence_scores']) if stats['confidence_scores'] else 0
                stats['developers'] = list(stats['developers'])
                stats['developer_count'] = len(stats['developers'])
                stats['primary_classification'] = stats['classifications'].most_common(1)[0][0] if stats['classifications'] else 'unknown'
        
        return dict(repo_stats)
    
    def _generate_classification_breakdown(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate classification type breakdown."""
        classified_commits = [c for c in commits if 'predicted_class' in c]
        
        if not classified_commits:
            return {}
        
        breakdown = {
            'total_classified': len(classified_commits),
            'distribution': {},
            'by_confidence': {
                'high_confidence': {'threshold': 0.8, 'count': 0, 'types': Counter()},
                'medium_confidence': {'threshold': 0.6, 'count': 0, 'types': Counter()},
                'low_confidence': {'threshold': 0.0, 'count': 0, 'types': Counter()}
            }
        }
        
        # Count classifications
        type_counter = Counter()
        for commit in classified_commits:
            predicted_class = commit['predicted_class']
            confidence = commit.get('classification_confidence', 0)
            
            type_counter[predicted_class] += 1
            
            # Categorize by confidence
            if confidence >= 0.8:
                breakdown['by_confidence']['high_confidence']['count'] += 1
                breakdown['by_confidence']['high_confidence']['types'][predicted_class] += 1
            elif confidence >= 0.6:
                breakdown['by_confidence']['medium_confidence']['count'] += 1
                breakdown['by_confidence']['medium_confidence']['types'][predicted_class] += 1
            else:
                breakdown['by_confidence']['low_confidence']['count'] += 1
                breakdown['by_confidence']['low_confidence']['types'][predicted_class] += 1
        
        # Calculate percentages
        total = len(classified_commits)
        for type_name, count in type_counter.items():
            breakdown['distribution'][type_name] = {
                'count': count,
                'percentage': (count / total) * 100
            }
        
        return breakdown
    
    def _generate_confidence_analysis(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate confidence score analysis."""
        classified_commits = [c for c in commits if 'predicted_class' in c]
        confidence_scores = [c.get('classification_confidence', 0) for c in classified_commits]
        
        if not confidence_scores:
            return {}
        
        return {
            'total_predictions': len(confidence_scores),
            'average_confidence': sum(confidence_scores) / len(confidence_scores),
            'min_confidence': min(confidence_scores),
            'max_confidence': max(confidence_scores),
            'high_confidence_count': sum(1 for score in confidence_scores if score >= 0.8),
            'high_confidence_percentage': (sum(1 for score in confidence_scores if score >= 0.8) / len(confidence_scores)) * 100,
            'confidence_distribution': {
                'very_high': sum(1 for score in confidence_scores if score >= 0.9),
                'high': sum(1 for score in confidence_scores if 0.8 <= score < 0.9),
                'medium': sum(1 for score in confidence_scores if 0.6 <= score < 0.8),
                'low': sum(1 for score in confidence_scores if 0.4 <= score < 0.6),
                'very_low': sum(1 for score in confidence_scores if score < 0.4)
            }
        }
    
    def _generate_temporal_patterns(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate temporal pattern analysis."""
        daily_stats = defaultdict(lambda: {
            'total_commits': 0,
            'classifications': Counter(),
            'developers': set()
        })
        
        for commit in commits:
            date_key = commit['timestamp'].date().isoformat()
            stats = daily_stats[date_key]
            
            stats['total_commits'] += 1
            stats['developers'].add(commit.get('canonical_author_name', 'unknown'))
            
            if 'predicted_class' in commit:
                stats['classifications'][commit['predicted_class']] += 1
        
        # Convert to serializable format
        temporal_data = {}
        for date_key, stats in daily_stats.items():
            temporal_data[date_key] = {
                'total_commits': stats['total_commits'],
                'classifications': dict(stats['classifications']),
                'developer_count': len(stats['developers']),
                'developers': list(stats['developers'])
            }
        
        return temporal_data
    
    def save_reports(self, reports: Dict[str, Any], output_dir: Path):
        """Save reports to files.
        
        Args:
            reports: Generated reports data
            output_dir: Directory to save reports
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive JSON report
        json_path = output_dir / f'ewtn_classification_report_{self.test_start.date()}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'test_period': {
                        'start': self.test_start.isoformat(),
                        'end': self.test_end.isoformat()
                    },
                    'config_path': str(self.config_path),
                    'analysis_results': {
                        'repositories_analyzed': self.analysis_results['repositories_analyzed'],
                        'total_commits': self.analysis_results['total_commits'],
                        'classified_commits': self.analysis_results['classified_commits'],
                        'unique_developers': len(self.analysis_results['developers']),
                        'errors': self.analysis_results['errors']
                    }
                },
                'reports': reports
            }, f, indent=2, default=str)
        
        logger.info(f"Comprehensive report saved to: {json_path}")
        
        # Save summary CSV
        csv_path = output_dir / f'ewtn_classification_summary_{self.test_start.date()}.csv'
        self._save_summary_csv(reports, csv_path)
        
        logger.info(f"Summary CSV saved to: {csv_path}")
    
    def _save_summary_csv(self, reports: Dict[str, Any], csv_path: Path):
        """Save summary data as CSV."""
        import csv
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write summary section
            writer.writerow(['EWTN Classification Analysis Summary'])
            writer.writerow([''])
            writer.writerow(['Metric', 'Value'])
            
            summary = reports.get('summary', {})
            writer.writerow(['Total Commits', summary.get('total_commits', 0)])
            writer.writerow(['Classified Commits', summary.get('classified_commits', 0)])
            writer.writerow(['Classification Coverage', f"{summary.get('classification_coverage', 0):.1%}"])
            writer.writerow(['Unique Developers', summary.get('unique_developers', 0)])
            writer.writerow(['Unique Repositories', summary.get('unique_repositories', 0)])
            
            # Write classification breakdown
            writer.writerow([''])
            writer.writerow(['Classification Breakdown'])
            writer.writerow(['Type', 'Count', 'Percentage'])
            
            breakdown = reports.get('by_classification', {}).get('distribution', {})
            for class_type, data in breakdown.items():
                writer.writerow([class_type, data['count'], f"{data['percentage']:.1f}%"])
            
            # Write confidence analysis
            writer.writerow([''])
            writer.writerow(['Confidence Analysis'])
            writer.writerow(['Metric', 'Value'])
            
            confidence = reports.get('confidence_analysis', {})
            writer.writerow(['Average Confidence', f"{confidence.get('average_confidence', 0):.3f}"])
            writer.writerow(['High Confidence Rate', f"{confidence.get('high_confidence_percentage', 0):.1f}%"])
    
    def run_analysis(self, force_retrain: bool = False) -> Dict[str, Any]:
        """Run the complete EWTN classification analysis.
        
        Args:
            force_retrain: Force model retraining even if model exists
            
        Returns:
            Dictionary containing analysis results and reports
        """
        start_time = datetime.now()
        logger.info("Starting EWTN repository classification analysis")
        
        try:
            # Step 1: Discover repositories
            repositories = self.discover_repositories()
            self.analysis_results['repositories_analyzed'] = len(repositories)
            
            if not repositories:
                logger.error("No repositories found to analyze")
                return {'error': 'No repositories found'}
            
            # Step 2: Analyze commits from all repositories
            all_commits = []
            for repo_info in repositories:
                repo_commits = self.analyze_repository_commits(repo_info)
                all_commits.extend(repo_commits)
            
            self.analysis_results['total_commits'] = len(all_commits)
            logger.info(f"Collected {len(all_commits)} commits across {len(repositories)} repositories")
            
            if not all_commits:
                logger.warning("No commits found in the specified date range")
                return {'error': 'No commits found in date range'}
            
            # Step 3: Apply identity normalization
            normalized_commits = self.apply_identity_normalization(all_commits)
            
            # Step 4: Apply classification
            if force_retrain:
                # Clear existing model to force retraining
                model_path = self.classifier.model_path / 'commit_classifier.joblib'
                if model_path.exists():
                    model_path.unlink()
                    logger.info("Cleared existing model for retraining")
            
            classified_commits = self.classify_commits(normalized_commits)
            self.analysis_results['classified_commits'] = len([c for c in classified_commits if 'predicted_class' in c])
            
            # Step 5: Generate reports
            reports = self.generate_classification_reports(classified_commits)
            
            # Step 6: Save reports
            output_dir = Path('ewtn_classification_results')
            self.save_reports(reports, output_dir)
            
            # Calculate processing time
            self.analysis_results['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Analysis completed in {self.analysis_results['processing_time']:.1f} seconds")
            
            return {
                'success': True,
                'analysis_results': self.analysis_results,
                'reports': reports,
                'output_directory': str(output_dir)
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'analysis_results': self.analysis_results
            }


def main():
    """Main entry point for the EWTN classification test script."""
    parser = argparse.ArgumentParser(
        description='Test EWTN repository classification system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_ewtn_classification.py
    python test_ewtn_classification.py --dry-run --debug
    python test_ewtn_classification.py --force-retrain
        """
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=Path.home() / 'Clients' / 'EWTN' / 'gfa' / 'config.yaml',
        help='Path to EWTN configuration file (default: ~/Clients/EWTN/gfa/config.yaml)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without making API calls or writing files'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force model retraining even if existing model found'
    )
    
    args = parser.parse_args()
    
    # Validate configuration file
    if not args.config.exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Initialize and run tester
    try:
        tester = EWTNClassificationTester(
            config_path=args.config,
            dry_run=args.dry_run,
            debug=args.debug
        )
        
        results = tester.run_analysis(force_retrain=args.force_retrain)
        
        if results.get('success'):
            print("\n✅ EWTN Classification Analysis Completed Successfully!")
            print(f"📊 Analysis Results:")
            print(f"   • Repositories Analyzed: {results['analysis_results']['repositories_analyzed']}")
            print(f"   • Total Commits: {results['analysis_results']['total_commits']}")
            print(f"   • Classified Commits: {results['analysis_results']['classified_commits']}")
            print(f"   • Unique Developers: {results['analysis_results'].get('unique_developers', len(results['analysis_results']['developers']))}")
            print(f"   • Processing Time: {results['analysis_results']['processing_time']:.1f}s")
            
            if results['analysis_results']['errors']:
                print(f"   ⚠️  Errors: {len(results['analysis_results']['errors'])}")
            
            print(f"\n📁 Reports saved to: {results['output_directory']}")
            
        else:
            print(f"\n❌ Analysis Failed: {results.get('error', 'Unknown error')}")
            if results.get('analysis_results', {}).get('errors'):
                print("Errors encountered:")
                for error in results['analysis_results']['errors'][:5]:  # Show first 5 errors
                    print(f"   • {error}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()