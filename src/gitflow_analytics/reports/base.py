"""Base classes for report generation abstraction layer.

This module provides the foundation for all report generators in GitFlow Analytics,
ensuring consistency, extensibility, and maintainability across different report formats.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from ..models.database import Database

logger = logging.getLogger(__name__)


@dataclass
class ReportMetadata:
    """Metadata for report generation."""
    
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    generation_time_seconds: float = 0.0
    source_repositories: List[str] = field(default_factory=list)
    analysis_period_weeks: int = 0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    total_commits: int = 0
    total_developers: int = 0
    excluded_authors: List[str] = field(default_factory=list)
    report_version: str = "1.0.0"
    generator_name: str = ""
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportData:
    """Standardized data container for report generation.
    
    This class provides a unified interface for passing data to report generators,
    ensuring all generators have access to the same data structure.
    """
    
    # Core data
    commits: List[Dict[str, Any]] = field(default_factory=list)
    pull_requests: List[Dict[str, Any]] = field(default_factory=list)
    developer_stats: List[Dict[str, Any]] = field(default_factory=list)
    
    # Analysis results
    activity_data: List[Dict[str, Any]] = field(default_factory=list)
    focus_data: List[Dict[str, Any]] = field(default_factory=list)
    insights_data: List[Dict[str, Any]] = field(default_factory=list)
    ticket_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    pr_metrics: Dict[str, Any] = field(default_factory=dict)
    dora_metrics: Dict[str, Any] = field(default_factory=dict)
    branch_health_metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    # Project management data
    pm_data: Optional[Dict[str, Any]] = None
    story_points_data: Optional[Dict[str, Any]] = None
    
    # Qualitative analysis
    qualitative_results: List[Dict[str, Any]] = field(default_factory=list)
    chatgpt_summary: Optional[str] = None
    
    # Metadata
    metadata: ReportMetadata = field(default_factory=ReportMetadata)
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    def get_required_fields(self) -> Set[str]:
        """Get the set of required fields for basic report generation."""
        return {"commits", "developer_stats"}
    
    def validate(self) -> bool:
        """Validate that required data is present and properly formatted."""
        # Check required fields
        for field_name in self.get_required_fields():
            if not getattr(self, field_name, None):
                logger.warning(f"Required field '{field_name}' is empty or missing")
                return False
        
        # Validate commits have required fields
        if self.commits:
            required_commit_fields = {"hash", "author_email", "timestamp"}
            sample_commit = self.commits[0]
            missing_fields = required_commit_fields - set(sample_commit.keys())
            if missing_fields:
                logger.warning(f"Commits missing required fields: {missing_fields}")
                return False
        
        return True


@dataclass
class ReportOutput:
    """Container for report generation output."""
    
    success: bool
    file_path: Optional[Path] = None
    content: Optional[Union[str, bytes]] = None
    format: str = ""
    size_bytes: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseReportGenerator(ABC):
    """Abstract base class for all report generators.
    
    This class defines the interface that all report generators must implement,
    ensuring consistency across different report formats.
    """
    
    def __init__(
        self,
        anonymize: bool = False,
        exclude_authors: Optional[List[str]] = None,
        identity_resolver: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the report generator.
        
        Args:
            anonymize: Whether to anonymize developer identities
            exclude_authors: List of authors to exclude from reports
            identity_resolver: Identity resolver for consolidating developer identities
            config: Additional configuration options
        """
        self.anonymize = anonymize
        self.exclude_authors = exclude_authors or []
        self.identity_resolver = identity_resolver
        self.config = config or {}
        self._anonymization_map: Dict[str, str] = {}
        self._anonymous_counter = 0
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate(self, data: ReportData, output_path: Optional[Path] = None) -> ReportOutput:
        """Generate the report.
        
        Args:
            data: Standardized report data
            output_path: Optional path to write the report to
            
        Returns:
            ReportOutput containing the results of generation
        """
        pass
    
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Get the list of required data fields for this report generator.
        
        Returns:
            List of field names that must be present in ReportData
        """
        pass
    
    @abstractmethod
    def get_format_type(self) -> str:
        """Get the format type this generator produces.
        
        Returns:
            Format identifier (e.g., 'csv', 'markdown', 'json', 'html')
        """
        pass
    
    def validate_data(self, data: ReportData) -> bool:
        """Validate that the required data is present and properly formatted.
        
        Args:
            data: Report data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_fields = self.get_required_fields()
        
        for field_name in required_fields:
            if not hasattr(data, field_name):
                self.logger.error(f"Missing required field: {field_name}")
                return False
            
            field_value = getattr(data, field_name)
            if field_value is None:
                self.logger.error(f"Required field '{field_name}' is None")
                return False
            
            # Check if collections are empty when they shouldn't be
            if isinstance(field_value, (list, dict)) and not field_value:
                if field_name in ["commits", "developer_stats"]:  # Core required fields
                    self.logger.error(f"Required field '{field_name}' is empty")
                    return False
        
        return True
    
    def pre_process(self, data: ReportData) -> ReportData:
        """Pre-process data before report generation.
        
        This method can be overridden by subclasses to perform any necessary
        data transformation or filtering before the main generation logic.
        
        Args:
            data: Input report data
            
        Returns:
            Processed report data
        """
        # Apply author exclusions if configured
        if self.exclude_authors:
            data = self._filter_excluded_authors(data)
        
        # Apply anonymization if configured
        if self.anonymize:
            data = self._anonymize_data(data)
        
        return data
    
    def post_process(self, output: ReportOutput) -> ReportOutput:
        """Post-process the report output.
        
        This method can be overridden by subclasses to perform any necessary
        post-processing on the generated report.
        
        Args:
            output: Initial report output
            
        Returns:
            Processed report output
        """
        return output
    
    def _filter_excluded_authors(self, data: ReportData) -> ReportData:
        """Filter out excluded authors from the report data.
        
        Args:
            data: Input report data
            
        Returns:
            Filtered report data
        """
        if not self.exclude_authors:
            return data
        
        excluded_lower = [author.lower() for author in self.exclude_authors]
        
        # Filter commits
        if data.commits:
            data.commits = [
                commit for commit in data.commits
                if not self._should_exclude_author(commit, excluded_lower)
            ]
        
        # Filter developer stats
        if data.developer_stats:
            data.developer_stats = [
                dev for dev in data.developer_stats
                if not self._should_exclude_developer(dev, excluded_lower)
            ]
        
        # Update other data structures as needed
        for field_name in ["activity_data", "focus_data", "insights_data"]:
            field_value = getattr(data, field_name, None)
            if field_value:
                filtered = [
                    item for item in field_value
                    if not self._should_exclude_item(item, excluded_lower)
                ]
                setattr(data, field_name, filtered)
        
        return data
    
    def _should_exclude_author(self, commit: Dict[str, Any], excluded_lower: List[str]) -> bool:
        """Check if a commit author should be excluded.
        
        Args:
            commit: Commit data
            excluded_lower: Lowercase list of excluded authors
            
        Returns:
            True if author should be excluded
        """
        # Check canonical_id first
        canonical_id = commit.get("canonical_id", "")
        if canonical_id and canonical_id.lower() in excluded_lower:
            return True
        
        # Check other identity fields
        for field in ["author_email", "author_name", "author"]:
            value = commit.get(field, "")
            if value and value.lower() in excluded_lower:
                return True
        
        # Check for bot patterns
        author_name = commit.get("author_name", "").lower()
        author_email = commit.get("author_email", "").lower()
        
        bot_indicators = ["[bot]", "bot@", "-bot", "_bot", ".bot"]
        for indicator in bot_indicators:
            if indicator in author_name or indicator in author_email:
                return True
        
        return False
    
    def _should_exclude_developer(self, dev: Dict[str, Any], excluded_lower: List[str]) -> bool:
        """Check if a developer should be excluded.
        
        Args:
            dev: Developer data
            excluded_lower: Lowercase list of excluded authors
            
        Returns:
            True if developer should be excluded
        """
        # Check various identity fields
        identity_fields = [
            "canonical_id", "primary_email", "primary_name",
            "developer", "author", "name", "display_name"
        ]
        
        for field in identity_fields:
            value = dev.get(field, "")
            if value and value.lower() in excluded_lower:
                return True
        
        return False
    
    def _should_exclude_item(self, item: Dict[str, Any], excluded_lower: List[str]) -> bool:
        """Generic exclusion check for data items.
        
        Args:
            item: Data item to check
            excluded_lower: Lowercase list of excluded authors
            
        Returns:
            True if item should be excluded
        """
        # Try common identity fields
        identity_fields = [
            "canonical_id", "developer", "author", "author_email",
            "primary_email", "name", "display_name"
        ]
        
        for field in identity_fields:
            value = item.get(field, "")
            if value and value.lower() in excluded_lower:
                return True
        
        return False
    
    def _anonymize_data(self, data: ReportData) -> ReportData:
        """Anonymize developer identities in the report data.
        
        Args:
            data: Input report data
            
        Returns:
            Anonymized report data
        """
        # Anonymize commits
        if data.commits:
            for commit in data.commits:
                self._anonymize_commit(commit)
        
        # Anonymize developer stats
        if data.developer_stats:
            for dev in data.developer_stats:
                self._anonymize_developer(dev)
        
        # Anonymize other data structures
        for field_name in ["activity_data", "focus_data", "insights_data"]:
            field_value = getattr(data, field_name, None)
            if field_value:
                for item in field_value:
                    self._anonymize_item(item)
        
        return data
    
    def _anonymize_commit(self, commit: Dict[str, Any]) -> None:
        """Anonymize a commit record in-place.
        
        Args:
            commit: Commit data to anonymize
        """
        for field in ["author_name", "author_email", "canonical_id"]:
            if field in commit:
                commit[field] = self._get_anonymous_name(commit[field])
    
    def _anonymize_developer(self, dev: Dict[str, Any]) -> None:
        """Anonymize a developer record in-place.
        
        Args:
            dev: Developer data to anonymize
        """
        identity_fields = [
            "canonical_id", "primary_email", "primary_name",
            "developer", "author", "name", "display_name"
        ]
        
        for field in identity_fields:
            if field in dev:
                dev[field] = self._get_anonymous_name(dev[field])
    
    def _anonymize_item(self, item: Dict[str, Any]) -> None:
        """Anonymize a generic data item in-place.
        
        Args:
            item: Data item to anonymize
        """
        identity_fields = [
            "canonical_id", "developer", "author", "author_email",
            "primary_email", "name", "display_name", "author_name"
        ]
        
        for field in identity_fields:
            if field in item:
                item[field] = self._get_anonymous_name(item[field])
    
    def _get_anonymous_name(self, original: str) -> str:
        """Get an anonymous name for a given original name.
        
        Args:
            original: Original name to anonymize
            
        Returns:
            Anonymous name
        """
        if not original:
            return original
        
        if original not in self._anonymization_map:
            self._anonymous_counter += 1
            self._anonymization_map[original] = f"Developer{self._anonymous_counter:03d}"
        
        return self._anonymization_map[original]
    
    def write_to_file(self, content: Union[str, bytes], output_path: Path) -> None:
        """Write report content to a file.
        
        Args:
            content: Report content to write
            output_path: Path to write to
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(content, bytes):
            output_path.write_bytes(content)
        else:
            output_path.write_text(content, encoding="utf-8")
        
        self.logger.info(f"Report written to {output_path}")


class CompositeReportGenerator(BaseReportGenerator):
    """Generator that can produce multiple report formats in a single run."""
    
    def __init__(self, generators: List[BaseReportGenerator], **kwargs):
        """Initialize composite generator with multiple sub-generators.
        
        Args:
            generators: List of report generators to compose
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(**kwargs)
        self.generators = generators
    
    def generate(self, data: ReportData, output_path: Optional[Path] = None) -> ReportOutput:
        """Generate reports using all configured generators.
        
        Args:
            data: Report data
            output_path: Base output path (will be modified per generator)
            
        Returns:
            Composite report output
        """
        outputs = []
        errors = []
        warnings = []
        
        for generator in self.generators:
            try:
                # Determine output path for this generator
                gen_output_path = None
                if output_path:
                    suffix = self._get_suffix_for_format(generator.get_format_type())
                    gen_output_path = output_path.with_suffix(suffix)
                
                # Generate report
                output = generator.generate(data, gen_output_path)
                outputs.append(output)
                
                # Collect errors and warnings
                errors.extend(output.errors)
                warnings.extend(output.warnings)
                
            except Exception as e:
                self.logger.error(f"Error in {generator.__class__.__name__}: {e}")
                errors.append(f"{generator.__class__.__name__}: {str(e)}")
        
        # Create composite output
        return ReportOutput(
            success=all(o.success for o in outputs),
            errors=errors,
            warnings=warnings,
            metadata={"outputs": outputs}
        )
    
    def get_required_fields(self) -> List[str]:
        """Get union of all required fields from sub-generators."""
        required = set()
        for generator in self.generators:
            required.update(generator.get_required_fields())
        return list(required)
    
    def get_format_type(self) -> str:
        """Get composite format type."""
        formats = [g.get_format_type() for g in self.generators]
        return f"composite[{','.join(formats)}]"
    
    def _get_suffix_for_format(self, format_type: str) -> str:
        """Get file suffix for a given format type.
        
        Args:
            format_type: Format type identifier
            
        Returns:
            File suffix including dot
        """
        suffix_map = {
            "csv": ".csv",
            "markdown": ".md",
            "json": ".json",
            "html": ".html",
            "xml": ".xml",
            "yaml": ".yaml",
            "pdf": ".pdf"
        }
        return suffix_map.get(format_type, f".{format_type}")


class ChainedReportGenerator(BaseReportGenerator):
    """Generator that chains multiple generators, passing output of one as input to the next."""
    
    def __init__(self, generators: List[BaseReportGenerator], **kwargs):
        """Initialize chained generator.
        
        Args:
            generators: Ordered list of generators to chain
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(**kwargs)
        self.generators = generators
    
    def generate(self, data: ReportData, output_path: Optional[Path] = None) -> ReportOutput:
        """Generate reports in sequence, chaining outputs.
        
        Args:
            data: Initial report data
            output_path: Final output path
            
        Returns:
            Final report output
        """
        current_data = data
        outputs = []
        
        for i, generator in enumerate(self.generators):
            try:
                # Generate report
                is_last = (i == len(self.generators) - 1)
                gen_output_path = output_path if is_last else None
                
                output = generator.generate(current_data, gen_output_path)
                outputs.append(output)
                
                if not output.success:
                    return ReportOutput(
                        success=False,
                        errors=[f"Chain broken at {generator.__class__.__name__}"] + output.errors,
                        metadata={"completed_steps": outputs}
                    )
                
                # Transform output to input for next generator if not last
                if not is_last and output.content:
                    current_data = self._transform_output_to_input(output, current_data)
                
            except Exception as e:
                self.logger.error(f"Error in chain at {generator.__class__.__name__}: {e}")
                return ReportOutput(
                    success=False,
                    errors=[f"Chain error at {generator.__class__.__name__}: {str(e)}"],
                    metadata={"completed_steps": outputs}
                )
        
        # Return the final output
        return outputs[-1] if outputs else ReportOutput(success=False, errors=["No generators in chain"])
    
    def get_required_fields(self) -> List[str]:
        """Get required fields from first generator in chain."""
        return self.generators[0].get_required_fields() if self.generators else []
    
    def get_format_type(self) -> str:
        """Get format type of final generator in chain."""
        return self.generators[-1].get_format_type() if self.generators else "unknown"
    
    def _transform_output_to_input(self, output: ReportOutput, original_data: ReportData) -> ReportData:
        """Transform generator output to input for next generator.
        
        Args:
            output: Output from previous generator
            original_data: Original input data
            
        Returns:
            Transformed data for next generator
        """
        # Default implementation: add output content to additional data
        new_data = ReportData(
            commits=original_data.commits,
            pull_requests=original_data.pull_requests,
            developer_stats=original_data.developer_stats,
            activity_data=original_data.activity_data,
            focus_data=original_data.focus_data,
            insights_data=original_data.insights_data,
            ticket_analysis=original_data.ticket_analysis,
            pr_metrics=original_data.pr_metrics,
            dora_metrics=original_data.dora_metrics,
            branch_health_metrics=original_data.branch_health_metrics,
            pm_data=original_data.pm_data,
            story_points_data=original_data.story_points_data,
            qualitative_results=original_data.qualitative_results,
            chatgpt_summary=original_data.chatgpt_summary,
            metadata=original_data.metadata,
            config=original_data.config
        )
        
        # Add previous output to config for next generator
        new_data.config["previous_output"] = output
        
        return new_data