"""Report interfaces and data contracts for GitFlow Analytics.

This module defines the interfaces and contracts that all report generators
must adhere to, ensuring consistency and interoperability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Union


class ReportFormat(Enum):
    """Supported report format types."""
    
    CSV = "csv"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    XML = "xml"
    YAML = "yaml"
    EXCEL = "excel"
    
    @classmethod
    def from_string(cls, value: str) -> "ReportFormat":
        """Create format from string value.
        
        Args:
            value: String representation of format
            
        Returns:
            ReportFormat enum value
            
        Raises:
            ValueError: If format is not recognized
        """
        value_upper = value.upper()
        if value_upper in cls.__members__:
            return cls[value_upper]
        
        # Try by value
        for member in cls:
            if member.value == value.lower():
                return member
        
        raise ValueError(f"Unknown report format: {value}")


class ReportType(Enum):
    """Types of reports that can be generated."""
    
    WEEKLY_METRICS = "weekly_metrics"
    DEVELOPER_STATS = "developer_stats"
    ACTIVITY_DISTRIBUTION = "activity_distribution"
    DEVELOPER_FOCUS = "developer_focus"
    QUALITATIVE_INSIGHTS = "qualitative_insights"
    NARRATIVE = "narrative"
    DORA_METRICS = "dora_metrics"
    BRANCH_HEALTH = "branch_health"
    STORY_POINTS = "story_points"
    UNTRACKED_COMMITS = "untracked_commits"
    WEEKLY_TRENDS = "weekly_trends"
    PR_ANALYSIS = "pr_analysis"
    COMPREHENSIVE = "comprehensive"
    CUSTOM = "custom"


@dataclass
class ReportField:
    """Definition of a report field."""
    
    name: str
    field_type: type
    required: bool = False
    default: Any = None
    description: str = ""
    validator: Optional[Callable[[Any], bool]] = None
    transformer: Optional[Callable[[Any], Any]] = None
    
    def validate(self, value: Any) -> bool:
        """Validate a field value.
        
        Args:
            value: Value to validate
            
        Returns:
            True if valid, False otherwise
        """
        if value is None:
            return not self.required
        
        if not isinstance(value, self.field_type):
            return False
        
        if self.validator:
            return self.validator(value)
        
        return True
    
    def transform(self, value: Any) -> Any:
        """Transform a field value.
        
        Args:
            value: Value to transform
            
        Returns:
            Transformed value
        """
        if self.transformer:
            return self.transformer(value)
        return value


@dataclass
class ReportSchema:
    """Schema definition for a report."""
    
    name: str
    version: str
    fields: List[ReportField]
    description: str = ""
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate data against the schema.
        
        Args:
            data: Data to validate
            
        Returns:
            True if valid, False otherwise
        """
        for field in self.fields:
            if field.required and field.name not in data:
                return False
            
            if field.name in data:
                if not field.validate(data[field.name]):
                    return False
        
        return True
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data according to the schema.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        result = {}
        
        for field in self.fields:
            if field.name in data:
                result[field.name] = field.transform(data[field.name])
            elif field.default is not None:
                result[field.name] = field.default
        
        return result


class ReportGenerator(Protocol):
    """Protocol defining the interface for report generators."""
    
    def generate(self, data: Any, output_path: Optional[Any] = None) -> Any:
        """Generate a report from the provided data."""
        ...
    
    def validate_data(self, data: Any) -> bool:
        """Validate input data."""
        ...
    
    def get_required_fields(self) -> List[str]:
        """Get list of required fields."""
        ...
    
    def get_format_type(self) -> str:
        """Get the format type produced by this generator."""
        ...


class ReportProcessor(Protocol):
    """Protocol for report processors that transform data."""
    
    def process(self, data: Any) -> Any:
        """Process data for report generation."""
        ...


class ReportFormatter(Protocol):
    """Protocol for report formatters."""
    
    def format(self, data: Any) -> str:
        """Format data for output."""
        ...


class ReportWriter(Protocol):
    """Protocol for report writers."""
    
    def write(self, content: Union[str, bytes], path: Any) -> None:
        """Write report content to storage."""
        ...


class IReportFactory(ABC):
    """Interface for report generator factories."""
    
    @abstractmethod
    def create_generator(
        self,
        report_type: ReportType,
        format_type: ReportFormat,
        **kwargs
    ) -> ReportGenerator:
        """Create a report generator.
        
        Args:
            report_type: Type of report to generate
            format_type: Format for the report
            **kwargs: Additional configuration
            
        Returns:
            Report generator instance
        """
        pass
    
    @abstractmethod
    def register_generator(
        self,
        report_type: ReportType,
        format_type: ReportFormat,
        generator_class: type
    ) -> None:
        """Register a report generator class.
        
        Args:
            report_type: Type of report
            format_type: Format type
            generator_class: Generator class to register
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self, report_type: ReportType) -> List[ReportFormat]:
        """Get supported formats for a report type.
        
        Args:
            report_type: Type of report
            
        Returns:
            List of supported formats
        """
        pass
    
    @abstractmethod
    def get_supported_reports(self) -> List[ReportType]:
        """Get list of supported report types.
        
        Returns:
            List of supported report types
        """
        pass


class IReportTemplate(ABC):
    """Interface for report templates."""
    
    @abstractmethod
    def render(self, context: Dict[str, Any]) -> str:
        """Render the template with the given context.
        
        Args:
            context: Template context data
            
        Returns:
            Rendered template string
        """
        pass
    
    @abstractmethod
    def get_required_context(self) -> List[str]:
        """Get list of required context variables.
        
        Returns:
            List of required variable names
        """
        pass
    
    @abstractmethod
    def validate_context(self, context: Dict[str, Any]) -> bool:
        """Validate template context.
        
        Args:
            context: Context to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass


class IReportAggregator(ABC):
    """Interface for report aggregators that combine multiple reports."""
    
    @abstractmethod
    def add_report(self, report_id: str, report_data: Any) -> None:
        """Add a report to the aggregation.
        
        Args:
            report_id: Unique identifier for the report
            report_data: Report data to add
        """
        pass
    
    @abstractmethod
    def aggregate(self) -> Any:
        """Aggregate all added reports.
        
        Returns:
            Aggregated report data
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all aggregated reports."""
        pass


class IReportCache(ABC):
    """Interface for report caching."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get cached report data.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data if exists, None otherwise
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Cache report data.
        
        Args:
            key: Cache key
            value: Data to cache
            ttl: Time to live in seconds
        """
        pass
    
    @abstractmethod
    def invalidate(self, key: str) -> None:
        """Invalidate cached data.
        
        Args:
            key: Cache key to invalidate
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached data."""
        pass


class IReportValidator(ABC):
    """Interface for report validators."""
    
    @abstractmethod
    def validate(self, report_data: Any, schema: Optional[ReportSchema] = None) -> bool:
        """Validate report data.
        
        Args:
            report_data: Data to validate
            schema: Optional schema to validate against
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_errors(self) -> List[str]:
        """Get validation errors.
        
        Returns:
            List of error messages
        """
        pass
    
    @abstractmethod
    def get_warnings(self) -> List[str]:
        """Get validation warnings.
        
        Returns:
            List of warning messages
        """
        pass


class IReportExporter(ABC):
    """Interface for report exporters."""
    
    @abstractmethod
    def export(
        self,
        report_data: Any,
        format_type: ReportFormat,
        output_path: Optional[Any] = None
    ) -> Any:
        """Export report data to specified format.
        
        Args:
            report_data: Data to export
            format_type: Target format
            output_path: Optional output path
            
        Returns:
            Exported data or path
        """
        pass
    
    @abstractmethod
    def supports_format(self, format_type: ReportFormat) -> bool:
        """Check if format is supported.
        
        Args:
            format_type: Format to check
            
        Returns:
            True if supported, False otherwise
        """
        pass


class IReportTransformer(ABC):
    """Interface for report data transformers."""
    
    @abstractmethod
    def transform(self, data: Any, target_schema: ReportSchema) -> Any:
        """Transform data to match target schema.
        
        Args:
            data: Input data
            target_schema: Target schema
            
        Returns:
            Transformed data
        """
        pass
    
    @abstractmethod
    def can_transform(self, source_type: type, target_schema: ReportSchema) -> bool:
        """Check if transformation is possible.
        
        Args:
            source_type: Type of source data
            target_schema: Target schema
            
        Returns:
            True if transformation is possible
        """
        pass


# Report configuration protocol
class ReportConfig(Protocol):
    """Protocol for report configuration."""
    
    format: ReportFormat
    output_path: Optional[str]
    include_metadata: bool
    anonymize: bool
    exclude_authors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        ...
    
    def validate(self) -> bool:
        """Validate configuration."""
        ...