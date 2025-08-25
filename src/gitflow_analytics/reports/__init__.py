# Reports package

# Legacy imports for backward compatibility
from .analytics_writer import AnalyticsReportGenerator

# New abstraction layer components
from .base import (
    BaseReportGenerator,
    ChainedReportGenerator,
    CompositeReportGenerator,
    ReportData,
    ReportMetadata,
    ReportOutput,
)
from .csv_writer import CSVReportGenerator
from .data_models import (
    CommitData,
    CommitType,
    DeveloperIdentity,
    DeveloperMetrics,
    DORAMetrics,
    ProjectMetrics,
    PullRequestData,
    ReportSummary,
    TicketMetrics,
    WeeklyMetrics,
    WorkStyle,
)
from .factory import (
    ReportBuilder,
    ReportFactory,
    create_multiple_reports,
    create_report,
    get_default_factory,
)
from .formatters import (
    CSVFormatter,
    DateFormatter,
    JSONFormatter,
    MarkdownFormatter,
    MetricFormatter,
    NumberFormatter,
    TextFormatter,
)
from .html_generator import HTMLReportGenerator
from .interfaces import ReportField, ReportFormat, ReportSchema, ReportType
from .json_exporter import ComprehensiveJSONExporter
from .narrative_writer import NarrativeReportGenerator

__all__ = [
    # Legacy generators
    'CSVReportGenerator',
    'AnalyticsReportGenerator', 
    'NarrativeReportGenerator',
    'ComprehensiveJSONExporter',
    'HTMLReportGenerator',
    
    # Base classes
    'BaseReportGenerator',
    'CompositeReportGenerator',
    'ChainedReportGenerator',
    'ReportData',
    'ReportOutput',
    'ReportMetadata',
    
    # Interfaces
    'ReportFormat',
    'ReportType',
    'ReportField',
    'ReportSchema',
    
    # Factory
    'ReportFactory',
    'ReportBuilder',
    'create_report',
    'create_multiple_reports',
    'get_default_factory',
    
    # Formatters
    'DateFormatter',
    'NumberFormatter',
    'TextFormatter',
    'MarkdownFormatter',
    'CSVFormatter',
    'JSONFormatter',
    'MetricFormatter',
    
    # Data models
    'CommitData',
    'PullRequestData',
    'DeveloperMetrics',
    'ProjectMetrics',
    'WeeklyMetrics',
    'TicketMetrics',
    'DORAMetrics',
    'ReportSummary',
    'DeveloperIdentity',
    'CommitType',
    'WorkStyle'
]