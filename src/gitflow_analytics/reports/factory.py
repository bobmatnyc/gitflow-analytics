"""Report factory for creating and managing report generators.

This module implements the factory pattern for report generation,
allowing dynamic creation and registration of report generators.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from .base import BaseReportGenerator, CompositeReportGenerator, ReportData, ReportOutput
from .interfaces import IReportFactory, ReportFormat, ReportType

logger = logging.getLogger(__name__)


class ReportGeneratorRegistry:
    """Registry for report generator classes."""
    
    def __init__(self):
        """Initialize the registry."""
        self._generators: Dict[tuple, Type[BaseReportGenerator]] = {}
        self._aliases: Dict[str, tuple] = {}
    
    def register(
        self,
        report_type: ReportType,
        format_type: ReportFormat,
        generator_class: Type[BaseReportGenerator],
        alias: Optional[str] = None
    ) -> None:
        """Register a report generator class.
        
        Args:
            report_type: Type of report
            format_type: Format type
            generator_class: Generator class to register
            alias: Optional alias for the generator
        """
        key = (report_type, format_type)
        self._generators[key] = generator_class
        
        if alias:
            self._aliases[alias] = key
        
        logger.debug(f"Registered {generator_class.__name__} for {report_type.value}/{format_type.value}")
    
    def get(
        self,
        report_type: Union[ReportType, str],
        format_type: Union[ReportFormat, str]
    ) -> Optional[Type[BaseReportGenerator]]:
        """Get a registered generator class.
        
        Args:
            report_type: Type of report or alias
            format_type: Format type
            
        Returns:
            Generator class if registered, None otherwise
        """
        # Handle string inputs
        if isinstance(report_type, str):
            if report_type in self._aliases:
                key = self._aliases[report_type]
            else:
                try:
                    report_type = ReportType(report_type)
                except ValueError:
                    return None
        
        if isinstance(format_type, str):
            try:
                format_type = ReportFormat.from_string(format_type)
            except ValueError:
                return None
        
        if isinstance(report_type, ReportType) and isinstance(format_type, ReportFormat):
            key = (report_type, format_type)
            return self._generators.get(key)
        
        return None
    
    def get_formats_for_report(self, report_type: ReportType) -> List[ReportFormat]:
        """Get all registered formats for a report type.
        
        Args:
            report_type: Type of report
            
        Returns:
            List of supported formats
        """
        formats = []
        for (r_type, f_type), _ in self._generators.items():
            if r_type == report_type:
                formats.append(f_type)
        return formats
    
    def get_report_types(self) -> List[ReportType]:
        """Get all registered report types.
        
        Returns:
            List of report types
        """
        types = set()
        for (r_type, _), _ in self._generators.items():
            types.add(r_type)
        return list(types)
    
    def clear(self) -> None:
        """Clear all registrations."""
        self._generators.clear()
        self._aliases.clear()


class ReportFactory(IReportFactory):
    """Factory for creating report generators."""
    
    def __init__(self):
        """Initialize the factory."""
        self._registry = ReportGeneratorRegistry()
        self._default_config: Dict[str, Any] = {}
        self._register_default_generators()
    
    def create_generator(
        self,
        report_type: Union[ReportType, str],
        format_type: Union[ReportFormat, str],
        **kwargs
    ) -> BaseReportGenerator:
        """Create a report generator.
        
        Args:
            report_type: Type of report to generate
            format_type: Format for the report
            **kwargs: Additional configuration passed to generator
            
        Returns:
            Report generator instance
            
        Raises:
            ValueError: If no generator is registered for the combination
        """
        # Convert strings to enums if needed
        if isinstance(report_type, str):
            try:
                report_type = ReportType(report_type)
            except ValueError:
                # Check if it's an alias
                pass
        
        if isinstance(format_type, str):
            format_type = ReportFormat.from_string(format_type)
        
        # Get generator class
        generator_class = self._registry.get(report_type, format_type)
        
        if not generator_class:
            raise ValueError(
                f"No generator registered for {report_type}/{format_type}. "
                f"Available types: {self.get_supported_reports()}"
            )
        
        # Merge with default config
        config = {**self._default_config, **kwargs}
        
        # Create and return instance
        return generator_class(**config)
    
    def create_composite_generator(
        self,
        report_types: List[tuple[Union[ReportType, str], Union[ReportFormat, str]]],
        **kwargs
    ) -> CompositeReportGenerator:
        """Create a composite generator for multiple report types.
        
        Args:
            report_types: List of (report_type, format_type) tuples
            **kwargs: Configuration passed to all generators
            
        Returns:
            Composite generator instance
        """
        generators = []
        
        for report_type, format_type in report_types:
            generator = self.create_generator(report_type, format_type, **kwargs)
            generators.append(generator)
        
        return CompositeReportGenerator(generators, **kwargs)
    
    def register_generator(
        self,
        report_type: ReportType,
        format_type: ReportFormat,
        generator_class: Type[BaseReportGenerator],
        alias: Optional[str] = None
    ) -> None:
        """Register a report generator class.
        
        Args:
            report_type: Type of report
            format_type: Format type
            generator_class: Generator class to register
            alias: Optional alias for quick access
        """
        self._registry.register(report_type, format_type, generator_class, alias)
    
    def get_supported_formats(self, report_type: ReportType) -> List[ReportFormat]:
        """Get supported formats for a report type.
        
        Args:
            report_type: Type of report
            
        Returns:
            List of supported formats
        """
        return self._registry.get_formats_for_report(report_type)
    
    def get_supported_reports(self) -> List[ReportType]:
        """Get list of supported report types.
        
        Returns:
            List of supported report types
        """
        return self._registry.get_report_types()
    
    def set_default_config(self, config: Dict[str, Any]) -> None:
        """Set default configuration for all generators.
        
        Args:
            config: Default configuration dictionary
        """
        self._default_config = config
    
    def _register_default_generators(self) -> None:
        """Register the default set of report generators."""
        # Import here to avoid circular dependencies
        try:
            from .analytics_writer import AnalyticsReportGenerator
            from .csv_writer import CSVReportGenerator
            from .json_exporter import ComprehensiveJSONExporter
            from .narrative_writer import NarrativeReportGenerator

            # Register CSV generators
            self.register_generator(
                ReportType.WEEKLY_METRICS,
                ReportFormat.CSV,
                CSVReportGenerator
            )
            self.register_generator(
                ReportType.DEVELOPER_STATS,
                ReportFormat.CSV,
                CSVReportGenerator
            )
            
            # Register analytics generators
            self.register_generator(
                ReportType.ACTIVITY_DISTRIBUTION,
                ReportFormat.CSV,
                AnalyticsReportGenerator
            )
            self.register_generator(
                ReportType.DEVELOPER_FOCUS,
                ReportFormat.CSV,
                AnalyticsReportGenerator
            )
            self.register_generator(
                ReportType.QUALITATIVE_INSIGHTS,
                ReportFormat.CSV,
                AnalyticsReportGenerator
            )
            
            # Register narrative generator
            self.register_generator(
                ReportType.NARRATIVE,
                ReportFormat.MARKDOWN,
                NarrativeReportGenerator
            )
            
            # Register JSON exporter for all types
            for report_type in ReportType:
                self.register_generator(
                    report_type,
                    ReportFormat.JSON,
                    ComprehensiveJSONExporter
                )
            
            logger.info("Default report generators registered")
            
        except ImportError as e:
            logger.warning(f"Could not import default generators: {e}")


class ReportBuilder:
    """Builder pattern for constructing complex report configurations."""
    
    def __init__(self, factory: Optional[ReportFactory] = None):
        """Initialize the builder.
        
        Args:
            factory: Report factory to use (creates default if None)
        """
        self.factory = factory or ReportFactory()
        self._report_types: List[tuple[ReportType, ReportFormat]] = []
        self._config: Dict[str, Any] = {}
        self._output_dir: Optional[Path] = None
        self._data: Optional[ReportData] = None
    
    def add_report(
        self,
        report_type: Union[ReportType, str],
        format_type: Union[ReportFormat, str]
    ) -> "ReportBuilder":
        """Add a report to generate.
        
        Args:
            report_type: Type of report
            format_type: Format type
            
        Returns:
            Self for chaining
        """
        if isinstance(report_type, str):
            report_type = ReportType(report_type)
        if isinstance(format_type, str):
            format_type = ReportFormat.from_string(format_type)
        
        self._report_types.append((report_type, format_type))
        return self
    
    def with_config(self, **kwargs) -> "ReportBuilder":
        """Add configuration options.
        
        Args:
            **kwargs: Configuration options
            
        Returns:
            Self for chaining
        """
        self._config.update(kwargs)
        return self
    
    def with_output_dir(self, output_dir: Union[str, Path]) -> "ReportBuilder":
        """Set output directory.
        
        Args:
            output_dir: Output directory path
            
        Returns:
            Self for chaining
        """
        self._output_dir = Path(output_dir)
        return self
    
    def with_data(self, data: ReportData) -> "ReportBuilder":
        """Set report data.
        
        Args:
            data: Report data
            
        Returns:
            Self for chaining
        """
        self._data = data
        return self
    
    def build(self) -> Union[BaseReportGenerator, CompositeReportGenerator]:
        """Build the report generator(s).
        
        Returns:
            Single generator or composite generator
            
        Raises:
            ValueError: If no reports have been added
        """
        if not self._report_types:
            raise ValueError("No reports added to builder")
        
        if len(self._report_types) == 1:
            report_type, format_type = self._report_types[0]
            return self.factory.create_generator(report_type, format_type, **self._config)
        else:
            return self.factory.create_composite_generator(self._report_types, **self._config)
    
    def generate(self) -> Union[ReportOutput, List[ReportOutput]]:
        """Build and generate reports.
        
        Returns:
            Report output(s)
            
        Raises:
            ValueError: If data has not been set
        """
        if not self._data:
            raise ValueError("Report data has not been set")
        
        generator = self.build()
        
        if self._output_dir:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            
            if isinstance(generator, CompositeReportGenerator):
                # Generate multiple reports with appropriate names
                outputs = []
                for (report_type, format_type), gen in zip(self._report_types, generator.generators):
                    filename = f"{report_type.value}.{format_type.value}"
                    output_path = self._output_dir / filename
                    output = gen.generate(self._data, output_path)
                    outputs.append(output)
                return outputs
            else:
                # Single report
                report_type, format_type = self._report_types[0]
                filename = f"{report_type.value}.{format_type.value}"
                output_path = self._output_dir / filename
                return generator.generate(self._data, output_path)
        else:
            return generator.generate(self._data)


# Singleton factory instance
_default_factory: Optional[ReportFactory] = None


def get_default_factory() -> ReportFactory:
    """Get the default report factory instance.
    
    Returns:
        Default factory instance
    """
    global _default_factory
    if _default_factory is None:
        _default_factory = ReportFactory()
    return _default_factory


def create_report(
    report_type: Union[ReportType, str],
    format_type: Union[ReportFormat, str],
    data: ReportData,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> ReportOutput:
    """Convenience function to create and generate a report.
    
    Args:
        report_type: Type of report
        format_type: Format type
        data: Report data
        output_path: Optional output path
        **kwargs: Additional configuration
        
    Returns:
        Report output
    """
    factory = get_default_factory()
    generator = factory.create_generator(report_type, format_type, **kwargs)
    
    if output_path:
        output_path = Path(output_path)
    
    return generator.generate(data, output_path)


def create_multiple_reports(
    reports: List[tuple[Union[ReportType, str], Union[ReportFormat, str]]],
    data: ReportData,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> List[ReportOutput]:
    """Convenience function to create multiple reports.
    
    Args:
        reports: List of (report_type, format_type) tuples
        data: Report data
        output_dir: Optional output directory
        **kwargs: Additional configuration
        
    Returns:
        List of report outputs
    """
    builder = ReportBuilder()
    
    for report_type, format_type in reports:
        builder.add_report(report_type, format_type)
    
    builder.with_config(**kwargs)
    builder.with_data(data)
    
    if output_dir:
        builder.with_output_dir(output_dir)
    
    result = builder.generate()
    
    if isinstance(result, list):
        return result
    else:
        return [result]