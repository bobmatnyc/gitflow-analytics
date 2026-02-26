"""CSV report generation for GitFlow Analytics."""

import csv
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..metrics.activity_scoring import ActivityScorer
from .base import BaseReportGenerator, ReportData, ReportOutput
from .interfaces import ReportFormat

# Get logger for this module
logger = logging.getLogger(__name__)



from .csv_reports_weekly import CSVWeeklyReportsMixin
from .csv_reports_developer import CSVDeveloperReportsMixin
from .csv_reports_dora import CSVDoraReportsMixin

class CSVReportGenerator(CSVWeeklyReportsMixin, CSVDeveloperReportsMixin, CSVDoraReportsMixin, BaseReportGenerator):
    """Generate CSV reports with weekly metrics."""

    def __init__(
        self,
        anonymize: bool = False,
        exclude_authors: list[str] = None,
        identity_resolver=None,
        **kwargs,
    ):
        """Initialize report generator."""
        super().__init__(
            anonymize=anonymize,
            exclude_authors=exclude_authors,
            identity_resolver=identity_resolver,
            **kwargs,
        )
        self.activity_scorer = ActivityScorer()

    # Implementation of abstract methods from BaseReportGenerator

    def generate(self, data: ReportData, output_path: Optional[Path] = None) -> ReportOutput:
        """Generate CSV report from standardized data.

        Args:
            data: Standardized report data
            output_path: Optional path to write the report to

        Returns:
            ReportOutput containing the results
        """
        try:
            # Validate data
            if not self.validate_data(data):
                return ReportOutput(success=False, errors=["Invalid or incomplete data provided"])

            # Pre-process data (apply filters and anonymization)
            data = self.pre_process(data)

            # Generate appropriate CSV based on available data
            if output_path:
                # Determine report type based on filename or available data
                filename = output_path.name.lower()

                if "weekly" in filename and data.commits:
                    self.generate_weekly_report(data.commits, data.developer_stats, output_path)
                elif "developer" in filename and data.developer_stats:
                    self.generate_developer_report(data.developer_stats, output_path)
                elif "activity" in filename and data.activity_data:
                    # Write activity data directly
                    df = pd.DataFrame(data.activity_data)
                    df.to_csv(output_path, index=False)
                elif "focus" in filename and data.focus_data:
                    # Write focus data directly
                    df = pd.DataFrame(data.focus_data)
                    df.to_csv(output_path, index=False)
                elif data.commits:
                    # Default to weekly report
                    self.generate_weekly_report(data.commits, data.developer_stats, output_path)
                else:
                    return ReportOutput(
                        success=False, errors=["No suitable data found for CSV generation"]
                    )

                # Calculate file size
                file_size = output_path.stat().st_size if output_path.exists() else 0

                return ReportOutput(
                    success=True, file_path=output_path, format="csv", size_bytes=file_size
                )
            else:
                # Generate in-memory CSV
                import io

                buffer = io.StringIO()

                # Default to generating weekly report in memory
                if data.commits:
                    # Create temporary dataframe
                    df = pd.DataFrame(
                        self._aggregate_weekly_data(
                            data.commits,
                            datetime.now(timezone.utc) - timedelta(weeks=52),
                            datetime.now(timezone.utc),
                        )
                    )
                    df.to_csv(buffer, index=False)
                    content = buffer.getvalue()

                    return ReportOutput(
                        success=True, content=content, format="csv", size_bytes=len(content)
                    )
                else:
                    return ReportOutput(
                        success=False, errors=["No data available for CSV generation"]
                    )

        except Exception as e:
            self.logger.error(f"Error generating CSV report: {e}")
            return ReportOutput(success=False, errors=[str(e)])

    def get_required_fields(self) -> list[str]:
        """Get the list of required data fields for CSV generation.

        Returns:
            List of required field names
        """
        # CSV reports can work with various combinations of data
        # At minimum, we need either commits or developer_stats
        return ["commits"]  # Primary requirement

    def get_format_type(self) -> str:
        """Get the format type this generator produces.

        Returns:
            Format identifier
        """
        return ReportFormat.CSV.value

    def _filter_excluded_authors_list(
        self, data_list: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Filter out excluded authors from any data list using canonical_id and enhanced bot detection.

        WHY: Bot exclusion happens in Phase 2 (reporting) instead of Phase 1 (data collection)
        to ensure manual identity mappings work correctly. This allows the system to see
        consolidated bot identities via canonical_id instead of just original author_email/author_name.

        ENHANCEMENT: Added enhanced bot pattern matching to catch bots that weren't properly
        consolidated via manual mappings, preventing bot leakage in reports.

        Args:
            data_list: List of data dictionaries containing canonical_id field

        Returns:
            Filtered list with excluded authors removed
        """
        if not self.exclude_authors:
            return data_list

        logger.debug(
            f"DEBUG EXCLUSION: Starting filter with {len(self.exclude_authors)} excluded authors: {self.exclude_authors}"
        )
        logger.debug(f"DEBUG EXCLUSION: Filtering {len(data_list)} items from data list")

        excluded_lower = [author.lower() for author in self.exclude_authors]
        logger.debug(f"DEBUG EXCLUSION: Excluded authors (lowercase): {excluded_lower}")

        # Separate explicit excludes from bot patterns
        explicit_excludes = []
        bot_patterns = []

        for exclude in excluded_lower:
            if "[bot]" in exclude or "bot" in exclude.split():
                bot_patterns.append(exclude)
            else:
                explicit_excludes.append(exclude)

        logger.debug(f"DEBUG EXCLUSION: Explicit excludes: {explicit_excludes}")
        logger.debug(f"DEBUG EXCLUSION: Bot patterns: {bot_patterns}")

        filtered_data = []
        excluded_count = 0

        # Sample first 5 items to see data structure
        for i, item in enumerate(data_list[:5]):
            logger.debug(
                f"DEBUG EXCLUSION: Sample item {i}: canonical_id='{item.get('canonical_id', '')}', "
                f"author_email='{item.get('author_email', '')}', author_name='{item.get('author_name', '')}', "
                f"author='{item.get('author', '')}', primary_name='{item.get('primary_name', '')}', "
                f"name='{item.get('name', '')}', developer='{item.get('developer', '')}', "
                f"display_name='{item.get('display_name', '')}'"
            )

        for item in data_list:
            canonical_id = item.get("canonical_id", "")
            # Also check original author fields as fallback for data without canonical_id
            author_email = item.get("author_email", "")
            author_name = item.get("author_name", "")

            # Check all possible author fields to ensure we catch every variation
            author = item.get("author", "")
            primary_name = item.get("primary_name", "")
            name = item.get("name", "")
            developer = item.get("developer", "")  # Common in CSV data
            display_name = item.get("display_name", "")  # Common in some data structures

            # Collect all identity fields for checking
            identity_fields = [
                canonical_id,
                item.get("primary_email", ""),
                author_email,
                author_name,
                author,
                primary_name,
                name,
                developer,
                display_name,
            ]

            should_exclude = False
            exclusion_reason = ""

            # Check for exact matches with explicit excludes first
            for field in identity_fields:
                if field and field.lower() in explicit_excludes:
                    should_exclude = True
                    exclusion_reason = f"exact match with '{field}' in explicit excludes"
                    break

            # If not explicitly excluded, check for bot patterns
            if not should_exclude:
                for field in identity_fields:
                    if not field:
                        continue
                    field_lower = field.lower()

                    # Enhanced bot detection: check if any field contains bot-like patterns
                    for bot_pattern in bot_patterns:
                        if bot_pattern in field_lower:
                            should_exclude = True
                            exclusion_reason = (
                                f"bot pattern '{bot_pattern}' matches field '{field}'"
                            )
                            break

                    # Additional bot detection: check for common bot patterns not in explicit list
                    if not should_exclude:
                        bot_indicators = [
                            "[bot]",
                            "bot@",
                            "-bot",
                            "automated",
                            "github-actions",
                            "dependabot",
                            "renovate",
                        ]
                        for indicator in bot_indicators:
                            if indicator in field_lower:
                                # Only exclude if this bot-like pattern matches something in our exclude list
                                for exclude in excluded_lower:
                                    if (
                                        indicator.replace("[", "").replace("]", "") in exclude
                                        or exclude in field_lower
                                    ):
                                        should_exclude = True
                                        exclusion_reason = f"bot indicator '{indicator}' in field '{field}' matches exclude pattern '{exclude}'"
                                        break
                                if should_exclude:
                                    break

                    if should_exclude:
                        break

            if should_exclude:
                excluded_count += 1
                logger.debug(f"DEBUG EXCLUSION: EXCLUDING item - {exclusion_reason}")
                logger.debug(
                    f"  canonical_id='{canonical_id}', primary_email='{item.get('primary_email', '')}', "
                    f"author_email='{author_email}', author_name='{author_name}', author='{author}', "
                    f"primary_name='{primary_name}', name='{name}', developer='{developer}', "
                    f"display_name='{display_name}'"
                )
            else:
                filtered_data.append(item)

        logger.debug(
            f"DEBUG EXCLUSION: Excluded {excluded_count} items, kept {len(filtered_data)} items"
        )
        return filtered_data

    def _get_canonical_display_name(self, canonical_id: str, fallback_name: str) -> str:
        """
        Get the canonical display name for a developer.

        WHY: Manual identity mappings may have updated display names that aren't
        reflected in the developer_stats data passed to report generators. This
        method ensures we get the most current display name from the identity resolver.

        Args:
            canonical_id: The canonical ID to get the display name for
            fallback_name: The fallback name to use if identity resolver is not available

        Returns:
            The canonical display name or fallback name
        """
        if self.identity_resolver and canonical_id:
            try:
                canonical_name = self.identity_resolver.get_canonical_name(canonical_id)
                if canonical_name and canonical_name != "Unknown":
                    return canonical_name
            except Exception as e:
                logger.debug(f"Error getting canonical name for {canonical_id}: {e}")

        return fallback_name

    def _log_datetime_comparison(
        self, dt1: datetime, dt2: datetime, operation: str, location: str
    ) -> None:
        """Log datetime comparison details for debugging timezone issues."""
        logger.debug(f"Comparing dates in {location} ({operation}):")
        logger.debug(f"  dt1: {dt1} (tzinfo: {dt1.tzinfo}, aware: {dt1.tzinfo is not None})")
        logger.debug(f"  dt2: {dt2} (tzinfo: {dt2.tzinfo}, aware: {dt2.tzinfo is not None})")

    def _safe_datetime_compare(
        self, dt1: datetime, dt2: datetime, operation: str, location: str
    ) -> bool:
        """Safely compare datetimes with logging and error handling."""
        try:
            self._log_datetime_comparison(dt1, dt2, operation, location)

            if operation == "lt":
                result = dt1 < dt2
            elif operation == "gt":
                result = dt1 > dt2
            elif operation == "le":
                result = dt1 <= dt2
            elif operation == "ge":
                result = dt1 >= dt2
            elif operation == "eq":
                result = dt1 == dt2
            else:
                raise ValueError(f"Unknown operation: {operation}")

            logger.debug(f"  Result: {result}")
            return result

        except TypeError as e:
            logger.error(f"Timezone comparison error in {location}:")
            logger.error(
                f"  dt1: {dt1} (type: {type(dt1)}, tzinfo: {getattr(dt1, 'tzinfo', 'N/A')})"
            )
            logger.error(
                f"  dt2: {dt2} (type: {type(dt2)}, tzinfo: {getattr(dt2, 'tzinfo', 'N/A')})"
            )
            logger.error(f"  Operation: {operation}")
            logger.error(f"  Error: {e}")

            # Import traceback for detailed error info
            import traceback

            logger.error(f"  Full traceback:\n{traceback.format_exc()}")

            # Try to fix by making both timezone-aware in UTC
            try:
                if dt1.tzinfo is None:
                    dt1 = dt1.replace(tzinfo=timezone.utc)
                    logger.debug(f"  Fixed dt1 to UTC: {dt1}")
                if dt2.tzinfo is None:
                    dt2 = dt2.replace(tzinfo=timezone.utc)
                    logger.debug(f"  Fixed dt2 to UTC: {dt2}")

                # Retry comparison
                if operation == "lt":
                    result = dt1 < dt2
                elif operation == "gt":
                    result = dt1 > dt2
                elif operation == "le":
                    result = dt1 <= dt2
                elif operation == "ge":
                    result = dt1 >= dt2
                elif operation == "eq":
                    result = dt1 == dt2
                else:
                    raise ValueError(f"Unknown operation: {operation}")

                logger.info(f"  Fixed comparison result: {result}")
                return result

            except Exception as fix_error:
                logger.error(f"  Failed to fix timezone issue: {fix_error}")
                raise

    def _safe_datetime_format(self, dt: datetime, format_str: str) -> str:
        """Safely format datetime with logging."""
        try:
            logger.debug(
                f"Formatting datetime: {dt} (tzinfo: {getattr(dt, 'tzinfo', 'N/A')}) with format {format_str}"
            )
            result = dt.strftime(format_str)
            logger.debug(f"  Format result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error formatting datetime {dt}: {e}")
            return str(dt)


    def _get_week_start(self, date: datetime) -> datetime:
        """Get Monday of the week for a given date."""
        logger.debug(
            f"Getting week start for date: {date} (tzinfo: {getattr(date, 'tzinfo', 'N/A')})"
        )

        # Ensure consistent timezone handling - keep timezone info
        if hasattr(date, "tzinfo") and date.tzinfo is not None:
            # Keep timezone-aware but ensure it's UTC
            if date.tzinfo != timezone.utc:
                date = date.astimezone(timezone.utc)
                logger.debug(f"  Converted to UTC: {date}")
        else:
            # Convert naive datetime to UTC timezone-aware
            date = date.replace(tzinfo=timezone.utc)
            logger.debug(f"  Made timezone-aware: {date}")

        days_since_monday = date.weekday()
        monday = date - timedelta(days=days_since_monday)
        result = monday.replace(hour=0, minute=0, second=0, microsecond=0)

        logger.debug(f"  Week start result: {result} (tzinfo: {result.tzinfo})")
        return result


    def _anonymize_value(self, value: str, field_type: str) -> str:
        """Anonymize a value if anonymization is enabled."""
        if not self.anonymize or not value:
            return value

        if field_type == "email" and "@" in value:
            # Keep domain for email
            local, domain = value.split("@", 1)
            value = local  # Anonymize only local part
            suffix = f"@{domain}"
        else:
            suffix = ""

        if value not in self._anonymization_map:
            self._anonymous_counter += 1
            if field_type == "name":
                anonymous = f"Developer{self._anonymous_counter}"
            elif field_type == "email":
                anonymous = f"dev{self._anonymous_counter}"
            elif field_type == "id":
                anonymous = f"ID{self._anonymous_counter:04d}"
            else:
                anonymous = f"anon{self._anonymous_counter}"

            self._anonymization_map[value] = anonymous

        return self._anonymization_map[value] + suffix

