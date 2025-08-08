feat: release v1.1.0 with database-backed reporting and enhanced classification

## Major Features

### Database-Backed Reporting System
- Store daily metrics in SQLite for 80% faster report generation
- Weekly trend analysis showing classification pattern changes
- New DailyMetrics and WeeklyTrends database tables
- Database report generator pulls directly from SQLite

### Enhanced Commit Classification  
- All commits properly classified (feature, bug_fix, refactor, etc.)
- Removed "tracked_work" as a category - tracked commits classified normally
- Ticket information enhances classification accuracy
- Project reports show classification breakdowns

### Configuration Improvements
- Support flexible field names (api_key/openrouter_api_key, model/primary_model)
- Auto-enable qualitative analysis when configured (no CLI flag needed)
- Proper ticket platform filtering (e.g., JIRA-only)
- Cost tracking configuration mapping

### Bug Fixes
- Fixed datetime import scope issues in CLI
- Corrected ticket platform filtering
- Enhanced identity resolution for developer mappings
- Fixed classification data structure in narrative reports

## Files Changed
- src/gitflow_analytics/_version.py - Bumped to 1.1.0
- src/gitflow_analytics/cli.py - Database storage integration, HTML disabled
- src/gitflow_analytics/config.py - Flexible field mapping
- src/gitflow_analytics/core/metrics_storage.py - New storage system
- src/gitflow_analytics/models/database.py - New database models
- src/gitflow_analytics/reports/database_report_generator.py - New generator
- src/gitflow_analytics/reports/narrative_writer.py - Classification enhancements
- src/gitflow_analytics/reports/csv_writer.py - Updated classification logic
- CHANGELOG.md - Added v1.1.0 entry
- README.md - Added new features documentation
- RELEASE_NOTES_1.1.0.md - Comprehensive release notes

Co-Authored-By: Claude <noreply@anthropic.com>