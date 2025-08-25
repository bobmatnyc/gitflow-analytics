# Changelog

All notable changes to GitFlow Analytics will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.2] - 2025-01-07

### Fixed
- Fixed repositories not being updated from remote before analysis
- Added automatic git fetch/pull before analyzing repositories
- Impact: Ensures latest commits are included in analysis (fixes EWTN missing commits issue)

## [1.2.1] - 2025-01-07

### Fixed
- Fixed commits not being stored in CachedCommit table during fetch step
- Fixed narrative report generation when CSV generation is disabled (now default)
- Fixed canonical_id not being set on commits loaded from database
- Fixed timezone comparison issues in batch classification
- Fixed missing `classify_commits_batch` method in LLMCommitClassifier
- Fixed complexity_delta None value handling in narrative reports
- Fixed LLM classification to properly use API keys from .env files

### Added
- Added token tracking and cost display for LLM classification
- Added LLM usage statistics display after batch classification
- Shows model, API calls, total tokens, cost, and cache hits
- Improved error handling in commit storage with detailed logging

### Changed
- Commits are now properly stored in CachedCommit table during data fetch
- Identity resolver now updates canonical_id on commits for proper attribution
- Batch classifier now correctly queries commits with timezone-aware filtering

## [1.2.0] - 2025-01-07

### Added
- Two-step process (fetch then classify) is now the default behavior for better performance and cost efficiency
- Automatic data fetching when using batch classification mode
- New `--use-legacy-classification` flag to use the old single-step process if needed

### Changed
- `analyze` command now uses two-step process by default (fetch raw data, then classify)
- `--use-batch-classification` is now enabled by default (was previously opt-in)
- Improved messaging to clearly indicate Step 1 (fetch) and Step 2 (classify) operations
- Better integration between fetch and analyze operations for seamless user experience

### Fixed
- Fixed JIRA integration error: "'IntegrationOrchestrator' object has no attribute 'jira'"
- Corrected attribute access to use `orchestrator.integrations.get('jira')` instead of `orchestrator.jira`
- Fixed batch classification mode to automatically perform data fetching when needed

### Performance
- Two-step process reduces LLM costs by batching classification requests
- Faster subsequent runs when data is already fetched and cached
- More efficient processing of large repositories with many commits

## [1.1.0] - 2025-01-06

### Added
- Database-backed reporting system with SQLite storage for daily metrics
- Weekly trend analysis showing week-over-week changes in classification patterns
- Commit classification breakdown in project activity sections of narrative reports
- Support for flexible configuration field names (api_key/openrouter_api_key, model/primary_model)
- Auto-enable qualitative analysis when configured (no CLI flag needed)
- New `DailyMetrics` and `WeeklyTrends` database tables
- Database report generator that pulls directly from SQLite
- Per-developer and per-project classification metrics
- Cost tracking configuration mapping from cost_tracking.daily_budget_usd

### Changed
- All commits now properly classified into meaningful categories (feature, bug_fix, refactor, etc.)
- Tracked commits no longer use "tracked_work" as a category - properly classified instead
- Ticket information now enhances classification accuracy
- Ticket coverage displayed separately from classifications as a process metric
- HTML report temporarily disabled pending redesign (code preserved)
- Improved configuration field mapping for better compatibility

### Fixed
- Ticket platform filtering now properly respects configuration (e.g., JIRA-only)
- DateTime import scope issues in CLI module
- Classification data structure in narrative reports
- Identity resolution for developer mappings
- Qualitative analysis auto-enablement from configuration

### Performance
- Database caching reduces report generation time by up to 80%
- Batch processing for daily metrics storage
- Optimized queries with proper indexing
- Pre-calculated weekly trends for instant retrieval

## [1.0.7] - 2025-08-01

### Fixed
- Fixed timezone comparison error when sorting deployments in DORA metrics
- Added proper timezone normalization for all timestamps before sorting
- Improved handling of None timestamps in DORA calculations

## [1.0.6] - 2025-08-01

### Fixed
- Fixed timezone comparison errors in DORA metrics calculation
- Added comprehensive timezone handling for all deployment and PR timestamps
- Enhanced debug logging to trace analysis pipeline stages

## [1.0.5] - 2025-08-01

### Fixed
- Fixed DEBUG logging not appearing due to logger configuration issues
- Added timestamp normalization to ensure all Git commits use UTC timezone
- Enhanced debug output to identify which report fails

## [1.0.4] - 2025-08-01

### Added
- Structured logging with --log option (none|INFO|DEBUG)
- Enhanced timezone error debugging capabilities
- Safe datetime comparison functions

### Fixed
- Improved debugging output for timezone-related issues
- Better error messages for datetime comparison failures

## [1.0.3] - 2025-08-01

### Fixed
- Fixed comprehensive timezone comparison issues in database queries and report generation
- Improved timezone-aware datetime handling across all components
- Fixed timezone-related errors that were still affecting v1.0.2

## [1.0.2] - 2025-08-01

### Fixed
- Fixed SQLite index naming conflicts that could cause database errors
- Fixed PR cache UNIQUE constraint errors with proper upsert logic
- Fixed timezone comparison errors in report generation
- Added loading screen to TUI (before abandoning TUI approach)
- Moved Rich to core dependencies for better CLI output

## [1.0.1] - 2025-07-31

### Added
- Path exclusion support for filtering boilerplate/generated files from line count metrics
  - Configurable via `analysis.exclude.paths` in YAML configuration
  - Default exclusions for common patterns (node_modules, lock files, minified files, etc.)
  - Filtered metrics available as `filtered_insertions`, `filtered_deletions`, `filtered_files_changed`
- JIRA integration for fetching story points from tickets
  - Configurable story point field names via `jira_integration.story_point_fields`
  - Automatic story point extraction from JIRA tickets referenced in commits
  - Support for custom field IDs and field names
- Organization-based repository discovery from GitHub
  - Automatic discovery of all non-archived repositories in an organization
  - No manual repository configuration needed for organization-wide analysis
- Ticket platform filtering via `analysis.ticket_platforms`
  - Ability to track only specific platforms (e.g., only JIRA, ignoring GitHub Issues)
- Enhanced `.env` file support
  - Automatic loading from configuration directory
  - Validation of required environment variables
  - Clear error messages for missing credentials
- New CLI command: `discover-jira-fields` to find custom field IDs

### Changed
- All report generators now use filtered line counts when available
- Cache and output directories now default to config file location (not current directory)
- Improved developer identity resolution with better consolidation

### Fixed
- Timezone comparison errors between GitHub and local timestamps
- License configuration in pyproject.toml for PyPI compatibility
- Manual identity mapping format validation
- Linting errors for better code quality

### Documentation
- Added comprehensive environment variable configuration guide
- Complete configuration examples with `.env` and YAML files
- Path exclusion documentation with default patterns
- Updated README with clearer setup instructions

## [1.0.0] - 2025-07-29

### Added
- Initial release of GitFlow Analytics
- Core Git repository analysis with batch processing
- Developer identity resolution with fuzzy matching
- Manual identity mapping support
- Story point extraction from commit messages
- Multi-platform ticket tracking (GitHub, JIRA, Linear, ClickUp)
- Comprehensive caching system with SQLite
- CSV report generation:
  - Weekly metrics
  - Developer statistics
  - Activity distribution
  - Developer focus analysis
  - Qualitative insights
- Markdown narrative reports with insights
- JSON export for API integration
- DORA metrics calculation:
  - Deployment frequency
  - Lead time for changes
  - Mean time to recovery
  - Change failure rate
- GitHub PR enrichment (optional)
- Branch to project mapping
- YAML configuration with environment variable support
- Progress bars for long operations
- Anonymization support for reports

### Configuration Features
- Repository definitions with project keys
- Story point extraction patterns
- Developer identity similarity threshold
- Manual identity mappings
- Default ticket platform specification
- Branch mapping rules
- Output format selection
- Cache TTL configuration

### Developer Experience
- Clear CLI with helpful error messages
- Comprehensive documentation
- Sample configuration files
- Progress indicators during analysis
- Detailed logging of operations

[1.0.7]: https://github.com/bobmatnyc/gitflow-analytics/releases/tag/v1.0.7
[1.0.6]: https://github.com/bobmatnyc/gitflow-analytics/releases/tag/v1.0.6
[1.0.5]: https://github.com/bobmatnyc/gitflow-analytics/releases/tag/v1.0.5
[1.0.4]: https://github.com/bobmatnyc/gitflow-analytics/releases/tag/v1.0.4
[1.0.3]: https://github.com/bobmatnyc/gitflow-analytics/releases/tag/v1.0.3
[1.0.2]: https://github.com/bobmatnyc/gitflow-analytics/releases/tag/v1.0.2
[1.0.1]: https://github.com/bobmatnyc/gitflow-analytics/releases/tag/v1.0.1
[1.0.0]: https://github.com/bobmatnyc/gitflow-analytics/releases/tag/v1.0.0
