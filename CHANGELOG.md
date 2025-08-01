# Changelog

All notable changes to GitFlow Analytics will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.0.3]: https://github.com/bobmatnyc/gitflow-analytics/releases/tag/v1.0.3
[1.0.2]: https://github.com/bobmatnyc/gitflow-analytics/releases/tag/v1.0.2
[1.0.1]: https://github.com/bobmatnyc/gitflow-analytics/releases/tag/v1.0.1
[1.0.0]: https://github.com/bobmatnyc/gitflow-analytics/releases/tag/v1.0.0