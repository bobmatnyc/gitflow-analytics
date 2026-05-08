# Changelog

All notable changes to GitFlow Analytics will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Performance
- Eliminate redundant API fetches across incremental data pulls — every
  unnecessary call costs money and consumes rate-limit budget:
  - **GitHub open-PR refresh now has a per-PR TTL guard.** Previously
    `_refresh_stale_open_prs()` re-fetched up to 50 open PRs on every run with
    no freshness check. PRs whose `cached_at` is newer than
    `open_pr_refresh_ttl_hours` (default 1.0) are now skipped. Configurable via
    `github.open_pr_refresh_ttl_hours` in YAML; set to `0` to disable.
  - **GitHub PR pagination now anchors to a cache watermark.** Re-introduced
    `since = max(MAX(cached_at), requested_since)` so back-to-back runs no
    longer re-paginate from the user-supplied `start_date`. `cache_pr()` upsert
    is idempotent so any overlap is safe. `--backfill-since` /
    `--backfill-prs-since` bypass this optimization.
  - **Confluence space scan now respects an incremental watermark + TTL.**
    Mirrors the JIRA-activity `_get_effective_since` pattern: scan is skipped
    entirely when the last successful scan completed within
    `full_scan_ttl_hours` (default 1.0); otherwise the lower bound advances to
    `max(last_processed, requested_since)` to avoid re-scanning unchanged pages.
- Replace deprecated `datetime.utcnow()` in `JIRAIntegration._is_ticket_stale`
  with timezone-aware `datetime.now(timezone.utc)`; naive cached_at values are
  normalized to UTC consistently.

## [3.16.1] - 2026-05-08

### Added
- **Tier-1.5 issuetype classifier**: Issue-linked commits are now classified via their JIRA/GH ticket `issuetype` field (confidence 0.90) before falling through to the LLM — eliminates LLM overhead for commits where the ticket type already encodes the answer (#68)
- `IssueCache.issue_type` column (v17 migration) — `issuetype` extracted from JIRA API response during sync and stored as a queryable top-level field
- `business_domain` now populated from ticket `components` / `labels` for issue-linked commits (was always "unknown")
- Issuetype → change_type mapping: Bug→bugfix, Story/Feature/Epic→feature, Task+platform-label→maintenance, Task+refactor-label→refactor, Documentation→documentation, Test→test; ambiguous Task/Sub-task falls through to LLM

### Fixed
- "Platform work" metric no longer collapses to 0% — issue-linked commits with Bug/Story/Task issuetype bypass the gitflow_cache rule-based path (which cannot produce "Platform work") and use the authoritative ticket signal instead

## [3.16.0] - 2026-05-07

### Added
- **Classification overrides**: `classification_overrides` table + `gfa override set/list/remove` CLI commands (#63)
- **AI detection cache**: AI-detection results stored to cache DB + `gfa backfill-ai-detection` CLI command (#47)
- **Revert tracking**: `is_revert` field in `cached_commits` + `reversion_commits` count in metrics (#64)
- **Coverage warnings**: Classification coverage warnings + `--validate-coverage` / `--coverage-threshold` flags (#65)
- **JIRA tier-3 classifier**: JIRA project-key → `work_type` mapping via `jira_project_mappings` config (#62)
- **AI footer detection**: Made-with-AI trailer detection for Cursor, Claude, Copilot commit footers (#61)
- **PR metrics**: `pr_merge_rate` and `avg_cycle_time_hrs` fields in `weekly_pr_metrics` (#66)
- **Ticket IDs from PR titles**: Extract ticket IDs from PR title text + regex false-positive suppression (#54)

### Changed
- Performance: TTL guards + watermark anchors to eliminate redundant API fetches
- `gfa backfill-ticket-ids` extended to scan PR titles in addition to commit messages

### Fixed
- Pyright type errors across test files (spacy import guard, revert detection imports)

## [3.15.2] - 2026-05-06
### Fixed
- Resolve Pyright type errors in reports factory and example_usage introduced
  by 3.15.1 formatting refactor:
  - `factory.py`: widen `register_generator` `generator_class` parameter to
    `type[Any]` (generators conform via Protocol, not strict inheritance);
    use `Sequence` for covariant `report_types` parameter; guard
    empty-string `output_path` against `Path | None` typing
  - `example_usage.py`: match `generate()` override signatures to base
    `Path | None`; use `Optional[Path]` in custom generator constructors
  - `test_github_username_sync.py`: silence unused `params`/`timeout`
    parameters in mock callbacks via `del` (preserves keyword arg matching
    expected by `requests.Session.get` mocks)

## [3.15.1] - 2026-05-06
### Changed
- Apply Black/Ruff formatting across the `reports` module and JIRA/Confluence
  integrations: consistent double-quote style, one-argument-per-line call
  sites, and collapsed implicit string concatenations into single f-strings
- Modernise `reports` module type annotations from `typing.List`/`Dict`/`Type`
  to built-in `list`/`dict`/`type` (Python 3.10+ style)

### Fixed
- `factory.py`: replace bare `try/except/pass` with `contextlib.suppress`
- `example_usage.py`: add missing local import of `BaseReportGenerator` /
  `ReportOutput` in `example_template_based_generation()`
- `base.py`: rename loop variable `field` → `field_name` to avoid shadowing
  `dataclasses.field`; flatten nested `if` to single compound condition
- `formatters.py`: add missing `import os`
- `csv_reports_dora.py`: add missing `import csv`
- `test_github_username_sync`: update mock `_get()` signatures from
  `_params`/`_timeout` to `params`/`timeout` to match production keyword args
- `test_jira_activity_integration`: migrate `_fetch_issues` mocks from
  `_get_with_retries` to `_post_with_retries` (POST `/rest/api/3/search/jql`);
  update pagination shape to `isLast`/`nextPageToken`
- `test_pr_reporting`: replace `.replace(day=N+2)` arithmetic with `timedelta`
  addition to prevent `ValueError` when running near month boundaries

## [3.15.0] - 2026-05-05
### Changed
- `UnifiedIssue.story_points` widened from `int` to `float`; JIRA adapter now preserves fractional values (e.g., 3.5 instead of 3). SQLite schema updated from INTEGER to REAL. Teams using modified Fibonacci scales will now see correct values in reports. (#56)

## [3.14.24] - 2026-05-05
### Fixed
- #55: `--backfill-since` now applies to both commit fetching and PR fetching
  - Previously, `--backfill-since` only backfilled commits; `pull_request_cache` remained empty for historical dates
  - Added `--backfill-prs-since YYYY-MM-DD` flag for PR-only window override (takes priority over `--backfill-since` for the PR fetch)
  - Adds end-to-end regression test ensuring `--backfill-since` threads through to `enrich_repository_data`

## [3.14.22] - 2026-04-29
### Added
- feat: add `commit_count` and `ticket_ids` columns to `pull_request_cache` (#53)
  - `commit_count INTEGER` — `len(commit_hashes)`, populated automatically at fetch time with no additional API calls
  - `ticket_ids JSON` — deduplicated list of JIRA-style ticket IDs (e.g. `["DUE-1234", "CORE-567"]`) extracted from all commit messages in the PR via `pr.get_commits()` during enrichment
- feat: add `gfa backfill-ticket-ids` command to populate `ticket_ids` and `commit_count` on existing cached PRs using `cached_commits.message` — no GitHub API calls, idempotent

## [3.14.18] - 2026-04-29
### Added
- feat: add `--backfill-since YYYY-MM-DD` to `gfa fetch` and `gfa analyze` for historical PR hydration (#52)
  - Fetches all merged PRs from the GitHub API back to the specified date
  - Bypasses the incremental fetch gate that previously blocked historical fetches
  - Auto-triggers `weekly_pr_metrics` rollup for the same date range
  - Idempotent — safe to re-run with the same date
  - Does not change default (incremental) behavior

## [3.14.8] - 2026-04-23
### Fixed
- #43: Pass cache/since/until to CSVReportGenerator in gfa analyze path
- #44: Populate github_username from noreply emails and config manual_mappings

## [3.14.7] - 2026-04-23
### Fixed
- #42: Thread cache/since/until into CSVReportGenerator (gfa report path)

## [3.14.6] - 2026-04-23
### Fixed
- #41: Resolve canonical identity for ticketing_score lookup in developer CSV

## [3.14.5] - 2026-04-23
### Fixed
- #40: Add ticketing_score column to developer activity CSV output

## [3.14.4] - 2026-04-23
### Fixed
- #39: Guard against NULL total_commits in identity sort and merge

## [3.14.3] - 2026-04-23
### Added
- #37: ActivityScorer ticketing_weight blends ticketing_score into raw_activity_score
- #38: JIRAActivityIntegration fetches JIRA issues/comments via JQL

## [3.14.2] - 2026-04-22
### Added
- Boilerplate filter: flag/exclude bulk auto-generated commits from velocity metrics

## [3.14.1] - 2026-04-22
### Fixed
- #32: Wire ticketing reports into gfa analyze (github_issues_summary.json, confluence_activity_summary.json, ticketing_activity_summary.json now produced without separate gfa report run)
- #33: Fix Confluence 401 — warn on unset env vars, pre-flight credential check on init
- #34: UNIQUE constraint violations now log at DEBUG not ERROR

## [3.14.0] - 2026-04-22
### Added
- #31: GitHub Issues and Confluence ticketing activity tracking
- New DB tables: ticketing_activity_cache, confluence_page_cache (v10 migration)
- New reports: github_issues_summary.json, confluence_activity_summary.json, ticketing_activity_summary.json

## [3.13.17] - 2026-02-27

### Added
- PR status tracking: `pr_state`, `closed_at`, and `is_merged` columns in `PullRequestCache` (v4 schema migration)
- Incremental stale-open-PR refresh: up to 50 stale open PRs updated per `gfa collect` run
- Rejection metrics in narrative reports and CSV output (merge rate, rejection rate, per-author breakdown)
- `gfa aliases` now supports AWS Bedrock as an LLM provider; auto-detected from config
- Configurable `strip_suffixes` list under `analysis.identity.strip_suffixes` for alias generation
- `gfa report` loads cached PR data from the database via new `get_cached_prs_for_report()` method
- DORA metrics, velocity, and narrative reports now incorporate real PR lifecycle data
- `gfa collect` enriches already-cached repos with PR data when `github_repo` is set — no re-collect needed

### Fixed
- `fetch_pr_reviews` config option now correctly wired through to the GitHub fetch layer (was a no-op)

### Changed
- Alias generation provider priority: Bedrock > OpenRouter > heuristic-only
- `github.organization` field is now the recommended way to scope PR collection to an org

## [3.13.4] - 2025-12-08

### Added
- Comprehensive documentation organization and standards
- Documentation standards based on Edgar project best practices
- Interactive launcher examples with complete workflows
- Story points configuration guide for JIRA integration
- Refactoring guide moved to developer documentation
- Project organization standards documentation

### Fixed
- All internal documentation links validated and corrected
- Documentation structure reorganized according to new standards
- Broken links in main README and examples documentation
- Test file path issues in error handling tests

### Changed
- Moved and consolidated documentation files according to new standards
- Archived outdated documentation files with proper date suffixes
- Updated all README files to reflect new organization structure
- Backfilled changelog with all missing versions from 1.2.24 to 3.13.3

## [3.13.3] - 2025-12-08

### Added
- Numbered selection UX for renaming developer aliases
- Interactive CLI menu with alias-rename option
- Interactive menu system with canonical name fixes

### Fixed
- Enhanced developer alias management workflow
- Improved user experience for alias operations

## [3.13.2] - 2025-12-08

### Added
- Interactive menu system for developer alias management
- Alias-rename command functionality
- Canonical name fixes for developer identities

## [3.13.1] - 2025-12-08

### Added
- Interactive CLI menu with alias-rename option
- Enhanced developer alias management

## [3.13.0] - 2025-12-08

### Added
- Interactive menu system for developer management
- Alias-rename command for developer identities
- Canonical name fixes and improvements

## [3.12.6] - 2025-12-08

### Added
- Claude MPM configuration file for enhanced AI integration

## [3.12.5] - 2025-12-08

### Fixed
- Black formatting in commit_utils.py

## [3.12.4] - 2025-12-08

### Fixed
- Black formatting in utils __init__.py

## [3.12.3] - 2025-12-08

### Fixed
- Default branch handling in test fixtures

## [3.12.2] - 2025-12-08

### Fixed
- Default branch handling in second merge operation

## [3.12.1] - 2025-12-08

### Fixed
- Default branch name handling in integration test fixture

## [3.12.0] - 2025-12-08

### Added
- Comprehensive improvements to merge commit exclusion feature

### Fixed
- Merge commit exclusion in GitDataFetcher for two-step architecture

### Changed
- Removed Python cache files from version control

## [3.11.1] - 2025-12-08

### Fixed
- Reverted direct spaCy model dependency due to PyPI restrictions

## [3.11.0] - 2025-12-08

### Added
- Automatic spaCy model installation

## [3.10.7] - 2025-12-08

### Added
- PROJECT_ORGANIZATION.md standard documentation
- Updated CLAUDE.md configuration

### Changed
- Archived temporary documentation files

## [3.10.6] - 2025-12-08

### Fixed
- Simplified return condition in is_qualitative_enabled

## [3.10.5] - 2025-12-08

### Fixed
- Support for nested qualitative config under analysis section

## [3.10.4] - 2025-12-08

### Fixed
- Black formatting in install_wizard.py

## [3.10.3] - 2025-12-08

### Fixed
- Moved git imports to module level
- Alphabetized git imports

## [3.10.2] - 2025-12-08

### Fixed
- Moved re and shutil imports to top level

## [3.10.1] - 2025-12-08

### Fixed
- Linting errors in install_wizard.py

## [3.10.0] - 2025-12-08

### Added
- Git URL cloning support to manual repository mode

## [3.9.3] - 2025-12-08

### Fixed
- Activity score normalization for reports without PR data

## [3.9.2] - 2025-12-08

### Fixed
- Black formatting in install_wizard.py

## [3.9.1] - 2025-12-08

### Fixed
- Removed unused pm_config variable in install wizard

## [3.9.0] - 2025-12-08

### Added
- Multi-platform PM ticketing support to installation wizard

## [3.8.1] - 2025-12-08

### Changed
- Ignore qualitative_cache and uv.lock files

## [3.8.0] - 2025-12-08

### Added
- 'gfa' as a shorthand command alias

## [3.7.5] - 2025-12-08

### Fixed
- Progress callback support to organization repository discovery

## [3.7.4] - 2025-12-08

### Fixed
- Black code formatting

## [3.7.3] - 2025-12-08

### Fixed
- Ruff F821 linter errors from lazy imports

## [3.7.2] - 2025-12-08

### Fixed
- Clone progress, retry logic, and PM platform filtering

## [3.7.1] - 2025-12-08

### Performance
- Optimized CLI startup time with lazy imports

## [3.7.0] - 2025-12-08

### Added
- Repository cloning to emergency fetch
- Automatic schema migration for timezone fix

### Fixed
- Automatically trim whitespace from interactive setup inputs

### Changed
- Removed TUI code

## [3.6.2] - 2025-12-08

### Fixed
- Uninitialized variable error when all repos use cached data

## [3.6.1] - 2025-12-08

### Fixed
- Critical timezone mismatch causing zero commits in database queries

### Changed
- Added Claude MPM cache directories to .gitignore

## [3.6.0] - 2025-12-08

### Added
- Guide users through config creation when file not found

## [3.5.2] - 2025-12-08

### Fixed
- Applied black formatting to new code

## [3.5.1] - 2025-12-08

### Fixed
- Linting errors in aliases system implementation

## [3.5.0] - 2025-12-08

### Added
- Developer aliases system with LLM generation
- Installation profiles for enhanced setup

## [3.4.7] - 2025-12-08

### Fixed
- All remaining ruff linting errors across project

## [3.4.6] - 2025-12-08

### Fixed
- Ruff linting errors in verify_activity

## [3.4.5] - 2025-12-08

### Fixed
- Removed failing test files from repository

## [3.4.4] - 2025-12-08

### Added
- Interactive launcher and enhanced identity detection

## [3.4.3] - 2025-12-08

### Added
- Comprehensive refactoring guide and tracking

## [3.4.2] - 2025-12-08

### Changed
- Extracted magic numbers to centralized constants module

## [3.4.1] - 2025-12-08

### Fixed
- Bare exception handlers and added type hints

## [3.4.0] - 2025-12-08

### Added
- Pre-flight git authentication and enhanced error reporting

### Fixed
- Remote branch analysis by preserving full branch references
- UnboundLocalError from redundant import in CLI

### Changed
- Applied Black formatting and auto-fix Ruff linting issues

## [3.3.0] - 2025-12-08

### Added
- Security analysis module and project cleanup

### Fixed
- F-string syntax error in git_timeout_wrapper.py

## [3.2.1] - 2025-12-08

### Fixed
- Thread safety in GitDataFetcher with thread-local storage

## [3.2.0] - 2025-12-08

### Added
- Progress tracking functionality

### Fixed
- Unhashable dict error
- Respect ticket_platforms configuration for ticket detection

## [3.1.12] - 2025-12-08

### Fixed
- Changed default display to simple output to prevent TUI hanging

## [3.1.11] - 2025-12-08

### Fixed
- TUI slow shutdown by properly managing thread executors
- TUI hanging during parallel repository analysis
- Missing RadioButton import in results screen
- TUI status reporting to distinguish 'no commits' from 'failed'
- TUI showing all repositories as failed when they have commits

## [3.1.10] - 2025-12-08

### Fixed
- TUI hanging during parallel repository analysis

## [3.1.9] - 2025-12-08

### Fixed
- TUI widget mounting errors in results_screen

## [3.1.8] - 2025-12-08

### Fixed
- Limited TUI parallel processing to single worker to avoid GitPython thread safety issues

## [3.1.7] - 2025-12-08

### Fixed
- TUIProgressAdapter signature mismatch causing all repositories to fail

## [3.1.6] - 2025-12-08

### Fixed
- 'core_progress' not accessible error in TUI

## [3.1.5] - 2025-12-08

### Fixed
- Properly set up TUI progress service for parallel repository processing

## [3.1.4] - 2025-12-08

### Fixed
- Set up progress service for TUI parallel repository processing

## [3.1.3] - 2025-12-08

### Fixed
- Initialize dark mode attribute in TUI app

## [3.1.2] - 2025-12-08

### Fixed
- Update JIRA API endpoints to use new /search/jql path

## [3.1.1] - 2025-12-08

### Fixed
- TUI stuck at 50% due to repository access issues

## [3.1.0] - 2025-12-08

### Added
- Comprehensive testing framework with TUI integration

### Fixed
- TUI progress tracking bugs and syntax errors
- Rich Pretty with Textual Static widget replacement
- TUI configuration loading and Pretty widget issues
- Added common CLI options to TUI command

## [3.0.0] - 2025-12-08

### Added
- TUI as the default interface with CLI fallback

### Breaking Changes
- TUI is now the default interface (major version bump)

## [2.0.0] - 2025-12-08

### Added
- Full-screen terminal interface restoration

### Breaking Changes
- Restored TUI command with full-screen terminal interface (major version bump)

## [1.6.6] - 2025-12-08

### Fixed
- Enabled Rich terminal UI by default

## [1.6.5] - 2025-12-08

### Fixed
- Hide PM framework and JIRA adapter debug messages

## [1.6.4] - 2025-12-08

### Fixed
- Clean up debug output and fix full-screen UI transition

## [1.6.3] - 2025-12-08

### Fixed
- Restart full-screen UI for Step 2 batch classification

## [1.6.2] - 2025-12-08

### Fixed
- Enable full-screen terminal UI in batch processing mode

## [1.6.1] - 2025-12-08

### Fixed
- Repository table comparison bug in full-screen UI

## [1.6.0] - 2025-12-08

### Added
- Live repository status tracking during analysis

## [1.5.0] - 2025-12-08

### Added
- Enhanced repository progress display during analysis

## [1.4.3] - 2025-12-08

### Fixed
- Made psutil an optional dependency for progress display

## [1.4.2] - 2025-12-08

### Fixed
- Environment variables resolution in PM integration config

## [1.4.1] - 2025-12-08

### Fixed
- Filtered stats storage for accurate line count exclusions

## [1.4.0] - 2025-12-08

### Added
- Sophisticated Rich-based progress display for better UX

## [1.3.12] - 2025-12-08

### Fixed
- Filtered stats storage for accurate line count exclusions

## [1.3.11] - 2025-12-08

### Fixed
- Applied black formatting to schema.py

## [1.3.10] - 2025-12-08

### Fixed
- Applied black formatting

## [1.3.9] - 2025-12-08

### Fixed
- Removed unused imports from data_fetcher

## [1.3.8] - 2025-12-08

### Fixed
- Branch analysis and added granular progress tracking

## [1.3.7] - 2025-12-08

### Fixed
- Temporarily disabled mypy in CI to unblock PyPI releases

## [1.3.6] - 2025-12-08

### Fixed
- Relaxed mypy configuration to allow PyPI release

## [1.3.5] - 2025-12-08

### Fixed
- Applied Black formatting for consistent code style

## [1.3.4] - 2025-12-08

### Fixed
- All remaining linting issues for clean CI/CD

## [1.3.3] - 2025-12-08

### Fixed
- Critical linting errors blocking PyPI release

## [1.3.2] - 2025-12-08

### Fixed
- All remaining test failures for PyPI publishing

## [1.3.1] - 2025-12-08

### Fixed
- Updated tests to match new comprehensive help system

## [1.3.0] - 2025-12-08

### Added
- Comprehensive help system with enhanced CLI documentation

## [1.2.24] - 2025-01-26

### Fixed
- Consolidated all multi-repository analysis fixes for EWTN organization
- Verified accuracy with sniff test on Aug 18-24 data (4 commits confirmed)
- All previous fixes working correctly:
  - Repository processing progress indicator
  - Authentication prompt elimination
  - Qualitative analysis error handling
  - Timestamp type handling

### Verified
- Multi-repository analysis accuracy confirmed
- Proper commit attribution across 95 repositories
- Correct date range filtering
- Accurate ticket coverage calculation (75%)

## [1.2.23] - 2025-01-25

### Fixed
- Fixed qualitative analysis 'int' object is not subscriptable error
  - Corrected timestamp default value in NLP engine from time.time() to datetime.now()
  - Added proper datetime import to nlp_engine module
  - This resolves type mismatch when timestamp field is missing

## [1.2.22] - 2025-01-25

### Fixed
- Fixed qualitative analysis commit format handling
  - Now handles both dict and object formats for commits
  - Fixed 'dict' object has no attribute 'hash' error
- Fixed SQLAlchemy warning about text expressions
  - Added proper text() wrapper for SELECT 1 statement

## [1.2.21] - 2025-01-25

### Fixed
- Fixed password prompts in data_fetcher during fetch/pull operations
  - Replaced GitPython's fetch() and pull() with subprocess calls
  - Added same environment variables to prevent credential prompts
  - Added 30-second timeout for both fetch and pull operations
  - This fixes the issue in the two-step fetch/classify process

## [1.2.20] - 2025-01-25

### Fixed
- Replaced GitPython clone with subprocess for better control
  - Uses subprocess.run with explicit timeout (30 seconds)
  - Disables credential helper to prevent prompts
  - Sets GIT_TERMINAL_PROMPT=0 and GIT_ASKPASS= to force failure
  - Shows which repository is being cloned for debugging
  - Properly handles timeout with clear error message

## [1.2.19] - 2025-01-25

### Fixed
- Improved clone operation with timeout and better credential handling
  - Added HTTP timeout (30 seconds) to prevent hanging on network issues
  - Fixed environment variable passing to GitPython
  - Added progress counter (x/95) to description for better visibility
  - Enhanced credential failure detection

## [1.2.18] - 2025-01-25

### Fixed
- Enhanced GitHub authentication handling to prevent interactive password prompts
  - Added GIT_TERMINAL_PROMPT=0 to disable git credential prompts
  - Added GIT_ASKPASS=/bin/echo to prevent password dialogs
  - Better detection of authentication failures (401, 403, permission denied)
  - Clear error messages when authentication fails instead of hanging

## [1.2.17] - 2025-01-25

### Fixed
- **CRITICAL**: Fixed indentation bug in CLI that prevented multi-repository analysis
  - Repository analysis code was incorrectly placed outside the for loop
  - This caused only the last repository to be analyzed instead of all repositories
  - Progress indicator now correctly updates for each repository (fixes "0/95" issue)
- Added authentication error handling for GitHub operations
  - Prevents password prompts when GitHub token is invalid or expired
  - Continues with local repository state if authentication fails
  - Provides clear error messages for authentication issues

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
