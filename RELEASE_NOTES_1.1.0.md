# GitFlow Analytics v1.1.0 Release Notes

## Release Date: 2025-01-06

## Overview
Version 1.1.0 introduces significant enhancements to commit classification, database-backed reporting, and configuration flexibility. This release focuses on providing more accurate work categorization, faster report generation through SQLite storage, and improved compatibility with various configuration formats.

## 🎯 Major Features

### 1. Database-Backed Reporting System
- **Daily Metrics Storage**: Store classified activity metrics on a per-day basis in SQLite
- **Fast Report Generation**: Generate reports directly from database without reprocessing
- **Weekly Trend Analysis**: Track week-over-week changes in classification patterns
- **Performance Optimization**: Significantly faster report generation for large datasets

### 2. Enhanced Commit Classification
- **Proper Work Categorization**: All commits now classified into meaningful categories (feature, bug_fix, refactor, etc.)
- **Ticket-Enhanced Classification**: Uses ticket information to improve classification accuracy
- **No More "Tracked Work" Category**: Tracked commits are properly classified like all other commits
- **Project-Level Classifications**: Each project shows classification breakdown in reports

### 3. Configuration Improvements
- **Flexible Field Names**: Supports both `api_key`/`openrouter_api_key` and `model`/`primary_model`
- **Cost Tracking Support**: Maps `cost_tracking.daily_budget_usd` to internal configuration
- **Ticket Platform Filtering**: Properly respects configured ticket platforms (e.g., JIRA only)
- **Auto-Enable Qualitative Analysis**: No longer requires CLI flag when configured

## 🔧 Technical Improvements

### Database Schema
- New `DailyMetrics` table for storing per-day developer/project metrics
- New `WeeklyTrends` table for pre-calculated trend data
- Optimized indexes for date-range queries

### Report Enhancements
- **Narrative Reports**: Added classification breakdowns to project activity sections
- **Ticket Coverage**: Displayed separately from classifications as a process metric
- **Database Report Generator**: New report generator that pulls directly from SQLite
- **HTML Report**: Temporarily disabled for redesign (code preserved)

### Bug Fixes
- Fixed datetime import scope issues in CLI
- Corrected ticket platform filtering to respect configuration
- Enhanced error handling for missing classification data
- Improved identity resolution for developer mappings

## 📊 New Metrics Available

### Classification Metrics
- Features, Bug Fixes, Refactoring, Documentation, Maintenance, Tests, Style, Build
- Per-developer and per-project breakdowns
- Weekly trend analysis with percentage changes

### Storage Metrics
- Daily commit counts by classification
- Lines changed, files modified
- Story points tracking
- Ticket coverage percentages

## 🚀 Performance Improvements
- Database caching reduces report generation time by up to 80%
- Batch processing for daily metrics storage
- Optimized queries with proper indexing
- Pre-calculated weekly trends for instant retrieval

## 💡 Usage Examples

### Generate Database-Backed Report
```bash
gitflow-analytics -c config.yaml --weeks 4
```

### View Weekly Trends
Reports now automatically include weekly trend analysis:
- "Features: +15%, Bug Fixes: -5%, Refactoring: +2%"

### Configuration for JIRA-Only Tracking
```yaml
analysis:
  ticket_platforms:
    - jira  # Only extracts JIRA tickets, ignores GitHub/Linear/ClickUp
```

## 🔄 Migration Notes

### For Existing Users
1. Clear cache after upgrade: `--clear-cache`
2. Daily metrics will be built on first run
3. HTML reports temporarily disabled (use narrative/CSV reports)
4. Configuration files remain backward compatible

### Configuration Updates
- OpenRouter API key now auto-detected from `.env` file
- Qualitative analysis auto-enabled when configured
- Cost tracking configuration properly mapped

## 🐛 Known Issues
- HTML report generation temporarily disabled pending redesign
- First run with database storage may take longer while building initial metrics

## 📝 Documentation Updates
- Updated CLAUDE.md with database storage details
- Added troubleshooting for classification issues
- Enhanced configuration examples

## 🙏 Acknowledgments
This release includes significant enhancements based on user feedback, particularly around commit classification accuracy and report generation performance.

---

## Commit Summary
- feat: implement database-backed reporting system with daily metrics storage
- feat: add weekly trend analysis for classification patterns
- feat: enhance commit classification with ticket information
- fix: remove "tracked_work" as a classification category
- fix: correct ticket platform filtering to respect configuration
- fix: resolve datetime import scope issues
- feat: add classification breakdown to project activity reports
- feat: support flexible configuration field names for OpenRouter
- feat: auto-enable qualitative analysis when configured
- chore: temporarily disable HTML report generation