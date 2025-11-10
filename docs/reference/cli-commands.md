# CLI Commands Reference

Complete command-line interface reference for GitFlow Analytics.

## 🚀 Basic Usage

### Default Command (Analyze)
```bash
# Simplified syntax (analyze is the default command)
gitflow-analytics [OPTIONS]

# Explicit analyze command (backward compatible)  
gitflow-analytics analyze [OPTIONS]
```

**Examples**:
```bash
# Basic analysis with configuration file
gitflow-analytics -c config.yaml

# Analyze last 8 weeks
gitflow-analytics -c config.yaml --weeks 8

# Clear cache and re-analyze
gitflow-analytics -c config.yaml --clear-cache
```

## 📋 Global Options

### Required Options
- `-c, --config PATH` - Path to YAML configuration file

### Analysis Options  
- `--weeks INTEGER` - Number of weeks to analyze (default: from config)
- `--clear-cache` - Clear analysis cache before running
- `--skip-identity-analysis` - Skip automatic identity resolution
- `--validate-only` - Validate configuration without running analysis

### Output Options
- `--format [csv,json,markdown,all]` - Output format(s) to generate
- `--output-dir PATH` - Override output directory from config
- `--quiet` - Suppress progress output
- `--verbose` - Enable verbose logging

### Utility Options
- `--version` - Show version information
- `--help` - Show help message and exit

## 🔧 Subcommands

### analyze (default)
Run comprehensive repository analysis and generate reports.

```bash
gitflow-analytics analyze -c config.yaml [OPTIONS]
```

**Options**:
- All global options apply
- `--repositories TEXT` - Comma-separated list of repositories to analyze (overrides config)

**Examples**:
```bash
# Analyze specific repositories only
gitflow-analytics analyze -c config.yaml --repositories "repo1,repo2"

# Quick 2-week analysis with JSON output
gitflow-analytics analyze -c config.yaml --weeks 2 --format json
```

### identities
Manage developer identity resolution and consolidation.

```bash
gitflow-analytics identities -c config.yaml [OPTIONS]
```

**Options**:
- `--interactive` - Interactive identity resolution mode
- `--auto-approve` - Automatically approve suggested identity mappings
- `--export PATH` - Export identity mappings to YAML file
- `--import PATH` - Import identity mappings from YAML file

**Examples**:
```bash
# Run interactive identity analysis
gitflow-analytics identities -c config.yaml --interactive

# Export current identity mappings
gitflow-analytics identities -c config.yaml --export identity-mappings.yaml
```

### validate
Validate configuration files and system setup.

```bash
gitflow-analytics validate -c config.yaml [OPTIONS]
```

**Options**:
- `--check-tokens` - Validate GitHub API tokens and permissions
- `--check-repos` - Verify repository access and cloning
- `--check-ml` - Validate ML model availability and setup

**Examples**:
```bash
# Comprehensive validation
gitflow-analytics validate -c config.yaml --check-tokens --check-repos --check-ml

# Quick config validation only
gitflow-analytics validate -c config.yaml
```

### cache
Manage analysis cache and performance optimization.

```bash
gitflow-analytics cache [SUBCOMMAND] [OPTIONS]
```

**Subcommands**:
- `clear` - Clear all cache databases
- `status` - Show cache statistics and disk usage  
- `optimize` - Optimize cache databases (VACUUM)

**Examples**:
```bash
# Clear all caches
gitflow-analytics cache clear

# Show cache status
gitflow-analytics cache status

# Optimize cache performance
gitflow-analytics cache optimize
```

### alias-rename
Rename a developer's canonical display name in manual mappings.

```bash
gitflow-analytics alias-rename -c config.yaml \
  --old-name "Current Name" \
  --new-name "New Name" \
  [OPTIONS]
```

**Required Options**:
- `--old-name TEXT` - Current canonical name to rename (must exist in manual_mappings)
- `--new-name TEXT` - New canonical display name to use in reports

**Optional Flags**:
- `--update-cache` - Update cached database records with the new name
- `--dry-run` - Show what would be changed without applying changes

**Examples**:
```bash
# Preview changes with dry-run
gitflow-analytics alias-rename -c config.yaml \
  --old-name "bianco-zaelot" \
  --new-name "Emiliozzo Bianco" \
  --dry-run

# Apply rename to config file only
gitflow-analytics alias-rename -c config.yaml \
  --old-name "bianco-zaelot" \
  --new-name "Emiliozzo Bianco"

# Update both config and database cache
gitflow-analytics alias-rename -c config.yaml \
  --old-name "bianco-zaelot" \
  --new-name "Emiliozzo Bianco" \
  --update-cache
```

**What It Does**:
1. Searches `analysis.identity.manual_mappings` for the old name
2. Updates the `name` field to the new name
3. Preserves all other fields (primary_email, aliases)
4. Optionally updates `developer_identities` and `developer_aliases` tables

**Use Cases**:
- Fix typos in developer names
- Use preferred names or nicknames
- Update names after marriage or legal name changes
- Standardize name formatting across team

**Notes**:
- Without `--update-cache`, old name persists in cached data until next analysis
- Always test with `--dry-run` first to preview changes
- See [Managing Aliases Guide](../guides/managing-aliases.md#renaming-developers) for detailed usage

## 📊 Output Formats

### CSV Format (`--format csv`)
Generates structured data files:
- `weekly_metrics_YYYYMMDD.csv` - Weekly development metrics
- `developers_YYYYMMDD.csv` - Developer profiles and statistics
- `summary_YYYYMMDD.csv` - Project-wide summary statistics
- `untracked_commits_YYYYMMDD.csv` - Commits without ticket references

### JSON Format (`--format json`)
Generates comprehensive data export:
- `comprehensive_export_YYYYMMDD.json` - Complete analysis data

### Markdown Format (`--format markdown`)
Generates human-readable reports:
- `narrative_report_YYYYMMDD.md` - Executive summary with insights

### All Formats (`--format all`)
Generates all available output formats.

## 🚨 Exit Codes

GitFlow Analytics uses standard exit codes:

- **0**: Success - Analysis completed successfully
- **1**: General error - Configuration or processing error
- **2**: Configuration error - Invalid YAML or missing required fields
- **3**: Authentication error - Invalid or missing GitHub token
- **4**: Repository error - Repository access or cloning failed
- **5**: Analysis error - Analysis processing failed
- **6**: Output error - Report generation failed

## 🔍 Environment Variables

GitFlow Analytics recognizes these environment variables:

### Authentication
- `GITHUB_TOKEN` - GitHub personal access token
- `JIRA_ACCESS_USER` - JIRA username for API access
- `JIRA_ACCESS_TOKEN` - JIRA API token or password

### Configuration  
- `GITFLOW_CONFIG` - Default configuration file path
- `GITFLOW_CACHE_DIR` - Override default cache directory
- `GITFLOW_LOG_LEVEL` - Set logging level (DEBUG, INFO, WARNING, ERROR)

### Performance
- `GITFLOW_MAX_WORKERS` - Maximum parallel processing workers
- `GITFLOW_BATCH_SIZE` - Commit processing batch size
- `GITFLOW_TIMEOUT` - Network request timeout in seconds

## 💡 Usage Patterns

### Daily Team Health Check
```bash
# Quick 1-week analysis for daily standup insights
gitflow-analytics -c config.yaml --weeks 1 --format markdown --quiet
```

### Weekly Sprint Review
```bash
# 2-week analysis with comprehensive data
gitflow-analytics -c config.yaml --weeks 2 --format all
```

### Monthly Planning Analysis
```bash
# 4-week analysis with cache clearing for fresh data
gitflow-analytics -c config.yaml --weeks 4 --clear-cache --format all
```

### Quarterly Strategic Review  
```bash
# 12-week comprehensive analysis
gitflow-analytics -c config.yaml --weeks 12 --format all --verbose
```

### CI/CD Integration
```bash
# Automated analysis with JSON export for dashboard integration
gitflow-analytics -c config.yaml --weeks 4 --format json --quiet
```

## 🔧 Advanced Usage

### Configuration Override
```bash
# Override output directory
gitflow-analytics -c config.yaml --output-dir /custom/reports/

# Analyze subset of repositories
gitflow-analytics -c config.yaml --repositories "critical-repo,main-app"
```

### Performance Optimization
```bash
# Use cached analysis for faster reporting
gitflow-analytics -c config.yaml --weeks 8

# Clear cache for fresh analysis (slower but current)
gitflow-analytics -c config.yaml --weeks 8 --clear-cache
```

### Debugging and Troubleshooting
```bash
# Verbose output for debugging
gitflow-analytics -c config.yaml --verbose

# Validate configuration before running
gitflow-analytics validate -c config.yaml --check-tokens --check-repos

# Test configuration without full analysis
gitflow-analytics -c config.yaml --validate-only
```

## 🆘 Common Issues

### "Command not found"
```bash
# Ensure GitFlow Analytics is installed and in PATH
pip show gitflow-analytics
which gitflow-analytics

# Install if missing
pip install gitflow-analytics
```

### "Configuration file not found"
```bash
# Provide absolute path to configuration
gitflow-analytics -c /full/path/to/config.yaml

# Check current directory for config file
ls -la *.yaml
```

### "GitHub API rate limit exceeded"
```bash
# Check token is set correctly
echo $GITHUB_TOKEN

# Validate token has necessary permissions
gitflow-analytics validate -c config.yaml --check-tokens
```

### "Repository not found or access denied"
```bash
# Verify repository names and permissions
gitflow-analytics validate -c config.yaml --check-repos

# Check GitHub token has access to repositories
```

## 📚 Related Documentation

- **[Configuration Guide](../guides/configuration.md)** - Complete YAML configuration reference
- **[Getting Started](../getting-started/)** - Installation and first steps
- **[Troubleshooting](../guides/troubleshooting.md)** - Common issues and solutions
- **[Examples](../examples/)** - Real-world usage scenarios

## 🔄 Command History and Aliases

### Useful Shell Aliases
```bash
# Add to your .bashrc or .zshrc
alias gfa='gitflow-analytics'
alias gfa-weekly='gitflow-analytics -c config.yaml --weeks 1'
alias gfa-monthly='gitflow-analytics -c config.yaml --weeks 4 --clear-cache'
alias gfa-validate='gitflow-analytics validate -c config.yaml --check-all'
```

### Bash Completion
GitFlow Analytics supports bash completion for commands and options:
```bash
# Enable bash completion (if supported)
eval "$(_GITFLOW_ANALYTICS_COMPLETE=bash_source gitflow-analytics)"
```