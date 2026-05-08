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
- `--backfill-since YYYY-MM-DD` - Hydrate `pull_request_cache` from this date forward. Bypasses the incremental fetch gate so historical PRs older than the last-processed checkpoint are fetched. Auto-triggers `weekly_pr_metrics` rollup for the same date range. Idempotent — safe to re-run. Does not change default behavior (#52).

**Examples**:
```bash
# Analyze specific repositories only
gitflow-analytics analyze -c config.yaml --repositories "repo1,repo2"

# Quick 2-week analysis with JSON output
gitflow-analytics analyze -c config.yaml --weeks 2 --format json

# Backfill all merged PRs back to a specific date
gfa analyze -c config.yaml --backfill-since 2025-01-01
```

### fetch
Fetch data from external platforms (GitHub PRs, JIRA, ClickUp) and cache it locally.

```bash
gfa fetch -c config.yaml [OPTIONS]
```

**Options**:
- `-c, --config PATH` - Path to YAML configuration file (required)
- `--weeks, -w INTEGER` - Number of weeks to fetch (default: 4)
- `--output, -o PATH` - Output directory for cache (overrides config)
- `--clear-cache` - Clear cache before fetching data
- `--backfill-since YYYY-MM-DD` - Hydrate `pull_request_cache` from this date forward. Bypasses the incremental fetch gate so historical PRs older than the last-processed checkpoint are fetched. Idempotent — safe to re-run. Does not change default behavior (#52).
- `--backfill-prs-since YYYY-MM-DD` - Override the PR-fetch lower bound independently of `--backfill-since`. When set, PRs are fetched back to this date; commits still use `--backfill-since`. Takes priority over `--backfill-since` for the PR window. Idempotent. (#55)
- `--log [none|INFO|DEBUG]` - Enable logging at the specified level (default: none)

**Examples**:
```bash
# Standard incremental fetch for the last 4 weeks
gfa fetch -c config.yaml

# Fetch the last 8 weeks
gfa fetch -c config.yaml --weeks 8

# Backfill all merged PRs back to a specific date
gfa fetch -c config.yaml --backfill-since 2025-01-01

# Re-run the same backfill safely (idempotent)
gfa fetch -c config.yaml --backfill-since 2025-01-01

# Backfill PRs further back than commits
gfa fetch -c config.yaml --backfill-since 2025-06-01 --backfill-prs-since 2025-01-01
```

### classify
Run batch LLM classification on commits already in the cache (Stage 2 of the
collect → classify → report pipeline).

```bash
gfa classify -c config.yaml [OPTIONS]
```

**Options**:
- `-c, --config PATH` - Path to YAML configuration file (required)
- `-w, --weeks INTEGER` - Number of weeks to classify (default: 4; should match the `collect --weeks` value)
- `--week YYYY-Www` - Target a specific ISO week (e.g. `2026-W07`). Repeatable for multiple discrete weeks. Mutually exclusive with `--weeks N`, `--from`, and `--to`.
- `--from YYYY-Www` - Start of an inclusive ISO week range (e.g. `--from 2026-W01`). Must be paired with `--to`. Mutually exclusive with `--weeks N` and `--week`.
- `--to YYYY-Www` - End of an inclusive ISO week range (e.g. `--to 2026-W18`). Must be paired with `--from`. Mutually exclusive with `--weeks N` and `--week`.
- `--reclassify` - Re-classify commits that were already classified
- `--show-jira-signals` - Log every commit short-circuited by the JIRA project-key
  mapping (issue #62). Useful for auditing which commits hit the tier-3
  classification path defined in `jira_project_mappings`.
- `--validate-coverage` - Warn when the fraction of classified commits falls below the
  coverage threshold. Exits with a non-zero code if coverage is insufficient (#65).
- `--coverage-threshold FLOAT` - Minimum acceptable classification coverage ratio
  (default: 0.8). Values between 0.0 and 1.0. Used in conjunction with
  `--validate-coverage`.
- `--log [none|INFO|DEBUG]` - Enable logging at the specified level (default: none)

**Examples**:
```bash
# Standard classification for the last 4 weeks
gfa classify -c config.yaml

# Force re-classification of all commits in range
gfa classify -c config.yaml --reclassify

# Classify a single ISO week
gfa classify -c config.yaml --week 2026-W07 --reclassify

# Classify an inclusive ISO week range
gfa classify -c config.yaml --from 2026-W01 --to 2026-W18 --reclassify

# Classify multiple discrete weeks (--week is repeatable)
gfa classify -c config.yaml --week 2026-W07 --week 2026-W08 --reclassify

# Audit which commits were classified via the JIRA project-key mapping
gfa classify -c config.yaml --show-jira-signals --log INFO

# Warn if fewer than 80% of commits are classified (default threshold)
gfa classify -c config.yaml --validate-coverage

# Warn if fewer than 90% of commits are classified
gfa classify -c config.yaml --validate-coverage --coverage-threshold 0.9
```

**Prerequisite**: `gfa collect -c config.yaml`

**Next step**: `gfa report -c config.yaml`

### collect
Collect commits and pull request data from GitHub into the local cache database
(Stage 1 of the collect → classify → report pipeline).

```bash
gfa collect -c config.yaml [OPTIONS]
```

**Options**:
- `-c, --config PATH` - Path to YAML configuration file (required)
- `-w, --weeks INTEGER` - Number of weeks to collect (default: 4)
- `--week YYYY-Www` - Target a specific ISO week (e.g. `2026-W07`). Repeatable for multiple discrete weeks. Mutually exclusive with `--weeks N`, `--from`, and `--to`.
- `--from YYYY-Www` - Start of an inclusive ISO week range (e.g. `--from 2026-W01`). Must be paired with `--to`. Mutually exclusive with `--weeks N` and `--week`.
- `--to YYYY-Www` - End of an inclusive ISO week range (e.g. `--to 2026-W04`). Must be paired with `--from`. Mutually exclusive with `--weeks N` and `--week`.
- `--log [none|INFO|DEBUG]` - Enable logging at the specified level (default: none)

**Examples**:
```bash
# Standard incremental collect for the last 4 weeks
gfa collect -c config.yaml

# Collect a single ISO week
gfa collect -c config.yaml --week 2026-W07

# Collect multiple discrete weeks
gfa collect -c config.yaml --week 2026-W07 --week 2026-W08
```

**Next step**: `gfa classify -c config.yaml`

### report
Generate reports from classified commit data already in the local cache
(Stage 3 of the collect → classify → report pipeline).

```bash
gfa report -c config.yaml [OPTIONS]
```

**Options**:
- `-c, --config PATH` - Path to YAML configuration file (required)
- `-w, --weeks INTEGER` - Number of weeks to include in the report (default: 4)
- `--week YYYY-Www` - Target a specific ISO week (e.g. `2026-W07`). Repeatable for multiple discrete weeks. Mutually exclusive with `--weeks N`, `--from`, and `--to`.
- `--from YYYY-Www` - Start of an inclusive ISO week range (e.g. `--from 2026-W01`). Must be paired with `--to`. Mutually exclusive with `--weeks N` and `--week`.
- `--to YYYY-Www` - End of an inclusive ISO week range (e.g. `--to 2026-W04`). Must be paired with `--from`. Mutually exclusive with `--weeks N` and `--week`.
- `--format [csv|json|markdown|all]` - Output format(s) to generate
- `--output-dir PATH` - Override output directory from config
- `--log [none|INFO|DEBUG]` - Enable logging at the specified level (default: none)

**Examples**:
```bash
# Generate a report for the last 4 weeks
gfa report -c config.yaml

# Report for a single ISO week
gfa report -c config.yaml --week 2026-W07

# Report for an inclusive ISO week range
gfa report -c config.yaml --from 2026-W01 --to 2026-W04
```

**Prerequisite**: `gfa classify -c config.yaml`

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

### add-alias
Add alias mappings to a configuration file non-interactively. Suitable for scripting and CI pipelines where interactive prompts are not available.

```bash
gfa add-alias -c config.yaml \
  --canonical "developer@example.com" \
  --alias "dev@personal.com" \
  --alias "Dev Name" \
  [OPTIONS]
```

**Required Options**:
- `-c, --config PATH` - Path to YAML configuration file

**Mapping Options** (mutually exclusive — use one or the other):
- `--canonical EMAIL` - Primary/canonical email for this developer identity; combine with one or more `--alias` flags
- `--from-file PATH` - YAML or JSON file containing batch alias mappings (cannot be combined with `--canonical`)
- `--alias EMAIL_OR_NAME` - Email address or display name to map to `--canonical`; repeatable

**Behaviour Flags**:
- `--dry-run` - Show what would be changed without writing to the config file
- `--apply` - Trigger identity re-resolution after updating the config

**Examples**:
```bash
# Map a personal email and display name to a canonical work email
gfa add-alias -c config.yaml \
  --canonical "alice@company.com" \
  --alias "alice@gmail.com" \
  --alias "Alice Smith"

# Preview changes before writing
gfa add-alias -c config.yaml \
  --canonical "alice@company.com" \
  --alias "alice@gmail.com" \
  --dry-run

# Load batch mappings from a YAML file and re-resolve identities
gfa add-alias -c config.yaml \
  --from-file aliases.yaml \
  --apply

# Load batch mappings from a JSON file
gfa add-alias -c config.yaml \
  --from-file aliases.json
```

**Supported `--from-file` Formats**:

1. **GFA native YAML** — a config file with a `developer_aliases:` key:
   ```yaml
   developer_aliases:
     - canonical: "alice@company.com"
       aliases: ["alice@gmail.com", "Alice Smith"]
   ```

2. **Flat YAML list** — a list of `{canonical, aliases}` objects:
   ```yaml
   - canonical: "alice@company.com"
     aliases:
       - alice@gmail.com
       - Alice Smith
   - canonical: "bob@company.com"
     aliases:
       - bob@personal.com
   ```

3. **JSON array** — equivalent structure in JSON:
   ```json
   [
     {"canonical": "alice@company.com", "aliases": ["alice@gmail.com", "Alice Smith"]},
     {"canonical": "bob@company.com",   "aliases": ["bob@personal.com"]}
   ]
   ```

**What It Does**:
1. Reads the existing `analysis.identity.manual_mappings` (or `developer_aliases`) section in the config
2. Merges new aliases into the matching canonical entry, or creates a new entry if the canonical is not yet present
3. Skips duplicates — existing aliases are never written twice (idempotent)
4. Writes the updated config back to disk (unless `--dry-run` is specified)
5. Optionally triggers identity re-resolution via `--apply`

**Use Cases**:
- Onboarding automation: script alias setup as part of repo initialisation
- CI pipelines: keep alias mappings in a separate file and apply them on deploy
- Bulk imports: migrate alias lists from another tool's export format
- Safe updates: use `--dry-run` to audit changes before committing them

**Notes**:
- `--from-file` and `--canonical` are mutually exclusive; combining them is an error
- The operation is idempotent: running the same command twice produces the same config
- Always verify with `--dry-run` before running in unattended automation
- See [Managing Aliases Guide](../guides/managing-aliases.md) for detailed identity management guidance

### backfill-ticket-ids
Populate NULL `ticket_ids` and `commit_count` values on existing cached pull requests using commit messages already stored in `cached_commits`. No GitHub API calls are made — all data is sourced from the local cache database. The operation is idempotent and safe to re-run.

```bash
gfa backfill-ticket-ids -c config.yaml
```

**Options**:
- `-c, --config PATH` - Path to YAML configuration file (required)

**Examples**:
```bash
# Backfill ticket IDs and commit counts for all cached PRs
gfa backfill-ticket-ids -c config.yaml
```

**What It Does**:
1. Queries `pull_request_cache` for rows where `ticket_ids` IS NULL or `commit_count` IS NULL
2. For each such PR, joins against `cached_commits` using commit hashes already in the local database
3. Extracts JIRA-style ticket IDs matching `[A-Z]+-\d+` from each commit message and deduplicates them
4. Counts commit hashes to produce `commit_count`
5. Writes `ticket_ids` (JSON array, e.g. `["DUE-1234", "CORE-567"]`) and `commit_count` back to the row
6. Makes no outbound GitHub API calls — entirely offline

**Use Cases**:
- Upgrading an existing installation to v3.14.22 and enriching previously cached PRs
- Recovering `ticket_ids` after a schema migration
- Re-running after updating the ticket-ID extraction regex without re-fetching PRs

**Notes**:
- PRs whose `ticket_ids` and `commit_count` are already populated are skipped
- Commit messages must already be present in `cached_commits`; PRs with no associated cached commits will have `ticket_ids` set to `[]` and `commit_count` set to `0`
- See [Cache System Reference](./cache-system.md) for the full `pull_request_cache` schema

### override
Manage manual classification overrides for individual commits. Overrides take
precedence over all automated classifiers (LLM, rule-based, JIRA mapping).

```bash
gfa override <subcommand> [OPTIONS]
```

**Subcommands**:

#### override set
Manually assign a `change_type` classification to a specific commit.

```bash
gfa override set <commit_sha> <change_type> -c config.yaml
```

**Arguments**:
- `commit_sha` - Full or abbreviated commit SHA to override
- `change_type` - Classification to apply (e.g. `feature`, `maintenance`, `bugfix`, `analytics`)

**Examples**:
```bash
# Override a commit to be classified as a feature
gfa override set abc1234 feature -c config.yaml

# Override a commit to be classified as maintenance
gfa override set def5678 maintenance -c config.yaml
```

#### override list
List all manual classification overrides currently stored in the cache database.

```bash
gfa override list -c config.yaml
```

**Options**:
- `-c, --config PATH` - Path to YAML configuration file (required)

**Examples**:
```bash
# Show all active overrides
gfa override list -c config.yaml
```

#### override remove
Remove a manual classification override, returning the commit to automated
classification on the next `gfa classify` run.

```bash
gfa override remove <commit_sha> -c config.yaml
```

**Arguments**:
- `commit_sha` - Full or abbreviated commit SHA whose override should be removed

**Examples**:
```bash
# Remove an override and allow automated re-classification
gfa override remove abc1234 -c config.yaml
```

**What It Does** (all override subcommands):
1. Reads/writes the `classification_overrides` table in the local cache database
2. On the next `gfa classify` run, overrides take the highest precedence —
   the LLM call for overridden commits is skipped entirely
3. `override remove` deletes the row; the commit reverts to standard
   classification on the next classify pass

### backfill-ai-detection
Backfill AI-detection results for commits already stored in `cached_commits`.
Scans commit footers and trailers for known AI-tool signatures (Cursor, Claude,
GitHub Copilot) and writes results to the cache database. No GitHub API calls
are made — all data is sourced from cached commit messages.

```bash
gfa backfill-ai-detection -c config.yaml
```

**Options**:
- `-c, --config PATH` - Path to YAML configuration file (required)

**Examples**:
```bash
# Detect AI footers for all cached commits
gfa backfill-ai-detection -c config.yaml
```

**What It Does**:
1. Queries `cached_commits` for all rows
2. Scans each commit message/body for AI-tool trailer patterns:
   - `Co-authored-by: Claude` (Anthropic Claude)
   - `Co-authored-by: GitHub Copilot` (GitHub Copilot)
   - Cursor-specific commit footers
3. Writes `is_ai_assisted` flag and detected tool name back to `cached_commits`
4. Operation is idempotent — safe to re-run after adding new AI-tool signatures

**Use Cases**:
- Upgrading an existing installation and enriching previously cached commits
- Adding AI-tool detection to a repository after the fact
- Re-running after adding new AI-tool signature patterns

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