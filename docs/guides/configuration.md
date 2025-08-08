# GitFlow Analytics Configuration Guide

## Quick Start

1. **Copy the example files:**
   ```bash
   cp config-sample.yaml my-config.yaml
   cp .env.example .env
   ```

2. **Edit `.env` with your credentials:**
   ```bash
   GITHUB_TOKEN=your_github_personal_access_token
   GITHUB_OWNER=your_github_username_or_org
   ```

3. **Clone repositories (optional):**
   ```bash
   ./setup-repos.sh
   ```

4. **Run the analysis:**
   ```bash
   gitflow-analytics -c my-config.yaml
   ```

## Configuration Structure

### GitHub Authentication

The `github` section supports both direct tokens and environment variables, plus organization-based repository discovery:

```yaml
github:
  token: "${GITHUB_TOKEN}"     # From environment variable
  owner: "${GITHUB_OWNER}"     # Default owner for repository-based config
  organization: "myorg"        # For organization-based discovery
  # token: "ghp_direct_token_here"  # Or direct token (not recommended)
```

#### Organization-based Configuration

When `organization` is specified, GitFlow Analytics automatically discovers all non-archived repositories:

```yaml
version: "1.0"

github:
  token: "${GITHUB_TOKEN}"
  organization: "myorg"  # Automatically discovers repositories

# No repositories section needed - they're discovered automatically!
analysis:
  story_point_patterns:
    - "(?:story\\s*points?|sp|pts?)\\s*[:=]\\s*(\\d+)"
```

**Benefits:**
- Automatically includes new repositories as they're added to the organization
- No need to manually update configuration for each new repository
- Perfect for organizations with many repositories

**Requirements:**
- GitHub token must have organization read access
- Token must have repository read access for all organization repositories

#### Repository-based Configuration

For manual control over which repositories to analyze:

```yaml
repositories:
  - name: "frontend"
    github_repo: "frontend"  # Will use GITHUB_OWNER/frontend
  
  - name: "external-repo"
    github_repo: "other-org/their-repo"  # Explicit owner overrides default
```

### Repository Configuration

Each repository can be configured with:

- `name`: Display name for reports
- `path`: Local filesystem path (supports `~` for home directory)
- `github_repo`: GitHub repository for PR/issue enrichment
- `project_key`: Override for grouping in reports (defaults to uppercase name)
- `branch`: Specific branch to analyze (defaults to all branches)

### Story Point Patterns

Customize regex patterns to match your team's story point format:

```yaml
analysis:
  story_point_patterns:
    - "(?:story\\s*points?|sp|pts?)\\s*[:=]\\s*(\\d+)"  # SP: 5, Story Points = 3
    - "\\[(\\d+)\\s*(?:sp|pts?)\\]"                     # [3sp], [5 pts]
    - "#(\\d+)sp"                                       # #3sp
    - "estimate:\\s*(\\d+)"                             # estimate: 5
    - "\\bSP(\\d+)\\b"                                  # SP5, SP13
```

### Filtering Commits and Files

#### Exclude Commits

Exclude bot commits and merge commits:

```yaml
analysis:
  exclude:
    authors:
      - "dependabot[bot]"
      - "renovate[bot]"
      - "github-actions[bot]"
    message_patterns:
      - "^Merge branch"
      - "^\\[skip ci\\]"
```

#### Path Exclusions (New Feature)

Filter out boilerplate and generated files from line count metrics:

```yaml
analysis:
  exclude:
    # Glob patterns for files to exclude from line counting
    paths:
      - "**/node_modules/**"    # Node.js dependencies
      - "**/vendor/**"          # PHP/Ruby dependencies
      - "**/*.min.js"           # Minified JavaScript
      - "**/package-lock.json"  # Lock files
      - "**/generated/**"       # Generated code directories
```

**Default Exclusions:**
If you don't specify custom `paths`, the following patterns are excluded by default:
- Package manager lock files: `package-lock.json`, `yarn.lock`, `poetry.lock`, etc.
- Build/distribution directories: `dist/`, `build/`, `.next/`
- Dependencies: `node_modules/`, `vendor/`
- Minified files: `*.min.js`, `*.min.css`
- Source maps: `*.map`
- Generated files: `*.generated.*`, `generated/`
- Coverage reports: `coverage/`, `htmlcov/`
- Python cache: `__pycache__/`

**Note:** The filtered line counts are used in all reports alongside the raw Git statistics for backward compatibility.

### Output Configuration

Control where reports are saved and which formats are generated:

```yaml
output:
  # Output directory for reports (supports ~ for home directory)
  directory: "~/Clients/project-name/reports"
  
  formats:
    - csv        # Weekly metrics CSV
    - markdown   # Narrative report
    # - json     # Structured data (uncomment to enable)
    # - html     # Web report (uncomment to enable)
```

#### Directory Defaults (New Behavior)

GitFlow Analytics now defaults directories to be relative to the configuration file location:

**Output Directory:**
- If `output.directory` is not specified: defaults to the config file's directory
- If `output.directory` is a relative path: resolved relative to config file directory
- If `output.directory` is an absolute path: used as-is
- CLI `--output` flag overrides configuration file setting

**Example:**
```
/project/
├── config.yaml           # Configuration file location
├── weekly_metrics.csv     # Reports generated here (default)
├── summary.csv
└── data/
    └── custom-reports/    # If output.directory: "data/custom-reports"
```

The output directory can be:
- Specified in the config file (recommended for project-specific locations)
- Overridden via CLI with `--output` flag
- Defaults to config file directory if not specified (changed from `./reports`)

### Anonymization

Enable anonymization for sharing reports externally:

```yaml
output:
  anonymization:
    enabled: true
    fields: [email, name]
    method: "hash"  # Consistent hashing
    # method: "sequential"  # Dev1, Dev2, etc.
```

### Caching

Configure cache behavior for performance:

```yaml
cache:
  directory: ".gitflow-cache"  # Cache location (relative to config file)
  ttl_hours: 168              # Cache validity (1 week)
  max_size_mb: 500            # Maximum cache size
```

**Cache Directory Behavior:**
- If `cache.directory` is not specified: defaults to `.gitflow-cache/` in config file directory
- If `cache.directory` is a relative path: resolved relative to config file directory
- If `cache.directory` is an absolute path: used as-is
- Contains SQLite databases for commit analysis and identity resolution

## Environment Variables

The following environment variables are supported:

- `GITHUB_TOKEN`: GitHub personal access token
- `GITHUB_OWNER`: Default GitHub owner/organization
- `GITFLOW_CACHE_DIR`: Override cache directory
- `GITFLOW_OUTPUT_DIR`: Override output directory

## Multiple Configurations

You can maintain multiple configurations for different teams or projects:

```bash
# Development team analysis
gitflow-analytics -c configs/dev-team.yaml

# QA team analysis  
gitflow-analytics -c configs/qa-team.yaml

# Executive summary (all teams)
gitflow-analytics -c configs/all-teams.yaml
```

## Validation

Always validate your configuration before running a full analysis:

```bash
gitflow-analytics -c my-config.yaml --validate-only
```

This will check:
- Repository paths exist and are Git repositories
- GitHub token is available if GitHub repos are specified
- Cache directory is writable
- Configuration syntax is valid