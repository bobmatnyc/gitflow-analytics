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

The `github` section supports both direct tokens and environment variables:

```yaml
github:
  token: "${GITHUB_TOKEN}"  # From environment variable
  owner: "${GITHUB_OWNER}"  # Default owner for repositories
  # token: "ghp_direct_token_here"  # Or direct token (not recommended)
```

The `owner` field is used as a default when repository names don't include the owner:

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

### Filtering Commits

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

The output directory can be:
- Specified in the config file (recommended for project-specific locations)
- Overridden via CLI with `--output` flag
- Defaults to `./reports` if not specified

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
  directory: ".gitflow-cache"  # Cache location
  ttl_hours: 168              # Cache validity (1 week)
  max_size_mb: 500            # Maximum cache size
```

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