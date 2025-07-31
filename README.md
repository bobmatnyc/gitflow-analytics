# GitFlow Analytics

[![PyPI version](https://badge.fury.io/py/gitflow-analytics.svg)](https://badge.fury.io/py/gitflow-analytics)
[![Python Support](https://img.shields.io/pypi/pyversions/gitflow-analytics.svg)](https://pypi.org/project/gitflow-analytics/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for analyzing Git repositories to generate comprehensive developer productivity reports. It extracts data directly from Git history and GitHub APIs, providing weekly summaries, productivity insights, and gap analysis.

## Features

- 🚀 **Multi-repository analysis** with project grouping
- 🏢 **Organization-based repository discovery** from GitHub
- 👥 **Developer identity resolution** and normalization
- 📊 **Work volume analysis** (absolute vs relative effort)
- 🎯 **Story point extraction** from commit messages and PR descriptions
- 🎫 **Multi-platform ticket tracking** (JIRA, GitHub Issues, ClickUp, Linear)
- 📈 **Weekly CSV reports** with productivity metrics
- 🔒 **Data anonymization** for external sharing
- ⚡ **Smart caching** for fast repeated analyses
- 🔄 **Batch processing** for large repositories

## Quick Start

### Installation

**From PyPI (Recommended):**

```bash
pip install gitflow-analytics
```

**From Source (Development):**

```bash
git clone https://github.com/bobmatnyc/gitflow-analytics.git
cd gitflow-analytics
pip install -e ".[dev]"
```

### Basic Usage

1. Create a configuration file (`config.yaml`):

**Option A: Organization-based (Automatic Repository Discovery)**
```yaml
version: "1.0"

github:
  token: "${GITHUB_TOKEN}"
  organization: "myorg"  # Automatically discovers all repositories

analysis:
  story_point_patterns:
    - "(?:story\\s*points?|sp|pts?)\\s*[:=]\\s*(\\d+)"
    - "\\[(\\d+)\\s*(?:sp|pts?)\\]"
```

**Option B: Repository-based (Manual Configuration)**
```yaml
version: "1.0"

github:
  token: "${GITHUB_TOKEN}"
  owner: "${GITHUB_OWNER}"

repositories:
  - name: "frontend"
    path: "~/repos/frontend"
    github_repo: "myorg/frontend"
    project_key: "FRONTEND"
    
  - name: "backend"
    path: "~/repos/backend"
    github_repo: "myorg/backend"
    project_key: "BACKEND"

analysis:
  story_point_patterns:
    - "(?:story\\s*points?|sp|pts?)\\s*[:=]\\s*(\\d+)"
    - "\\[(\\d+)\\s*(?:sp|pts?)\\]"
```

2. Create a `.env` file in the same directory as your `config.yaml`:

```bash
# .env
GITHUB_TOKEN=ghp_your_github_token_here
GITHUB_OWNER=your_github_org  # Only for repository-based setup
```

3. Run the analysis:

```bash
gitflow-analytics analyze -c config.yaml
```

## Configuration Options

### Environment Variables and Credentials

GitFlow Analytics automatically loads environment variables from a `.env` file in the same directory as your configuration YAML. This is the recommended approach for managing credentials securely.

#### Step 1: Create a `.env` file

Create a `.env` file next to your configuration YAML:

```bash
# .env file (same directory as your config.yaml)
# GitHub credentials (required)
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
GITHUB_OWNER=myorg  # Optional: default owner for repositories

# JIRA credentials (optional - only if using JIRA integration)
JIRA_ACCESS_USER=your.email@company.com
JIRA_ACCESS_TOKEN=xxxxxxxxxxxxxxxxxxxx

# Other optional tokens
CLICKUP_TOKEN=pk_xxxxxxxxxxxx
LINEAR_TOKEN=lin_api_xxxxxxxxxxxx
```

#### Step 2: Reference in YAML configuration

Use `${VARIABLE_NAME}` syntax in your YAML to reference environment variables:

```yaml
# config.yaml
version: "1.0"

github:
  token: "${GITHUB_TOKEN}"        # Required
  owner: "${GITHUB_OWNER}"        # Optional
  organization: "${GITHUB_ORG}"   # Optional (for org-based discovery)

# Optional: JIRA integration
jira:
  access_user: "${JIRA_ACCESS_USER}"
  access_token: "${JIRA_ACCESS_TOKEN}"
  base_url: "https://yourcompany.atlassian.net"

# Optional: Configure which JIRA fields contain story points
jira_integration:
  story_point_fields:
    - "Story Points"
    - "customfield_10016"  # Your custom field ID
```

#### Important Notes:

- **Never commit `.env` files** to version control (add to `.gitignore`)
- If credentials are not found in the `.env` file, the tool will exit with an informative error
- The `.env` file must be in the same directory as your YAML configuration
- All configured services must have corresponding environment variables set

### Organization vs Repository-based Setup

GitFlow Analytics supports two main configuration approaches:

#### Organization-based Configuration (Recommended)

Automatically discovers all non-archived repositories from a GitHub organization:

```yaml
version: "1.0"

github:
  token: "${GITHUB_TOKEN}"
  organization: "myorg"  # Your GitHub organization name

# Optional: Customize analysis settings
analysis:
  story_point_patterns:
    - "(?:story\\s*points?|sp|pts?)\\s*[:=]\\s*(\\d+)"
  
  exclude:
    authors:
      - "dependabot[bot]"
      - "github-actions[bot]"
```

**Benefits:**
- Automatically discovers new repositories as they're added to the organization
- No need to manually configure each repository
- Simplified configuration management
- Perfect for teams with many repositories

**Requirements:**
- Your GitHub token must have organization read access
- Repositories will be automatically cloned to local directories if they don't exist

#### Repository-based Configuration

Manually specify each repository to analyze:

```yaml
version: "1.0"

github:
  token: "${GITHUB_TOKEN}"
  owner: "${GITHUB_OWNER}"  # Default owner for repositories

repositories:
  - name: "frontend"
    path: "~/repos/frontend"
    github_repo: "myorg/frontend"
    project_key: "FRONTEND"
    
  - name: "backend"
    path: "~/repos/backend"
    github_repo: "myorg/backend"
    project_key: "BACKEND"

analysis:
  story_point_patterns:
    - "(?:story\\s*points?|sp|pts?)\\s*[:=]\\s*(\\d+)"
```

**Benefits:**
- Fine-grained control over which repositories to analyze
- Custom project keys and local paths
- Works with mixed-ownership repositories
- Compatible with existing configurations

### Directory Defaults

GitFlow Analytics now defaults cache and report directories to be relative to the configuration file location:

- **Reports**: Default to same directory as config file (unless overridden with `--output`)
- **Cache**: Default to `.gitflow-cache/` in config file directory
- **Backward compatibility**: Absolute paths in configuration continue to work as before

Example directory structure:
```
/project/
├── config.yaml          # Configuration file
├── weekly_metrics.csv    # Reports generated here by default
├── summary.csv
└── .gitflow-cache/       # Cache directory
    ├── gitflow_cache.db
    └── identities.db
```

## Command Line Interface

### Main Commands

```bash
# Analyze repositories
gitflow-analytics analyze -c config.yaml --weeks 12 --output ./reports

# Show cache statistics
gitflow-analytics cache-stats -c config.yaml

# List known developers
gitflow-analytics list-developers -c config.yaml

# Merge developer identities
gitflow-analytics merge-identity -c config.yaml dev1_id dev2_id

# Discover JIRA story point fields
gitflow-analytics discover-jira-fields -c config.yaml
```

### Options

- `--weeks, -w`: Number of weeks to analyze (default: 12)
- `--output, -o`: Output directory for reports (default: ./reports)
- `--anonymize`: Anonymize developer information
- `--no-cache`: Disable caching for fresh analysis
- `--clear-cache`: Clear cache before analysis
- `--validate-only`: Validate configuration without running

## Complete Configuration Example

Here's a complete example showing `.env` file and corresponding YAML configuration:

### `.env` file
```bash
# GitHub Configuration
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
GITHUB_ORG=EWTN-Global

# JIRA Configuration
JIRA_ACCESS_USER=developer@ewtn.com
JIRA_ACCESS_TOKEN=ATATT3xxxxxxxxxxx

# Optional: Other integrations
# CLICKUP_TOKEN=pk_xxxxxxxxxxxx
# LINEAR_TOKEN=lin_api_xxxxxxxxxxxx
```

### `config.yaml` file
```yaml
version: "1.0"

# GitHub configuration with organization discovery
github:
  token: "${GITHUB_TOKEN}"
  organization: "${GITHUB_ORG}"

# JIRA integration for story points
jira:
  access_user: "${JIRA_ACCESS_USER}"
  access_token: "${JIRA_ACCESS_TOKEN}"
  base_url: "https://ewtn.atlassian.net"

jira_integration:
  enabled: true
  fetch_story_points: true
  story_point_fields:
    - "Story point estimate"     # Your field name
    - "customfield_10016"        # Fallback field ID

# Analysis configuration
analysis:
  # Only track JIRA tickets (ignore GitHub issues, etc.)
  ticket_platforms:
    - jira
  
  # Exclude bot commits and boilerplate files
  exclude:
    authors:
      - "dependabot[bot]"
      - "renovate[bot]"
    paths:
      - "**/node_modules/**"
      - "**/*.min.js"
      - "**/package-lock.json"
  
  # Developer identity consolidation
  identity:
    similarity_threshold: 0.85
    manual_mappings:
      - primary_email: "john.doe@company.com"
        aliases:
          - "jdoe@oldcompany.com"
          - "john@personal.com"

# Output configuration
output:
  directory: "./reports"
  formats:
    - csv
    - markdown
```

## Output Reports

The tool generates three CSV reports:

1. **Weekly Metrics** (`weekly_metrics_YYYYMMDD.csv`)
   - Week-by-week developer productivity
   - Story points, commits, lines changed
   - Ticket coverage percentages
   - Per-project breakdown

2. **Summary Statistics** (`summary_YYYYMMDD.csv`)
   - Overall project statistics
   - Platform-specific ticket counts
   - Top contributors

3. **Developer Report** (`developers_YYYYMMDD.csv`)
   - Complete developer profiles
   - Total contributions
   - Identity aliases

## Story Point Patterns

Configure custom regex patterns to match your team's story point format:

```yaml
story_point_patterns:
  - "SP: (\\d+)"           # SP: 5
  - "\\[([0-9]+) pts\\]"   # [3 pts]
  - "estimate: (\\d+)"     # estimate: 8
```

## Ticket Platform Support

Automatically detects and tracks tickets from:
- **JIRA**: `PROJ-123`
- **GitHub**: `#123`, `GH-123`
- **ClickUp**: `CU-abc123`
- **Linear**: `ENG-123`

### JIRA Integration

GitFlow Analytics can fetch story points directly from JIRA tickets. Configure your JIRA instance:

```yaml
jira:
  access_user: "${JIRA_ACCESS_USER}"
  access_token: "${JIRA_ACCESS_TOKEN}"
  base_url: "https://your-company.atlassian.net"

jira_integration:
  enabled: true
  story_point_fields:
    - "Story point estimate"  # Your custom field name
    - "customfield_10016"     # Or use field ID
```

To discover your JIRA story point fields:
```bash
gitflow-analytics discover-jira-fields -c config.yaml
```

## Caching

The tool uses SQLite for intelligent caching:
- Commit analysis results
- Developer identity mappings
- Pull request data

Cache is automatically managed with configurable TTL.

## Developer Identity Resolution

Intelligently merges developer identities across:
- Different email addresses
- Name variations
- GitHub usernames

Manual overrides supported in configuration.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.