# GitFlow Analytics

A Python package for analyzing Git repositories to generate comprehensive developer productivity reports. It extracts data directly from Git history and GitHub APIs, providing weekly summaries, productivity insights, and gap analysis.

## Features

- 🚀 **Multi-repository analysis** with project grouping
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

```bash
pip install gitflow-analytics
```

### Basic Usage

1. Create a configuration file (`config.yaml`):

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

2. Set environment variables:

```bash
export GITHUB_TOKEN=your_github_token
export GITHUB_OWNER=your_github_org
```

3. Run the analysis:

```bash
gitflow-analytics analyze -c config.yaml
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
```

### Options

- `--weeks, -w`: Number of weeks to analyze (default: 12)
- `--output, -o`: Output directory for reports (default: ./reports)
- `--anonymize`: Anonymize developer information
- `--no-cache`: Disable caching for fresh analysis
- `--clear-cache`: Clear cache before analysis
- `--validate-only`: Validate configuration without running

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