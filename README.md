# GitFlow Analytics

[![PyPI version](https://badge.fury.io/py/gitflow-analytics.svg)](https://badge.fury.io/py/gitflow-analytics)
[![Python Support](https://img.shields.io/pypi/pyversions/gitflow-analytics.svg)](https://pypi.org/project/gitflow-analytics/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for analyzing Git repositories to generate comprehensive developer productivity reports. It extracts data directly from Git history and GitHub APIs, providing weekly summaries, productivity insights, and gap analysis.

## Features

- 🚀 **Multi-repository analysis** with project grouping
- 🏢 **Organization-based repository discovery** from GitHub
- 👥 **Interactive developer identity resolution** with automatic suggestions
- 📊 **Work volume analysis** (absolute vs relative effort)
- 🎯 **Story point extraction** from commit messages and PR descriptions
- 🎫 **Multi-platform ticket tracking** (JIRA, GitHub Issues, ClickUp, Linear)
- 🧠 **ML-enhanced commit categorization** with confidence scoring and semantic analysis
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

**With ML Features (Recommended for enhanced categorization):**

```bash
pip install gitflow-analytics
python -m spacy download en_core_web_sm
```

**From Source (Development):**

```bash
git clone https://github.com/bobmatnyc/gitflow-analytics.git
cd gitflow-analytics
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

> **Note**: The spaCy English model is required for ML-enhanced commit categorization. If not available, the system will gracefully fall back to rule-based categorization.

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
  
  # Enhanced untracked commit analysis settings
  untracked_analysis:
    file_threshold: 1  # Minimum files changed to include commit (default: 1)
    # Categories that are acceptable to be untracked
    acceptable_categories:
      - "maintenance" 
      - "style"
      - "documentation"
  
  # ML-enhanced commit categorization (optional)
  ml_categorization:
    enabled: true  # Enable ML categorization (default: true)
    min_confidence: 0.6  # Minimum confidence for ML predictions
    hybrid_threshold: 0.5  # Threshold for ML vs rule-based fallback
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
# Default behavior - analyze command runs automatically
gitflow-analytics -c config.yaml

# Explicit analyze command (backward compatibility)
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
# Analyze repositories (default command)
gitflow-analytics -c config.yaml --weeks 12 --output ./reports

# Explicit analyze command (backward compatibility)
gitflow-analytics analyze -c config.yaml --weeks 12 --output ./reports

# Show cache statistics
gitflow-analytics cache-stats -c config.yaml

# List known developers
gitflow-analytics list-developers -c config.yaml

# Analyze developer identities
gitflow-analytics identities -c config.yaml

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
- `--skip-identity-analysis`: Skip automatic identity analysis
- `--apply-identity-suggestions`: Apply identity suggestions without prompting

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
      - name: "John Doe"
        primary_email: "john.doe@company.com"
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

The tool generates comprehensive CSV reports and markdown summaries:

### CSV Reports

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

4. **Untracked Commits Report** (`untracked_commits_YYYYMMDD.csv`)
   - Detailed analysis of commits without ticket references
   - Commit categorization (bug_fix, feature, refactor, documentation, maintenance, test, style, build)
   - Enhanced metadata: commit hash, author, timestamp, project, message, file/line changes
   - Configurable file change threshold for filtering significant commits

### Enhanced Untracked Commit Analysis

The untracked commits report provides deep insights into work that bypasses ticket tracking:

**CSV Columns:**
- `commit_hash` / `short_hash`: Full and abbreviated commit identifiers
- `author` / `author_email` / `canonical_id`: Developer identification (with anonymization support)
- `date`: Commit timestamp
- `project`: Project key for multi-repository analysis
- `message`: Commit message (truncated for readability)
- `category`: Automated categorization of work type
- `files_changed` / `lines_added` / `lines_removed` / `lines_changed`: Change metrics
- `is_merge`: Boolean flag for merge commits

**Automatic Categorization:**
- **Feature**: New functionality development (`add`, `new`, `implement`, `create`)
- **Bug Fix**: Error corrections (`fix`, `bug`, `error`, `resolve`, `hotfix`)
- **Refactor**: Code restructuring (`refactor`, `optimize`, `improve`, `cleanup`)
- **Documentation**: Documentation updates (`doc`, `readme`, `comment`, `guide`)
- **Maintenance**: Routine upkeep (`update`, `upgrade`, `dependency`, `config`)
- **Test**: Testing-related changes (`test`, `spec`, `mock`, `fixture`)
- **Style**: Formatting changes (`format`, `lint`, `prettier`, `whitespace`)
- **Build**: Build system changes (`build`, `compile`, `ci`, `docker`)

### Markdown Reports

5. **Narrative Summary** (`narrative_summary_YYYYMMDD.md`)
   - **Executive Summary**: High-level metrics and team overview
   - **Team Composition**: Developer profiles with project percentages and work patterns
   - **Project Activity**: Detailed breakdown by project with contributor percentages
   - **Development Patterns**: Key insights from productivity and collaboration analysis
   - **Pull Request Analysis**: PR metrics including size, lifetime, and review activity
   - **Issue Tracking**: Platform usage and coverage analysis with simplified display
   - **Enhanced Untracked Work Analysis**: Comprehensive categorization with dual percentage metrics
   - **PM Platform Integration**: Story point tracking and correlation insights (when available)
   - **Recommendations**: Actionable insights based on analysis patterns

### Enhanced Narrative Report Sections

The narrative report provides comprehensive insights through multiple detailed sections:

#### Team Composition Section
- **Developer Profiles**: Individual developer statistics with commit counts
- **Project Distribution**: Shows ALL projects each developer works on with precise percentages
- **Work Style Classification**: Categorizes developers as "Focused", "Multi-project", or "Highly Focused"
- **Activity Patterns**: Identifies time patterns like "Standard Hours" or "Extended Hours"

**Example developer profile:**
```markdown
**John Developer**
- Commits: 15
- Projects: FRONTEND (85.0%), SERVICE_TS (15.0%)
- Work Style: Focused
- Active Pattern: Standard Hours
```

#### Project Activity Section
- **Activity by Project**: Commits and percentage of total activity per project
- **Contributor Breakdown**: Shows each developer's contribution percentage within each project
- **Lines Changed**: Quantifies the scale of changes per project

#### Issue Tracking with Simplified Display
- **Platform Usage**: Clean display of ticket platform distribution (JIRA, GitHub, etc.)
- **Coverage Analysis**: Percentage of commits that reference tickets
- **Enhanced Untracked Work Analysis**: Detailed categorization and recommendations

### Interpreting Dual Percentage Metrics

The enhanced untracked work analysis provides two key percentage metrics for better context:

1. **Percentage of Total Untracked Work**: Shows how much each developer contributes to the overall untracked work pool
2. **Percentage of Developer's Individual Work**: Shows what proportion of a specific developer's commits are untracked

**Example interpretation:**
```
- John Doe: 25 commits (40% of untracked, 15% of their work) - maintenance, style
```

This means:
- John contributed 25 untracked commits
- These represent 40% of all untracked commits in the analysis period  
- Only 15% of John's total work was untracked (85% was properly tracked)
- Most untracked work was maintenance and style changes (acceptable categories)

**Process Insights:**
- High "% of untracked" + low "% of their work" = Developer doing most of the acceptable maintenance work
- Low "% of untracked" + high "% of their work" = Developer needs process guidance
- High percentages in feature/bug_fix categories = Process improvement opportunity

### Example Report Outputs

#### Untracked Commits CSV Sample
```csv
commit_hash,short_hash,author,author_email,canonical_id,date,project,message,category,files_changed,lines_added,lines_removed,lines_changed,is_merge
a1b2c3d4e5f6...,a1b2c3d,John Doe,john@company.com,ID0001,2024-01-15 14:30:22,FRONTEND,Update dependency versions for security patches,maintenance,2,45,12,57,false
f6e5d4c3b2a1...,f6e5d4c,Jane Smith,jane@company.com,ID0002,2024-01-15 09:15:10,BACKEND,Fix typo in error message,bug_fix,1,1,1,2,false
9876543210ab...,9876543,Bob Wilson,bob@company.com,ID0003,2024-01-14 16:45:33,FRONTEND,Add JSDoc comments to utility functions,documentation,3,28,0,28,false
```

#### Complete Narrative Report Sample
```markdown
# GitFlow Analytics Report

**Generated**: 2025-08-04 14:27:47
**Analysis Period**: Last 4 weeks

## Executive Summary

- **Total Commits**: 35
- **Active Developers**: 3
- **Lines Changed**: 910
- **Ticket Coverage**: 71.4%
- **Active Projects**: FRONTEND, SERVICE_TS, SERVICES
- **Top Contributor**: John Developer with 15 commits

## Team Composition

### Developer Profiles

**John Developer**
- Commits: 15
- Projects: FRONTEND (85.0%), SERVICE_TS (15.0%)
- Work Style: Focused
- Active Pattern: Standard Hours

**Jane Smith**
- Commits: 12
- Projects: SERVICE_TS (70.0%), FRONTEND (30.0%)
- Work Style: Multi-project
- Active Pattern: Extended Hours

## Project Activity

### Activity by Project

**FRONTEND**
- Commits: 14 (50.0% of total)
- Lines Changed: 450
- Contributors: John Developer (71.4%), Jane Smith (28.6%)

**SERVICE_TS**
- Commits: 8 (28.6% of total)
- Lines Changed: 280
- Contributors: Jane Smith (100.0%)

## Issue Tracking

### Platform Usage

- **Jira**: 15 tickets (60.0%)
- **Github**: 8 tickets (32.0%)
- **Clickup**: 2 tickets (8.0%)

### Untracked Work Analysis

**Summary**: 10 commits (28.6% of total) lack ticket references.

#### Work Categories

- **Maintenance**: 4 commits (40.0%), avg 23 lines *(acceptable untracked)*
- **Bug Fix**: 3 commits (30.0%), avg 15 lines *(should be tracked)*
- **Documentation**: 2 commits (20.0%), avg 12 lines *(acceptable untracked)*

#### Top Contributors (Untracked Work)

- **John Developer**: 1 commits (50.0% of untracked, 6.7% of their work) - *refactor*
- **Jane Smith**: 1 commits (50.0% of untracked, 8.3% of their work) - *style*

#### Recommendations for Untracked Work

🎯 **Excellent tracking**: Less than 20% of commits are untracked - the team shows strong process adherence.

## Recommendations

✅ The team shows healthy development patterns. Continue current practices while monitoring for changes.
```

### Configuration for Enhanced Narrative Reports

The narrative reports automatically include all available sections based on your configuration and data availability:

**Always Generated:**
- Executive Summary, Team Composition, Project Activity, Development Patterns, Issue Tracking, Recommendations

**Conditionally Generated:**
- **Pull Request Analysis**: Requires GitHub integration with PR data
- **PM Platform Integration**: Requires JIRA or other PM platform configuration
- **Qualitative Analysis**: Requires ChatGPT integration setup

**Customizing Report Content:**
```yaml
# config.yaml
output:
  formats:
    - csv
    - markdown  # Enables narrative report generation
  
# Optional: Enhance narrative reports with additional data
jira:
  access_user: "${JIRA_ACCESS_USER}"
  access_token: "${JIRA_ACCESS_TOKEN}"
  base_url: "https://company.atlassian.net"

# Optional: Add qualitative insights
analysis:
  chatgpt:
    enabled: true
    api_key: "${OPENAI_API_KEY}"
```

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

GitFlow Analytics intelligently consolidates developer identities across different email addresses and name variations:

### Automatic Identity Analysis (New!)

Identity analysis now runs **automatically by default** when no manual mappings exist. The system will:

1. **Analyze all developer identities** in your commits
2. **Show suggested consolidations** with a clear preview
3. **Prompt for approval** with a simple Y/n
4. **Update your configuration** automatically
5. **Continue analysis** with consolidated identities

Example of the interactive prompt:
```
🔍 Analyzing developer identities...

⚠️  Found 3 potential identity clusters:

📋 Suggested identity mappings:
   john.doe@company.com
     → 123456+johndoe@users.noreply.github.com
     → jdoe@personal.email.com

🤖 Found 2 bot accounts to exclude:
   - dependabot[bot]
   - renovate[bot]

────────────────────────────────────────────────────────────
Apply these identity mappings to your configuration? [Y/n]: 
```

This prompt appears at most once every 7 days. 

To skip automatic identity analysis:
```bash
# Simplified syntax (default)
gitflow-analytics -c config.yaml --skip-identity-analysis

# Explicit analyze command
gitflow-analytics analyze -c config.yaml --skip-identity-analysis
```

To manually run identity analysis:
```bash
gitflow-analytics identities -c config.yaml
```

### Smart Identity Matching

The system automatically detects:
- **GitHub noreply emails** (e.g., `150280367+username@users.noreply.github.com`)
- **Name variations** (e.g., "John Doe" vs "John D" vs "jdoe")
- **Common email patterns** across domains
- **Bot accounts** for automatic exclusion

### Manual Configuration

You can also manually configure identity mappings in your YAML:

```yaml
analysis:
  identity:
    manual_mappings:
      - name: "John Doe"  # Optional: preferred display name for reports
        primary_email: john.doe@company.com
        aliases:
          - jdoe@personal.email.com
          - 123456+johndoe@users.noreply.github.com
      - name: "Sarah Smith"
        primary_email: sarah.smith@company.com
        aliases:
          - s.smith@oldcompany.com
```

### Display Name Control

The optional `name` field in manual mappings allows you to control how developer names appear in reports. This is particularly useful for:

- **Standardizing display names** across different email formats
- **Resolving duplicates** when the same person appears with slight name variations
- **Using preferred names** instead of technical email formats

**Example use cases:**
```yaml
analysis:
  identity:
    manual_mappings:
      # Consolidate Austin Zach identities
      - name: "Austin Zach"
        primary_email: "azach@ewtn.com"
        aliases:
          - "150280367+azach-ewtn@users.noreply.github.com"
          - "azach-ewtn@users.noreply.github.com"
      
      # Standardize name variations
      - name: "John Doe"  # Consistent display across all reports
        primary_email: "john.doe@company.com"
        aliases:
          - "johndoe@company.com"
          - "j.doe@company.com"
```

Without the `name` field, the system uses the canonical email's associated name, which might not be ideal for reporting.

### Disabling Automatic Analysis

To disable the automatic identity prompt:
```yaml
analysis:
  identity:
    auto_analysis: false
```

## ML-Enhanced Commit Categorization

GitFlow Analytics includes sophisticated machine learning capabilities for categorizing commits with high accuracy and confidence scoring.

### How It Works

The ML categorization system uses a **hybrid approach** combining:

1. **Semantic Analysis**: Uses spaCy NLP models to understand commit message meaning
2. **File Pattern Recognition**: Analyzes changed files for additional context signals  
3. **Rule-based Fallback**: Falls back to traditional regex patterns when ML confidence is low
4. **Confidence Scoring**: Provides confidence metrics for all categorizations

### Categories Detected

The system automatically categorizes commits into:

- **Feature**: New functionality development (`add`, `implement`, `create`)
- **Bug Fix**: Error corrections (`fix`, `resolve`, `correct`)
- **Refactor**: Code restructuring (`refactor`, `optimize`, `improve`) 
- **Documentation**: Documentation updates (`docs`, `readme`, `comment`)
- **Maintenance**: Routine upkeep (`update`, `upgrade`, `dependency`)
- **Test**: Testing-related changes (`test`, `spec`, `coverage`)
- **Style**: Formatting changes (`format`, `lint`, `prettier`)
- **Build**: Build system changes (`build`, `ci`, `docker`)
- **Security**: Security-related fixes (`security`, `vulnerability`)
- **Hotfix**: Urgent production fixes (`hotfix`, `critical`, `emergency`)
- **Config**: Configuration changes (`config`, `settings`, `environment`)

### Configuration

```yaml
analysis:
  ml_categorization:
    # Enable/disable ML categorization (default: true)
    enabled: true
    
    # Minimum confidence for ML predictions (0.0-1.0, default: 0.6)
    min_confidence: 0.6
    
    # Semantic vs file pattern weighting (default: 0.7 vs 0.3)
    semantic_weight: 0.7
    file_pattern_weight: 0.3
    
    # Confidence threshold for ML vs rule-based (default: 0.5)
    hybrid_threshold: 0.5
    
    # Caching for performance
    enable_caching: true
    cache_duration_days: 30
    
    # Processing settings
    batch_size: 100
```

### Installation Requirements

For ML categorization, install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

**Alternative models** (if the default is unavailable):
```bash
# Medium model (more accurate, larger)
python -m spacy download en_core_web_md

# Large model (most accurate, largest) 
python -m spacy download en_core_web_lg
```

### Performance Expectations

- **Accuracy**: 85-95% accuracy on typical commit messages
- **Speed**: ~50-100 commits/second with caching enabled
- **Fallback**: Graceful degradation to rule-based when ML unavailable
- **Memory**: ~200MB additional memory usage for spaCy models

### Enhanced Reports

With ML categorization enabled, reports include:

- **Confidence scores** for each categorization
- **Method indicators** (ML, rules, or cached)
- **Alternative predictions** for uncertain cases
- **ML performance statistics** in analysis summaries

### Example Enhanced Output

```csv
commit_hash,category,ml_confidence,ml_method,message
a1b2c3d,feature,0.89,ml,"Add user authentication system"  
f6e5d4c,bug_fix,0.92,ml,"Fix memory leak in cache cleanup"
9876543,maintenance,0.74,rules,"Update dependency versions"
```

## Troubleshooting

### YAML Configuration Errors

GitFlow Analytics provides helpful error messages when YAML configuration issues are encountered. Here are common errors and their solutions:

#### Tab Characters Not Allowed
```
❌ YAML configuration error at line 3, column 1:
🚫 Tab characters are not allowed in YAML files!
```
**Fix**: Replace all tabs with spaces (use 2 or 4 spaces for indentation)
- Most editors can show whitespace characters and convert tabs to spaces
- In VS Code: View → Render Whitespace, then Edit → Convert Indentation to Spaces

#### Missing Colons
```
❌ YAML configuration error at line 5, column 10:
🚫 Missing colon (:) after a key name!
```
**Fix**: Add a colon and space after each key name
```yaml
# Correct:
repositories:
  - name: my-repo
    
# Incorrect:
repositories
  - name my-repo
```

#### Unclosed Quotes
```
❌ YAML configuration error at line 8, column 15:
🚫 Unclosed quoted string!
```
**Fix**: Ensure all quotes are properly closed
```yaml
# Correct:
token: "my-token-value"

# Incorrect:
token: "my-token-value
```

#### Invalid Indentation
```
❌ YAML configuration error:
🚫 Indentation error or invalid structure!
```
**Fix**: Use consistent indentation (either 2 or 4 spaces)
```yaml
# Correct:
analysis:
  exclude:
    paths:
      - "vendor/**"
      
# Incorrect:
analysis:
  exclude:
     paths:  # 3 spaces - inconsistent!
      - "vendor/**"
```

### Tips for Valid YAML

1. **Use a YAML validator**: Check your configuration with online YAML validators before using
2. **Enable whitespace display**: Make tabs and spaces visible in your editor
3. **Use quotes for special characters**: Wrap values containing `:`, `#`, `@`, etc. in quotes
4. **Consistent indentation**: Pick 2 or 4 spaces and stick to it throughout the file
5. **Check the sample config**: Reference `config-sample.yaml` for proper structure

### Configuration Validation

Beyond YAML syntax, GitFlow Analytics validates:
- Required fields (`repositories` must have `name` and `path`)
- Environment variable resolution
- File path existence
- Valid configuration structure

If you encounter persistent issues, run with `--debug` for detailed error information:
```bash
# Simplified syntax (default)
gitflow-analytics -c config.yaml --debug

# Explicit analyze command
gitflow-analytics analyze -c config.yaml --debug
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.