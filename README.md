# GitFlow Analytics

[![PyPI version](https://badge.fury.io/py/gitflow-analytics.svg)](https://badge.fury.io/py/gitflow-analytics)
[![Python Support](https://img.shields.io/pypi/pyversions/gitflow-analytics.svg)](https://pypi.org/project/gitflow-analytics/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://github.com/bobmatnyc/gitflow-analytics/tree/main/docs)
[![Tests](https://github.com/bobmatnyc/gitflow-analytics/workflows/Tests/badge.svg)](https://github.com/bobmatnyc/gitflow-analytics/actions)

A comprehensive Python package for analyzing Git repositories to generate developer productivity insights without requiring external project management tools. Extract actionable metrics directly from Git history with ML-enhanced commit categorization, automated developer identity resolution, and professional reporting.

## 🚀 Key Features

- **🔍 Zero Dependencies**: Analyze productivity without requiring JIRA, Linear, or other PM tools
- **🧠 ML-Powered Intelligence**: Advanced commit categorization with 85-95% accuracy
- **👥 Smart Identity Resolution**: Automatically consolidate developer identities across email addresses
- **🏢 Enterprise Ready**: Organization-wide repository discovery with intelligent caching
- **📊 Professional Reports**: Rich markdown narratives and CSV exports for executive dashboards

## 🎯 Quick Start

Get up and running in 5 minutes:

```bash
# 1. Install GitFlow Analytics
pip install gitflow-analytics

# 2. Install ML dependencies (optional but recommended)
python -m spacy download en_core_web_sm

# 3. Create a simple configuration
echo 'version: "1.0"
github:
  token: "${GITHUB_TOKEN}"
  organization: "your-org"' > config.yaml

# 4. Set your GitHub token
echo 'GITHUB_TOKEN=ghp_your_token_here' > .env

# 5. Run analysis
gitflow-analytics -c config.yaml --weeks 8
```

**What you get:**
- 📈 Weekly metrics CSV with developer productivity trends
- 👥 Developer profiles with project distribution and work styles
- 🔍 Untracked work analysis with ML-powered categorization
- 📋 Executive summary with actionable insights
- 📊 Rich markdown report ready for stakeholders

### Sample Output Preview

```markdown
## Executive Summary
- **Total Commits**: 156 across 3 projects
- **Active Developers**: 5 team members
- **Ticket Coverage**: 73.2% (industry benchmark: 60-80%)
- **Top Contributor**: Sarah Chen (32 commits, FRONTEND focus)

## Key Insights
🎯 **High Productivity**: Team averaged 31 commits/week
📊 **Balanced Workload**: No single developer >40% of total work
✅ **Good Process**: 73% ticket coverage shows strong tracking
```

## ✨ Latest Features (v1.2.x)

- **🚀 Two-Step Processing**: Optimized fetch-then-classify workflow for better performance
- **💰 Cost Tracking**: Monitor LLM API usage with detailed token and cost reporting
- **⚡ Smart Caching**: Intelligent caching reduces analysis time by up to 90%
- **🔄 Automatic Updates**: Repositories automatically fetch latest commits before analysis
- **📊 Weekly Trends**: Track classification pattern changes over time
- **🎯 Enhanced Categorization**: All commits properly categorized with confidence scores

## 🔥 Core Capabilities

**📊 Analysis & Insights**
- Multi-repository analysis with intelligent project grouping
- ML-enhanced commit categorization (85-95% accuracy)
- Developer productivity metrics and work pattern analysis
- Story point extraction from commits and PRs
- Ticket tracking across JIRA, GitHub, ClickUp, and Linear

**🏢 Enterprise Features**
- Organization-wide repository discovery from GitHub
- Automated developer identity resolution and consolidation
- Database-backed caching for sub-second report generation
- Data anonymization for secure external sharing
- Batch processing optimized for large repositories

**📈 Professional Reporting**
- Rich markdown narratives with executive summaries
- Weekly CSV exports with trend analysis
- Customizable output formats and filtering
- Performance benchmarking and team comparisons

## 📚 Documentation

Comprehensive guides for every use case:

| **Getting Started** | **Advanced Usage** | **Integration** |
|-------------------|------------------|---------------|
| [Installation](docs/getting-started/installation.md) | [Complete Configuration](docs/guides/configuration.md) | [CLI Reference](docs/reference/cli-commands.md) |
| [5-Minute Tutorial](docs/getting-started/quickstart.md) | [ML Categorization](docs/guides/ml-categorization.md) | [JSON Export Schema](docs/reference/json-export-schema.md) |
| [First Analysis](docs/getting-started/first-analysis.md) | [Enterprise Setup](docs/examples/enterprise-setup.md) | [CI Integration](docs/examples/ci-integration.md) |

**🎯 Quick Links:**
- 📖 [**Documentation Hub**](docs/README.md) - Complete guide index
- 🚀 [**Quick Start**](docs/getting-started/quickstart.md) - Get running in 5 minutes
- ⚙️ [**Configuration**](docs/guides/configuration.md) - Full reference
- 🤝 [**Contributing**](docs/developer/contributing.md) - Join the project

## ⚡ Installation Options

### Standard Installation
```bash
pip install gitflow-analytics
```

### With ML Enhancement (Recommended)
```bash
pip install gitflow-analytics
python -m spacy download en_core_web_sm
```

### Development Installation
```bash
git clone https://github.com/bobmatnyc/gitflow-analytics.git
cd gitflow-analytics
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

## 🔧 Configuration

### Option 1: Organization Analysis (Recommended)
```yaml
# config.yaml
version: "1.0"
github:
  token: "${GITHUB_TOKEN}"
  organization: "your-org"  # Auto-discovers all repositories

analysis:
  ml_categorization:
    enabled: true
    min_confidence: 0.7
```

### Option 2: Specific Repositories
```yaml
# config.yaml  
version: "1.0"
github:
  token: "${GITHUB_TOKEN}"
  
repositories:
  - name: "my-app"
    path: "~/code/my-app"
    github_repo: "myorg/my-app"
    project_key: "APP"
```

### Environment Setup
```bash
# .env (same directory as config.yaml)
GITHUB_TOKEN=ghp_your_token_here
```

### Run Analysis
```bash
# Analyze last 8 weeks
gitflow-analytics -c config.yaml --weeks 8

# With custom output directory
gitflow-analytics -c config.yaml --weeks 8 --output ./reports
```

> 💡 **Need more configuration options?** See the [Complete Configuration Guide](docs/guides/configuration.md) for advanced features, integrations, and customization.

## 🎯 Excluding Merge Commits from Metrics

GitFlow Analytics can exclude merge commits from filtered line count calculations, following DORA metrics best practices.

### Why Exclude Merge Commits?

Merge commits represent repository management, not original development work:
- **Average merge commit**: 236.6 filtered lines vs 30.8 for regular commits (7.7x higher)
- Merge commits can **skew productivity metrics** and velocity calculations
- **DORA metrics best practice**: Focus on original development work, not repository management

### Configuration

Add this setting to your analysis configuration:

```yaml
analysis:
  # Exclude merge commits from filtered line counts (DORA metrics best practice)
  exclude_merge_commits: true  # Default: false
```

### Impact Example

Real metrics from EWTN dataset analysis:

| Metric | With Merge Commits | Without Merge Commits | Change |
|--------|-------------------|----------------------|--------|
| **Total Filtered Lines** | 138,730 | 54,808 | -60% |
| **Merge Commits** | 355 commits | 355 commits | (excluded from line counts) |
| **Regular Commits** | 1,426 commits | 1,426 commits | (unchanged) |

### What Gets Excluded?

When `exclude_merge_commits: true`:

✅ **Filtered Stats**: Merge commits (2+ parents) have `filtered_insertions = 0` and `filtered_deletions = 0`
✅ **Raw Stats**: Always preserved for all commits (accurate commit counts)
✅ **Reports**: Line count metrics reflect only original development work

❌ **Not affected**: Commit counts, developer activity tracking, ticket references

### When to Use

**✅ Enable when:**
- You want DORA-compliant metrics for productivity tracking
- Your workflow uses merge commits for pull requests
- You need accurate developer velocity without repository overhead
- You're comparing metrics across teams with different merge strategies

**❌ Disable when:**
- You want to track all repository activity including management overhead
- Merge commits represent significant manual conflict resolution in your workflow
- You're analyzing repositories without merge-heavy workflows
- You need to measure total repository churn including merges

### Example Configuration

```yaml
# Full configuration example
analysis:
  weeks_back: 8
  include_weekends: true

  # DORA-compliant metrics: exclude merge commits
  exclude_merge_commits: true

  # Analyze ALL branches to capture feature branch work
  branch_patterns:
    - "*"  # Include all branches (feature, develop, hotfix, etc.)
```

> 💡 **Pro Tip**: Combine `exclude_merge_commits: true` with `branch_patterns: ["*"]` to analyze all development work without merge overhead.

## 📊 Generated Reports

GitFlow Analytics generates comprehensive reports for different audiences:

### 📈 CSV Data Files
- **weekly_metrics.csv** - Developer productivity trends by week
- **weekly_velocity.csv** - Lines-per-story-point velocity analysis
- **developers.csv** - Complete team profiles and statistics  
- **summary.csv** - Project-wide statistics and benchmarks
- **untracked_commits.csv** - ML-categorized uncommitted work analysis

### 📋 Executive Reports
- **narrative_summary.md** - Rich markdown report with:
  - Executive summary with key metrics
  - Team composition and work distribution  
  - Project activity breakdown
  - Development patterns and recommendations
  - Weekly trend analysis

### Sample Executive Summary
```markdown
## Executive Summary
- **Total Commits**: 324 commits across 4 projects
- **Active Developers**: 8 team members  
- **Ticket Coverage**: 78.4% (above industry benchmark)
- **Top Areas**: Frontend (45%), API (32%), Infrastructure (23%)

## Key Insights  
✅ **Strong Process Adherence**: 78% ticket coverage
🎯 **Balanced Team**: No developer >35% of total work
📈 **Growth Trend**: +15% productivity vs last quarter
```

## 🛠️ Common Use Cases

**👥 Team Lead Dashboard**
- Track individual developer productivity and growth
- Identify workload distribution and potential burnout
- Monitor code quality trends and technical debt

**📈 Engineering Management**  
- Generate executive reports on team velocity
- Analyze process adherence and ticket coverage
- Benchmark performance across projects and quarters

**🔍 Process Optimization**
- Identify untracked work patterns that should be formalized
- Optimize developer focus and reduce context switching  
- Improve estimation accuracy with historical data

**🏢 Enterprise Analytics**
- Organization-wide repository analysis across dozens of projects
- Automated identity resolution for large, distributed teams
- Cost-effective analysis without expensive PM tool dependencies

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

# Discover story point fields in your PM platform
gitflow-analytics discover-storypoint-fields -c config.yaml
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
GITHUB_ORG=your-organization

# PM Platform Configuration
JIRA_ACCESS_USER=developer@company.com
JIRA_ACCESS_TOKEN=ATATT3xxxxxxxxxxx
LINEAR_API_KEY=lin_api_xxxxxxxxxxxx
CLICKUP_API_TOKEN=pk_xxxxxxxxxxxx

# Note: GitHub Issues uses GITHUB_TOKEN automatically
```

### `config.yaml` file
```yaml
version: "1.0"

# GitHub configuration with organization discovery
github:
  token: "${GITHUB_TOKEN}"
  organization: "${GITHUB_ORG}"

# Multi-platform PM integration
pm:
  jira:
    access_user: "${JIRA_ACCESS_USER}"
    access_token: "${JIRA_ACCESS_TOKEN}"
    base_url: "https://company.atlassian.net"

  linear:
    api_key: "${LINEAR_API_KEY}"
    team_ids: ["team_123abc"]  # Optional: filter by specific teams

  clickup:
    api_token: "${CLICKUP_API_TOKEN}"
    workspace_url: "https://app.clickup.com/12345/v/"

# JIRA story point integration (optional)
jira_integration:
  enabled: true
  fetch_story_points: true
  story_point_fields:
    - "Story point estimate"     # Your field name
    - "customfield_10016"        # Fallback field ID

# Analysis configuration
analysis:
  # Track tickets from all configured platforms
  ticket_platforms:
    - jira
    - linear
    - clickup
    - github  # GitHub Issues (uses GITHUB_TOKEN)
  
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

2. **Weekly Velocity** (`weekly_velocity_YYYYMMDD.csv`)
   - Lines of code per story point analysis
   - Efficiency trends and velocity patterns
   - PR-based vs commit-based story points breakdown
   - Team velocity benchmarking and week-over-week trends

3. **Summary Statistics** (`summary_YYYYMMDD.csv`)
   - Overall project statistics
   - Platform-specific ticket counts
   - Top contributors

4. **Developer Report** (`developers_YYYYMMDD.csv`)
   - Complete developer profiles
   - Total contributions
   - Identity aliases

5. **Untracked Commits Report** (`untracked_commits_YYYYMMDD.csv`)
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
   - **Project Activity**: Detailed breakdown by project with contributor percentages and **commit classifications**
   - **Development Patterns**: Key insights from productivity and collaboration analysis
   - **Pull Request Analysis**: PR metrics including size, lifetime, and review activity
   - **Weekly Trends** (v1.1.0+): Week-over-week changes in classification patterns

6. **Database-Backed Qualitative Report** (`database_qualitative_report_YYYYMMDD.md`) (v1.1.0+)
   - Generated directly from SQLite storage for fast retrieval
   - Includes weekly trend analysis per developer/project
   - Shows classification changes over time (e.g., "Features: +15%, Bug Fixes: -5%")
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

Automatically detects and tracks tickets from multiple PM platforms:
- **JIRA**: `PROJ-123`
- **GitHub Issues**: `#123`, `GH-123`
- **ClickUp**: `CU-abc123`
- **Linear**: `ENG-123`

### Multi-Platform PM Integration

GitFlow Analytics supports multiple project management platforms simultaneously. You can configure one or more platforms based on your team's workflow:

```yaml
# Configure which platforms to track
analysis:
  ticket_platforms:
    - jira
    - linear
    - clickup
    - github  # GitHub Issues

# Platform-specific configuration
pm:
  jira:
    access_user: "${JIRA_ACCESS_USER}"
    access_token: "${JIRA_ACCESS_TOKEN}"
    base_url: "https://your-company.atlassian.net"

  linear:
    api_key: "${LINEAR_API_KEY}"
    team_ids:  # Optional: filter by team
      - "team_123abc"

  clickup:
    api_token: "${CLICKUP_API_TOKEN}"
    workspace_url: "https://app.clickup.com/12345/v/"

# GitHub Issues uses existing GitHub token automatically
github:
  token: "${GITHUB_TOKEN}"
```

### Platform Setup Guides

#### JIRA Setup
1. **Get API Token**: Go to [Atlassian API Tokens](https://id.atlassian.com/manage-profile/security/api-tokens)
2. **Required Permissions**: Read access to projects and issues
3. **Configuration**:
   ```yaml
   pm:
     jira:
       access_user: "${JIRA_ACCESS_USER}"  # Your Atlassian email
       access_token: "${JIRA_ACCESS_TOKEN}"
       base_url: "https://your-company.atlassian.net"
   ```

#### Linear Setup
1. **Get API Key**: Go to [Linear Settings → API](https://linear.app/settings/api)
2. **Required Permissions**: Read access to issues
3. **Configuration**:
   ```yaml
   pm:
     linear:
       api_key: "${LINEAR_API_KEY}"
       team_ids: ["team_123abc"]  # Optional: specify team IDs
   ```

#### ClickUp Setup
1. **Get API Token**: Go to [ClickUp Settings → Apps](https://app.clickup.com/settings/apps)
2. **Get Workspace URL**: Copy from browser when viewing your workspace
3. **Configuration**:
   ```yaml
   pm:
     clickup:
       api_token: "${CLICKUP_API_TOKEN}"
       workspace_url: "https://app.clickup.com/12345/v/"
   ```

#### GitHub Issues Setup
GitHub Issues is automatically enabled when GitHub integration is configured. No additional setup required:
```yaml
github:
  token: "${GITHUB_TOKEN}"  # Same token for repo access and issues
```

### JIRA Story Point Integration

GitFlow Analytics can fetch story points directly from JIRA tickets:

```yaml
jira_integration:
  enabled: true
  fetch_story_points: true
  story_point_fields:
    - "Story point estimate"  # Your custom field name
    - "customfield_10016"     # Or use field ID
```

To discover your JIRA story point fields:
```bash
gitflow-analytics discover-storypoint-fields -c config.yaml
```

### Environment Variables for Credentials

Store credentials securely in a `.env` file:

```bash
# .env file (keep this secure and don't commit to git!)
GITHUB_TOKEN=ghp_your_token_here

# PM Platform Credentials
JIRA_ACCESS_USER=your.email@company.com
JIRA_ACCESS_TOKEN=ATATT3xxxxxxxxxxx
LINEAR_API_KEY=lin_api_xxxxxxxxxxxx
CLICKUP_API_TOKEN=pk_xxxxxxxxxxxx
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
        primary_email: "john.smith@company.com"
        aliases:
          - "150280367+jsmith@users.noreply.github.com"
          - "jsmith-company@users.noreply.github.com"
      
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
- **Fallback**: Gracefully disables qualitative analysis if spaCy model unavailable (provides helpful error messages)
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

### Development Setup

```bash
# Clone the repository
git clone https://github.com/bobmatnyc/gitflow-analytics.git
cd gitflow-analytics

# Install development dependencies
make install-dev

# Run tests
make test

# Format code
make format

# Run all quality checks
make quality-gate
```

### Release Workflow

This project uses a Makefile-based release workflow for simplicity and transparency. See [RELEASE.md](RELEASE.md) for detailed documentation.

**Quick Reference:**
```bash
make release-patch   # Bug fixes (3.13.1 → 3.13.2)
make release-minor   # New features (3.13.1 → 3.14.0)
make release-major   # Breaking changes (3.13.1 → 4.0.0)
```

For more details, see:
- [RELEASE.md](RELEASE.md) - Comprehensive release guide
- [RELEASE_QUICKREF.md](RELEASE_QUICKREF.md) - Quick reference card
- `make help` - All available commands

## License

This project is licensed under the MIT License - see the LICENSE file for details.