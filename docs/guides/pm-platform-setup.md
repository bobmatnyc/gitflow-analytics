# Project Management Platform Setup Guide

GitFlow Analytics supports multiple project management platforms for ticket tracking and story point extraction. This guide covers setup for all supported platforms.

## Overview

GitFlow Analytics can track tickets from multiple PM platforms simultaneously:
- **JIRA** - Atlassian's project management tool
- **Linear** - Modern issue tracking
- **ClickUp** - All-in-one productivity platform
- **GitHub Issues** - Native GitHub issue tracking

## Quick Start

The easiest way to set up PM platforms is using the interactive launcher:

```bash
gitflow-analytics launcher
```

Select **Profile 1 (Standard)** and follow the prompts to configure your desired platforms.

## Platform Configuration

### JIRA Setup

JIRA is widely used for enterprise project management with extensive customization options.

#### Prerequisites
- JIRA account with access to projects
- API token (recommended) or password
- Base URL of your JIRA instance

#### Getting JIRA Credentials

1. **Generate API Token**:
   - Go to [Atlassian API Tokens](https://id.atlassian.com/manage-profile/security/api-tokens)
   - Click "Create API token"
   - Give it a label (e.g., "GitFlow Analytics")
   - Copy the generated token (you won't see it again!)

2. **Find Your Base URL**:
   - Your JIRA URL format: `https://your-company.atlassian.net`
   - Cloud JIRA: Use the full domain
   - Self-hosted: Use your server URL

#### Configuration

Add to your `config.yaml`:

```yaml
pm:
  jira:
    access_user: "${JIRA_ACCESS_USER}"
    access_token: "${JIRA_ACCESS_TOKEN}"
    base_url: "https://your-company.atlassian.net"

# Optional: Story point integration
jira_integration:
  enabled: true
  fetch_story_points: true
  story_point_fields:
    - "Story point estimate"  # Your custom field name
    - "customfield_10016"     # Or field ID

analysis:
  ticket_platforms:
    - jira
```

Add to your `.env` file:

```bash
JIRA_ACCESS_USER=your.email@company.com
JIRA_ACCESS_TOKEN=ATATT3xxxxxxxxxxx
```

#### Required Permissions
- Read access to projects
- Read access to issues
- Read access to custom fields (for story points)

#### Discovering Story Point Fields

JIRA story points are often stored in custom fields. To find yours:

```bash
gitflow-analytics discover-storypoint-fields -c config.yaml
```

This command lists all available custom fields in your JIRA instance.

#### Common Issues

**Authentication Failed**:
- Verify email address matches JIRA account
- Regenerate API token if expired
- Check base URL format (include `https://`)

**Story Points Not Found**:
- Run `discover-storypoint-fields` to find correct field
- Story points might use a different field name
- Some boards don't use story points

---

### Linear Setup

Linear is a modern issue tracking tool focused on speed and user experience.

#### Prerequisites
- Linear account with workspace access
- API key with read permissions
- Optional: Team IDs for filtering

#### Getting Linear Credentials

1. **Generate API Key**:
   - Go to [Linear Settings → API](https://linear.app/settings/api)
   - Click "Create new key"
   - Give it a label (e.g., "GitFlow Analytics")
   - Copy the generated key (starts with `lin_api_`)

2. **Find Team IDs** (Optional):
   - In Linear, go to your team settings
   - Team ID is in the URL: `linear.app/team-id/team`
   - Or use GraphQL API to query teams

#### Configuration

Add to your `config.yaml`:

```yaml
pm:
  linear:
    api_key: "${LINEAR_API_KEY}"
    team_ids:  # Optional: filter by specific teams
      - "team_123abc"
      - "team_456def"

analysis:
  ticket_platforms:
    - linear
```

Add to your `.env` file:

```bash
LINEAR_API_KEY=lin_api_xxxxxxxxxxxx
```

#### Required Permissions
- Read access to issues
- Read access to teams (if using team filters)

#### Team Filtering

If you don't specify `team_ids`, Linear will track issues from all teams you have access to. For large organizations, filtering by team improves performance:

```yaml
pm:
  linear:
    api_key: "${LINEAR_API_KEY}"
    team_ids: ["team_123abc"]  # Only track this team's issues
```

#### Common Issues

**API Key Invalid**:
- Verify key starts with `lin_api_`
- Regenerate if compromised
- Check key hasn't been revoked in Linear settings

**No Issues Found**:
- Verify team IDs are correct
- Check API key has access to specified teams
- Ensure issues exist in the analysis period

---

### ClickUp Setup

ClickUp is an all-in-one productivity platform with flexible task management.

#### Prerequisites
- ClickUp account with workspace access
- API token
- Workspace URL

#### Getting ClickUp Credentials

1. **Generate API Token**:
   - Go to [ClickUp Settings → Apps](https://app.clickup.com/settings/apps)
   - Under "API Token", click "Generate"
   - Copy the token (starts with `pk_`)

2. **Find Workspace URL**:
   - Navigate to your ClickUp workspace
   - Copy URL from browser: `https://app.clickup.com/12345/v/`
   - The number `12345` is your workspace ID

#### Configuration

Add to your `config.yaml`:

```yaml
pm:
  clickup:
    api_token: "${CLICKUP_API_TOKEN}"
    workspace_url: "https://app.clickup.com/12345/v/"

analysis:
  ticket_platforms:
    - clickup
```

Add to your `.env` file:

```bash
CLICKUP_API_TOKEN=pk_xxxxxxxxxxxx
```

#### Required Permissions
- Read access to tasks
- Read access to workspace

#### Workspace URL Format

The workspace URL must include the `/v/` suffix:
- ✅ Correct: `https://app.clickup.com/12345/v/`
- ❌ Incorrect: `https://app.clickup.com/12345`

#### Common Issues

**Authentication Failed**:
- Verify token starts with `pk_`
- Check workspace URL format
- Regenerate token if expired

**Tasks Not Found**:
- Verify workspace ID in URL
- Check API token has workspace access
- Ensure tasks exist with proper format (`CU-abc123`)

---

### GitHub Issues Setup

GitHub Issues integrates automatically when GitHub is configured for repository access.

#### Prerequisites
- GitHub personal access token
- Repository access (public or private with token)

#### Configuration

GitHub Issues requires no additional configuration beyond GitHub access:

```yaml
github:
  token: "${GITHUB_TOKEN}"
  organization: "your-org"  # Or specific repositories

analysis:
  ticket_platforms:
    - github  # Enable GitHub Issues tracking
```

Add to your `.env` file:

```bash
GITHUB_TOKEN=ghp_xxxxxxxxxxxx
```

#### Required Permissions

Your GitHub token needs:
- `repo` - Full control of repositories (for private repos)
- `public_repo` - Access public repositories (for public repos only)
- `read:org` - Read org data (for organization mode)

#### Issue Detection

GitHub Issues are detected by these patterns in commit messages:
- `#123` - Issue number
- `GH-123` - Explicit GitHub prefix
- `closes #123` - Closing keywords
- `fixes #456` - Fix keywords

#### Common Issues

**Issues Not Detected**:
- Verify commit messages include `#` or `GH-` prefix
- Check repository access with token
- Ensure issues exist in tracked repositories

**Rate Limiting**:
- Authenticated requests: 5000/hour
- Unauthenticated: 60/hour
- Use token to increase rate limit

---

## Multi-Platform Configuration

You can track tickets from multiple platforms simultaneously:

```yaml
analysis:
  ticket_platforms:
    - jira
    - linear
    - clickup
    - github

pm:
  jira:
    access_user: "${JIRA_ACCESS_USER}"
    access_token: "${JIRA_ACCESS_TOKEN}"
    base_url: "https://company.atlassian.net"

  linear:
    api_key: "${LINEAR_API_KEY}"
    team_ids: ["team_123"]

  clickup:
    api_token: "${CLICKUP_API_TOKEN}"
    workspace_url: "https://app.clickup.com/12345/v/"

github:
  token: "${GITHUB_TOKEN}"
```

### Environment Variables

Store all credentials in `.env`:

```bash
# GitHub
GITHUB_TOKEN=ghp_xxxxxxxxxxxx

# JIRA
JIRA_ACCESS_USER=you@company.com
JIRA_ACCESS_TOKEN=ATATT3xxxxxxxxxxx

# Linear
LINEAR_API_KEY=lin_api_xxxxxxxxxxxx

# ClickUp
CLICKUP_API_TOKEN=pk_xxxxxxxxxxxx
```

### Platform Priority

When a commit references multiple ticket types, GitFlow Analytics tracks all of them:

```
fix: resolve login issue PROJ-123 #456 CU-abc123

Tracked as:
- JIRA: PROJ-123
- GitHub: #456
- ClickUp: CU-abc123
```

### Coverage Analysis

The ticket coverage report shows distribution across platforms:

```markdown
### Platform Usage

- **JIRA**: 45 tickets (60.0%)
- **GitHub Issues**: 20 tickets (26.7%)
- **Linear**: 8 tickets (10.7%)
- **ClickUp**: 2 tickets (2.7%)
```

---

## Best Practices

### Security

1. **Use Environment Variables**: Never commit credentials to configuration files
2. **Rotate Tokens Regularly**: Generate new tokens every 6-12 months
3. **Minimum Permissions**: Grant only read access needed for analysis
4. **Secure .env Files**: Add `.env` to `.gitignore`

### Performance

1. **Team Filtering**: Use Linear team IDs to reduce API calls
2. **Caching**: Enable caching to avoid repeated API requests
3. **Rate Limiting**: Be aware of platform rate limits with large datasets

### Ticket Patterns

1. **Consistent Format**: Encourage team to use standard ticket references
2. **Closing Keywords**: Use `closes`, `fixes`, `resolves` for linking
3. **Multiple Platforms**: Reference all relevant tickets in commits

---

## Troubleshooting

### General Issues

**No Tickets Detected**:
- Check `ticket_platforms` list in `analysis` section
- Verify credentials are correct in `.env`
- Run with `--debug` flag for detailed logs
- Check commit message format matches patterns

**Low Coverage Rate**:
- Review commit message conventions with team
- Check if maintenance commits need tickets
- Some untracked work (style, docs) is acceptable

### Validation

Test your configuration before running full analysis:

```bash
gitflow-analytics -c config.yaml --validate-only
```

This checks:
- YAML syntax
- Environment variable resolution
- Credential validation
- Platform connectivity

### Debug Mode

Enable debug logging for troubleshooting:

```bash
gitflow-analytics -c config.yaml --debug
```

This shows:
- API request details
- Ticket detection patterns
- Platform response codes
- Cache operations

---

## Next Steps

- **[Configuration Guide](../configuration/configuration.md)** - Complete configuration reference
- **[Quick Start](../getting-started/quickstart.md)** - Run your first analysis
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## Support

For platform-specific issues:
- **JIRA**: [Atlassian Support](https://support.atlassian.com)
- **Linear**: [Linear Support](https://linear.app/contact)
- **ClickUp**: [ClickUp Help Center](https://help.clickup.com)
- **GitHub**: [GitHub Support](https://support.github.com)

For GitFlow Analytics issues:
- [GitHub Issues](https://github.com/bobmatnyc/gitflow-analytics/issues)
- [Documentation](../README.md)
