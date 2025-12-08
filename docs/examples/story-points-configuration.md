# Story Points Configuration Example

This example demonstrates how to configure GitFlow Analytics to track story points from both JIRA tickets and commit messages.

## Prerequisites

- JIRA account with API access
- JIRA API token generated
- Repository with commits that reference JIRA tickets

## Problem Solved

This configuration addresses the common issue where story points tracking is not properly configured, resulting in missing story points data in reports.

## Complete Configuration Example

```yaml
# config.yaml - Story Points Tracking Configuration
version: "1.0"

repositories:
  - name: "example-project"
    path: "/path/to/repository"
    project_key: "PROJ"

# JIRA integration for story points extraction
jira:
  access_user: "${JIRA_ACCESS_USER}"
  access_token: "${JIRA_ACCESS_TOKEN}"
  base_url: "https://company.atlassian.net"

jira_integration:
  enabled: true
  fetch_story_points: true
  story_point_fields:
    - "customfield_10063"  # Story Points (primary field)
    - "customfield_10016"  # Story point estimate (backup field)
    - "Story Points"       # Field name fallback
    - "timeestimate"       # Remaining Estimate
    - "timeoriginalestimate"  # Original estimate

analysis:
  weeks: 4
  
  # Story point extraction patterns (text-based patterns for commit messages)
  story_point_patterns:
    - "(?:story\\s*points?|sp|pts?)\\s*[:=]\\s*(\\d+)"  # SP: 5, story points = 3
    - "\\[(\\d+)\\s*(?:sp|pts?|points?)\\]"  # [3sp], [5 pts], [3 points]
    - "\\((\\d+)\\s*(?:story\\s*points?|sp|pts?)\\)"  # (5 story points), (3 sp)
    - "#(\\d+)sp"  # #3sp
    - "estimate:\\s*(\\d+)"  # estimate: 5
    - "\\bSP(\\d+)\\b"  # SP5, SP13
    - "points?:\\s*(\\d+)"  # points: 8

output:
  directory: "reports"
  formats:
    - csv
    - json
    - markdown
```

## Environment Variables Setup

Create a `.env` file (never commit this):

```bash
# .env file
JIRA_ACCESS_USER=your-email@company.com
JIRA_ACCESS_TOKEN=your-jira-api-token
```

Or set environment variables directly:

```bash
export JIRA_ACCESS_USER="your-email@company.com"
export JIRA_ACCESS_TOKEN="your-jira-api-token"
```

## How Story Points Tracking Works

### Two-Phase Approach
1. **Text-based extraction**: Extracts story points from commit messages using regex patterns
2. **JIRA integration**: Fetches story points from JIRA tickets referenced in commits

### Story Point Sources (in priority order)
1. **JIRA tickets** - Most authoritative source
   - Uses configured `story_point_fields` to find story points in JIRA
   - Matches ticket references in commit messages (e.g., "PROJ-1030")
   
2. **Commit messages** - Fallback for text-based extraction
   - Uses `story_point_patterns` to extract from commit text
   - Useful when JIRA integration is unavailable or tickets don't have story points

## Pattern Examples

The configuration includes patterns to match various story point formats:

```
SP: 5                    # Matches: story points: 5
[3sp]                    # Matches: [3sp], [5 pts], [3 points]
(5 story points)         # Matches: (5 story points), (3 sp)
#3sp                     # Matches: #3sp
estimate: 5              # Matches: estimate: 5
SP5                      # Matches: SP5, SP13
points: 8                # Matches: points: 8
```

## Usage Instructions

### 1. Run Analysis
```bash
gitflow-analytics analyze --config config.yaml --weeks 4
```

### 2. Check Reports
Story points will appear in:
- **Summary reports**: Total story points per developer/project
- **Developer reports**: Story points breakdown by developer
- **Weekly metrics**: Story points trends over time
- **Comprehensive export**: Full story points data in JSON format

## Verification

Test your configuration with a simple script:

```python
# test_story_points.py
import yaml
from gitflow_analytics.config import load_config

# Load and validate configuration
config = load_config('config.yaml')

# Check JIRA integration
jira_config = config.get('jira_integration', {})
print(f"JIRA integration enabled: {jira_config.get('enabled', False)}")
print(f"Story points fields: {jira_config.get('story_point_fields', [])}")

# Check story point patterns
patterns = config.get('analysis', {}).get('story_point_patterns', [])
print(f"Story point patterns configured: {len(patterns)}")

print("✅ Configuration loaded successfully")
```

## Troubleshooting

### Common Issues

1. **No story points in reports**
   - Verify JIRA credentials are set correctly
   - Check that `story_point_fields` match your JIRA configuration
   - Ensure commits reference JIRA tickets (e.g., "PROJ-123")

2. **JIRA authentication errors**
   - Verify API token is valid and not expired
   - Check JIRA base URL is correct
   - Ensure user has access to the projects

3. **Pattern matching issues**
   - Test patterns with actual commit messages
   - Add custom patterns for your team's conventions
   - Check regex syntax is valid

## See Also

- [JIRA Integration Guide](../guides/jira-integration.md)
- [Configuration Reference](../reference/configuration-schema.md)
- [PM Platform Setup](../guides/pm-platform-setup.md)
