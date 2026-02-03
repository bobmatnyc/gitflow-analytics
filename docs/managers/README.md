# Manager's Guide to GitFlow Analytics

Welcome! This guide helps engineering managers and team leads understand and use GitFlow Analytics to gain insights into team productivity and development patterns.

## What GitFlow Analytics Does

GitFlow Analytics analyzes your team's Git repositories to provide **actionable productivity insights without requiring JIRA, Linear, or other project management tools**. It gives you visibility into what your team is building, how work is distributed, and where improvements can be made.

**Key Benefits**:
- 📊 Understand team productivity and work patterns
- 🎯 Track process adherence (ticket coverage)
- ⚖️ Monitor work distribution and team balance
- 🔍 Identify untracked work and process gaps
- 📈 Analyze trends over time

## Quick Navigation

### Getting Started (5 Minutes)
- **[Quick Start Guide](quickstart.md)** - Get your first insights in 5 minutes
- **[FAQ](faq.md)** - Common questions answered

### Understanding Your Reports
- **[Report Interpretation Guide](interpreting-reports.md)** - How to read GitFlow Analytics reports
- **[Metrics Reference](metrics-reference.md)** - Plain-language metric definitions and benchmarks
- **[Dashboard Guide](dashboard-guide.md)** - Create Excel/Google Sheets dashboards

## What You'll Receive

When GitFlow Analytics runs, you get:

### 1. Executive Summary Report
A readable markdown file with:
- Key metrics snapshot (commits, developers, ticket coverage)
- Team composition and developer profiles
- Project activity breakdown
- Development patterns analysis
- Actionable recommendations

**Reading time**: 5-10 minutes

### 2. CSV Data Exports
Spreadsheet-ready files for dashboards:
- Weekly metrics with productivity trends
- Developer focus and work distribution
- Activity distribution by developer/project
- Summary statistics and benchmarks

**Use for**: Executive dashboards, quarterly reviews

### 3. Trend Analysis
Week-by-week patterns showing:
- Classification trends (features vs bugs vs maintenance)
- Velocity changes
- Process adherence shifts
- Team health indicators

**Use for**: Sprint retrospectives, monthly reviews

## Who Should Use This Guide?

This documentation is for:
- **Engineering Managers** - Team productivity and health monitoring
- **Team Leads** - Sprint planning and process improvement
- **Directors/VPs** - Cross-team comparisons and organizational trends
- **Product Managers** - Understanding engineering capacity and velocity

## Key Metrics at a Glance

| Metric | What It Shows | Healthy Range |
|--------|---------------|---------------|
| **Ticket Coverage** | % of commits linked to work items | 60-80% |
| **Work Distribution** | Team balance (Gini coefficient) | < 0.3 |
| **Classification Mix** | Features vs bugs vs maintenance | Varies by team |
| **Activity Score** | Developer productivity percentile | Context-dependent |
| **Velocity Trend** | Week-over-week commit patterns | Stable or growing |

See **[Metrics Reference](metrics-reference.md)** for detailed definitions.

## Common Use Cases

### Weekly Sprint Retrospectives
Review ticket coverage, untracked work, and velocity trends to improve sprint planning.

**Read**: [Quick Start Guide](quickstart.md) → [Report Interpretation](interpreting-reports.md)

### Monthly Team Health Checks
Analyze work distribution, developer focus scores, and activity patterns to ensure team balance.

**Read**: [Metrics Reference](metrics-reference.md) → [Dashboard Guide](dashboard-guide.md)

### Quarterly Planning Reviews
Compare classification trends (feature vs bug ratios) and identify tech debt patterns.

**Read**: [Report Interpretation](interpreting-reports.md) → [Dashboard Guide](dashboard-guide.md)

### Process Improvement
Track ticket coverage improvements and commit quality metrics over time.

**Read**: [Quick Start Guide](quickstart.md) → [FAQ](faq.md)

## How GitFlow Analytics Works

**No Setup Required for Managers**: Your technical team runs GitFlow Analytics, and you receive reports. You don't need to install anything or understand Python/Git commands.

**The Process**:
1. Your team runs analysis (5-minute command)
2. Reports are generated in `./reports/` directory
3. You review the narrative report and CSVs
4. Import CSVs to Excel/Sheets for dashboards (optional)

See **[Quick Start Guide](quickstart.md)** for the delegation workflow.

## Getting Help

- **Report interpretation questions**: See [Interpreting Reports](interpreting-reports.md)
- **Metric definitions**: See [Metrics Reference](metrics-reference.md)
- **Common issues**: See [FAQ](faq.md)
- **Technical setup**: Have your team reference [Getting Started](../getting-started/)
- **Dashboard creation**: See [Dashboard Guide](dashboard-guide.md)

## Related Documentation

For technical details and setup:
- [User Guide](../getting-started/) - Installation and configuration
- [CLI Reference](../reference/cli-commands.md) - Command-line options
- [Developer Guide](../developer/) - Contributing to GitFlow Analytics

---

**Next Steps**:
1. Read the [Quick Start Guide](quickstart.md) (5 minutes)
2. Review the [Report Interpretation Guide](interpreting-reports.md) (10 minutes)
3. Create your first dashboard using the [Dashboard Guide](dashboard-guide.md)

**Questions?** See the [FAQ](faq.md) or ask your technical lead to reference the [User Guide](../getting-started/).
