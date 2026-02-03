# Manager's Quick Start Guide

Get actionable insights into your team's productivity in 5 minutes—no technical knowledge required.

## What You'll Get

GitFlow Analytics generates **executive-ready reports** showing:
- ✅ Team productivity metrics (commits, velocity, work distribution)
- ✅ Process health (ticket coverage, commit quality)
- ✅ Work patterns (features vs bugs vs maintenance)
- ✅ Developer profiles (focus, activity, time patterns)
- ✅ Actionable recommendations for improvement

**All from Git history—no JIRA or PM tool required.**

## How It Works (The Delegation Model)

You don't need to install or run anything. Here's the workflow:

### Step 1: Share This Guide with Your Technical Lead (1 minute)

Send your technical lead this message:

> "Please set up GitFlow Analytics for our team using the [Installation Guide](../getting-started/installation.md). Run `gitflow-analytics -c config.yaml --weeks 8` to generate our first report. I'll need the narrative report and CSVs from the `./reports/` directory."

### Step 2: Receive Reports (Team generates in ~5 minutes)

Your team will provide:
- **narrative_report_YYYYMMDD.md** - Executive summary (start here)
- **CSV files** - Data for dashboards (optional)

### Step 3: Read the Narrative Report (5 minutes)

Open the narrative report. It's organized for quick reading:

#### Section 1: Executive Summary (30 seconds)
```markdown
## Executive Summary
- Total Commits: 324
- Active Developers: 8
- Ticket Coverage: 78.4% (above industry benchmark)
- Top Contributor: Sarah Chen with 54 commits
```

**What to look for**:
- ✅ **Ticket Coverage 60-80%** = Healthy process
- ⚠️ **Ticket Coverage 40-60%** = Needs improvement
- 🔴 **Ticket Coverage < 40%** = Process breakdown

#### Section 2: Team Composition (2 minutes)
Shows individual developer profiles:
- Work distribution across projects
- Work style (focused vs multi-project)
- Activity patterns (time of day, velocity)
- Ticket coverage by person

**What to look for**:
- ⚠️ Developers working "Extended Hours" consistently
- ⚠️ Single developers >40% of total work (bus factor risk)
- ⚠️ "Bottom 20%" activity scores (underutilization or blockers)

#### Section 3: Project Activity (1 minute)
Breakdown by repository:
- Commits per project
- Contributors per project
- Classification breakdown (features/bugs/maintenance)

**What to look for**:
- ⚠️ Critical projects with only 1 contributor
- ⚠️ Stale projects (no activity in weeks)
- 🔴 High bug fix percentage (>50% suggests quality issues)

#### Section 4: Development Patterns (1 minute)
Team workflow health:
- Commit message quality
- Branching strategy
- Work distribution balance (Gini coefficient)

**What to look for**:
- ✅ Commit messages with 40+ words (detailed context)
- ✅ Gini coefficient < 0.3 (balanced team)
- ⚠️ Gini > 0.5 (work concentrated on few people)

#### Section 5: Recommendations (1 minute)
Automated suggestions based on patterns:
- Process improvements (e.g., "Increase ticket coverage")
- Workload balancing suggestions
- Quality improvement opportunities

**Action**: Pick 1-2 recommendations to address in next sprint.

### Step 4: Create a Dashboard (Optional, 10 minutes)

If you want visual tracking:
1. Import CSVs to Excel or Google Sheets
2. Create charts for key metrics
3. Track trends week-over-week

See **[Dashboard Guide](dashboard-guide.md)** for step-by-step instructions.

## Key Metrics Explained (1 Minute Reference)

| Metric | What It Means | Good Range | Warning Signs |
|--------|---------------|------------|---------------|
| **Ticket Coverage** | % of commits linked to work items (JIRA, GitHub, etc.) | 60-80% | < 50% |
| **Work Distribution (Gini)** | Team balance (0 = perfect balance, 1 = one person) | < 0.3 | > 0.5 |
| **Classification Mix** | Features vs Bug Fixes vs Maintenance | Varies | >50% bugs |
| **Activity Score** | Developer productivity percentile | Context-dependent | "Bottom 20%" + declining |
| **Commit Quality** | Average words per commit message | 40+ words | < 10 words |

See **[Metrics Reference](metrics-reference.md)** for complete definitions and benchmarks.

## What to Look at First

On your first report, focus on these 3 metrics:

### 1. Ticket Coverage (30 seconds)
**Where**: Executive Summary section
**Question**: "Is our process working?"

- ✅ **60-80%** = Healthy tracking, good process adherence
- ⚠️ **40-60%** = Some untracked work, review with team
- 🔴 **< 40%** = Significant process gap, immediate action needed

**Action if low**: Review [FAQ: What if ticket coverage is low?](faq.md#what-if-ticket-coverage-is-low)

### 2. Work Distribution (1 minute)
**Where**: Development Patterns section (Gini coefficient)
**Question**: "Is work balanced across the team?"

- ✅ **< 0.3** = Work well-distributed, low bus factor risk
- ⚠️ **0.3-0.5** = Some concentration, monitor key contributors
- 🔴 **> 0.5** = Work concentrated on few people, high risk

**Action if high**: Review Team Composition to identify contributors carrying >40% of work.

### 3. Classification Breakdown (1 minute)
**Where**: Commit Classification Analysis section
**Question**: "What type of work is the team doing?"

- ✅ **Features 50-70%** = Building new capabilities
- ⚠️ **Bug Fixes > 50%** = Possible quality issues
- ⚠️ **Maintenance > 40%** = High tech debt burden

**Action**: Use this to inform sprint planning priorities.

## Common First Impressions

### "Ticket coverage is 0% or very low"
**Normal for new setups**. GitFlow Analytics looks for JIRA, GitHub, ClickUp, and Linear ticket references in commit messages.

**Fix**: See [FAQ: What if ticket coverage is low?](faq.md#what-if-ticket-coverage-is-low)

### "Gini coefficient shows high concentration"
**Common in small teams**. With 2-3 developers, it's mathematically hard to be perfectly balanced.

**Concern if**: In larger teams (5+), Gini > 0.5 suggests uneven workload or bus factor risk.

### "My top contributor is working extended hours"
**Worth investigating**. Check the Time Pattern in their developer profile.

**Action**: Review workload distribution and consider rebalancing.

## Using Reports in Different Cadences

### Weekly (Sprint Retrospectives)
**Time**: 5 minutes
**Focus**: Ticket coverage trend, velocity, untracked work
**Action**: Adjust sprint planning based on capacity trends

**Workflow**:
1. Check Ticket Coverage (improving or declining?)
2. Review Untracked Work section (what's missing tickets?)
3. Note velocity trend (stable, growing, or declining?)

### Monthly (Team Health Checks)
**Time**: 10 minutes
**Focus**: Work distribution, developer activity, classification trends
**Action**: Rebalance workload, address process gaps

**Workflow**:
1. Review Gini coefficient (team balance)
2. Check developer Activity Scores (identify outliers)
3. Examine Classification Breakdown (feature vs bug ratio)
4. Review Time Patterns (who's working extended hours?)

### Quarterly (Planning Reviews)
**Time**: 15 minutes
**Focus**: Long-term trends, tech debt, process improvements
**Action**: Set goals for next quarter

**Workflow**:
1. Compare reports from last 3 months
2. Track improvements in ticket coverage or quality metrics
3. Identify persistent patterns (e.g., consistently high bug %)
4. Set measurable goals (e.g., "Increase ticket coverage to 70%")

## Next Steps

Now that you've read your first report:

### Immediate Actions
1. ✅ Identify **1-2 recommendations** to address this sprint
2. ✅ Share key insights with your team (ticket coverage, balance)
3. ✅ Bookmark the [Report Interpretation Guide](interpreting-reports.md) for deeper analysis

### Within 1 Week
1. Review [Metrics Reference](metrics-reference.md) to understand all available metrics
2. Read [FAQ](faq.md) for common questions
3. Set up regular report cadence (weekly, bi-weekly, or monthly)

### Within 1 Month
1. Create a dashboard using [Dashboard Guide](dashboard-guide.md)
2. Track improvement trends (ticket coverage, quality metrics)
3. Share insights in team retrospectives or all-hands

## Getting Help

- **Report interpretation questions**: [Interpreting Reports Guide](interpreting-reports.md)
- **Metric definitions**: [Metrics Reference](metrics-reference.md)
- **Common issues**: [FAQ](faq.md)
- **Technical setup help**: Have your team reference [User Guide](../getting-started/)

---

**Congratulations!** You're now equipped to use GitFlow Analytics for team insights.

**Recommended next read**: [Report Interpretation Guide](interpreting-reports.md) for deeper analysis.
