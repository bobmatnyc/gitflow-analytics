# Understanding GitFlow Analytics Reports

A comprehensive guide to reading and interpreting GitFlow Analytics reports, with plain-language explanations of every section and metric.

## Report Types Overview

GitFlow Analytics generates multiple reports, each serving different purposes:

| Report | Purpose | Read Time | Best For |
|--------|---------|-----------|----------|
| **Narrative Report** | Executive overview with insights | 5-10 min | Weekly reviews, stakeholder updates |
| **Qualitative Report** | Trend analysis and classification insights | 3-5 min | Monthly reviews, pattern detection |
| **CSV Exports** | Data for dashboards and analysis | N/A | Executive dashboards, quarterly reporting |

**Start with the Narrative Report**—it's designed for quick reading and actionable insights.

## Narrative Report Walkthrough

The narrative report is organized for progressive reading: start at the top for quick insights, drill deeper into sections as needed.

### Section 1: Executive Summary

**Purpose**: 30-second snapshot of team health and productivity.

**Example**:
```markdown
## Executive Summary
- Total Commits: 324 commits across 4 projects
- Active Developers: 8 team members
- Ticket Coverage: 78.4% (above industry benchmark)
- Top Contributor: Sarah Chen with 54 commits
- Team Activity: High activity (avg 40.5 commits/developer)
```

#### What Each Metric Means

**Total Commits**
- **Definition**: Number of code changes submitted in the analysis period
- **Not a quality metric**: More commits ≠ better; shows volume of activity
- **Context matters**: 100 commits from 2 developers vs 10 developers tells different stories

**Active Developers**
- **Definition**: Number of team members who contributed code
- **Use for**: Team size tracking, capacity planning
- **Watch for**: Declining contributor count (attrition, reassignments)

**Ticket Coverage**
- **Definition**: Percentage of commits linked to work items (JIRA, GitHub, ClickUp, Linear)
- **Industry benchmark**: 60-80% is healthy
- **Interpretation**:
  - ✅ **60-80%**: Strong process adherence, work is tracked
  - ⚠️ **40-60%**: Some untracked work, review processes
  - 🔴 **< 40%**: Significant process gap, most work untracked

**Why it matters**: Low ticket coverage means you can't track what's being built against planned work.

**Top Contributor**
- **Definition**: Developer with most commits in period
- **Not a performance metric**: Volume ≠ value
- **Watch for**: Same person consistently >40% of total work (bus factor risk)

**Team Activity**
- **Definition**: Average commits per developer
- **Contextual**: "High" vs "Low" depends on team norms
- **Use for**: Velocity trends (compare week-to-week)

#### Red Flags in Executive Summary

- 🔴 **Ticket Coverage < 40%**: Process breakdown, work not tracked
- 🔴 **Single contributor > 50% of commits**: Extreme bus factor risk
- 🔴 **Active Developers declining**: Attrition or team transition
- ⚠️ **Ticket Coverage declining trend**: Process degradation

### Section 2: Team Composition

**Purpose**: Individual developer profiles with work patterns, focus, and activity.

**Example**:
```markdown
### Developer Profiles

**Sarah Chen**
- Commits: Features: 32 (59%), Bug Fixes: 18 (33%), Maintenance: 4 (7%)
- Ticket Coverage: 85.2%
- Activity Score: 82.3/100 (high activity, Top 10%)
- Projects: Frontend (65%), API (25%), Infrastructure (10%)
- Work Style: Focused contributor
- Active Pattern: Midday developer
```

#### Understanding Developer Profiles

**Classification Breakdown**
- **Features**: New functionality development
- **Bug Fixes**: Error corrections and patches
- **Maintenance**: Refactoring, tech debt, dependency updates
- **Healthy mix**: Varies by role (frontend = more features, platform = more maintenance)

**Watch for**:
- 🔴 **> 60% Bug Fixes**: Developer stuck firefighting, not building new features
- ⚠️ **0% Maintenance**: No tech debt work, future risk

**Ticket Coverage (Individual)**
- **Interpretation**: Same as team metric, but per developer
- **Action if low**: Review with individual about commit message practices

**Activity Score**
- **Definition**: Developer's commit volume percentile (0-100)
- **Context-dependent**: Top 10% in a high-velocity team ≠ Top 10% in low-velocity team
- **Use for**: Identifying outliers (underutilization or burnout)

**Watch for**:
- ⚠️ **Bottom 20% + declining**: Possible blockers, onboarding issues, or disengagement
- ⚠️ **Top 10% + "Extended Hours" pattern**: Possible burnout risk

**Projects Distribution**
- **Definition**: % of developer's work across repositories
- **Work Style Classifications**:
  - **Highly Focused**: > 80% on one project (specialist)
  - **Focused**: 60-80% on one project (primary assignment)
  - **Multi-project**: < 60% on one project (generalist or scattered)

**Context matters**: "Highly Focused" on a growing project = good. On a declining project = risk.

**Active Pattern (Time of Day)**
- **Definition**: When developer typically commits
- **Patterns**:
  - **Midday developer**: 9 AM - 5 PM
  - **Afternoon developer**: 12 PM - 8 PM
  - **Extended Hours**: Outside typical working hours

**Watch for**:
- 🔴 **Extended Hours consistently**: Possible burnout, unrealistic deadlines
- ⚠️ **Sudden pattern change**: May indicate team dynamics shift

#### Red Flags in Team Composition

- 🔴 **Developer with > 60% Bug Fixes**: Firefighting mode
- 🔴 **"Extended Hours" pattern across multiple developers**: Workload issue
- 🔴 **Bottom 20% activity + declining trend**: Blockers or disengagement
- ⚠️ **Single developer "Highly Focused" on critical project**: Bus factor risk

### Section 3: Project Activity

**Purpose**: Repository-level breakdown of work distribution and patterns.

**Example**:
```markdown
### Activity by Project

**Frontend Repository**
- Commits: 156 (48.1% of total)
- Lines Changed: 12,450
- Contributors: Sarah Chen (42%), Mike Johnson (28%), Lisa Park (18%), Others (12%)
- Classifications: Features: 78 (50%), Bug Fixes: 62 (40%), Maintenance: 16 (10%)
```

#### Interpreting Project Metrics

**Commits and Lines Changed**
- **Not quality metrics**: More code ≠ better
- **Use for**: Understanding project activity distribution
- **Context**: Large lines changed + low commits = big refactors or merges

**Contributors Distribution**
- **Healthy**: Multiple contributors, no single person > 40%
- **Watch for**:
  - 🔴 **Critical project with 1 contributor**: Bus factor risk
  - ⚠️ **Stale projects** (no activity in weeks): Candidates for sunsetting

**Classification Breakdown (Project-Level)**
- **Features %**: New functionality development
- **Bug Fixes %**: Corrections and patches
- **Maintenance %**: Refactoring, tech debt, updates

**Healthy ranges vary by project type**:
- **New projects**: 70-80% Features, 10-20% Bug Fixes, 10-20% Maintenance
- **Mature projects**: 40-50% Features, 30-40% Bug Fixes, 20-30% Maintenance
- **Platform/Infrastructure**: 30-40% Features, 20-30% Bug Fixes, 30-50% Maintenance

**Warning signs**:
- 🔴 **> 60% Bug Fixes**: Quality crisis, fire-fighting mode
- ⚠️ **0-5% Maintenance**: Tech debt accumulating
- ⚠️ **Features declining over time**: Project stagnating

### Section 4: Development Patterns

**Purpose**: Team workflow health indicators and process metrics.

**Example**:
```markdown
## Development Patterns

**Timing**:
- Peak commit hour: 2:00 PM (Indicates team working hours)

**Quality**:
- Commit message quality: Detailed (Average 45.2 words per message)

**Process**:
- Ticket tracking adherence: Strong tracking (78.4% commits have ticket references)

**Team**:
- Team size: Medium team (8 active developers)
- Work distribution: Balanced (Gini coefficient: 0.24)

**Workflow**:
- Branching strategy: Frequent branching (34.5% merge commits)

**Developer Focus**: Average focus score of 68.3% indicates healthy specialization
```

#### Key Patterns Explained

**Commit Message Quality**
- **Metric**: Average words per commit message
- **Interpretation**:
  - ✅ **40+ words**: Detailed context, future maintainability
  - ⚠️ **10-40 words**: Minimal info, harder to understand changes later
  - 🔴 **< 10 words**: Poor documentation (e.g., "fix", "update")

**Why it matters**: Detailed commit messages help with debugging, onboarding, and understanding code history.

**Ticket Tracking Adherence**
- **Same as Ticket Coverage**: % of commits with ticket references
- **Action if low**: Review commit message standards, consider pre-commit hooks

**Work Distribution (Gini Coefficient)**
- **Definition**: Statistical measure of distribution equality
- **Range**: 0 (perfect balance) to 1 (one person does everything)
- **Interpretation**:
  - ✅ **< 0.3**: Well-balanced team, low bus factor risk
  - ⚠️ **0.3-0.5**: Some concentration, monitor key contributors
  - 🔴 **> 0.5**: Highly concentrated, high risk if key people leave

**Real-world context**:
- **Small teams (2-3)**: Gini 0.3-0.4 is normal (mathematical limit)
- **Medium teams (5-8)**: Gini < 0.3 is achievable and healthy
- **Large teams (10+)**: Gini < 0.25 indicates excellent balance

**Branching Strategy**
- **Metric**: Percentage of merge commits
- **Interpretation**:
  - **> 30% merge commits**: Frequent feature branches (healthy Git workflow)
  - **< 10% merge commits**: Direct-to-main commits (risky for teams)
  - **0% merge commits**: No branching (acceptable for solo dev, risky for teams)

**Developer Focus Score (Team Average)**
- **Definition**: Average project concentration across all developers
- **Interpretation**:
  - ✅ **60-80%**: Healthy specialization, developers have primary projects
  - ⚠️ **< 50%**: Scattered work, context-switching overhead
  - ⚠️ **> 90%**: Over-specialized, bus factor risk

### Section 5: Commit Classification Analysis

**Purpose**: ML-powered categorization of work types with confidence scores.

**Example**:
```markdown
## Commit Classification Analysis

The team's commit patterns reveal the following automated classification insights:

- **Features**: 158 commits (48.8%) - avg confidence 87.3%
- **Bug Fixes**: 112 commits (34.6%) - avg confidence 91.2%
- **Maintenance**: 54 commits (16.7%) - avg confidence 78.9%

**Confidence**: 86.2% of commits classified with >80% confidence
```

#### Understanding ML Classifications

**Classification Types**
- **Features**: New functionality, enhancements, capabilities
- **Bug Fixes**: Error corrections, patches, fixes
- **Refactor**: Code restructuring without behavior change
- **Maintenance**: Dependency updates, config changes, tech debt
- **Documentation**: README updates, comments, docs
- **Testing**: Test additions, test refactoring
- **Build/CI**: Pipeline changes, build config, deployments
- **Performance**: Optimization work

**Confidence Scores**
- **Definition**: ML model's certainty about classification (0-100%)
- **Interpretation**:
  - ✅ **> 80% confidence**: Highly reliable classification
  - ⚠️ **60-80% confidence**: Moderately reliable, manual review recommended
  - 🔴 **< 60% confidence**: Uncertain, needs manual review

**Team average > 85% confidence**: ML categorization is highly accurate for your commit patterns.

**Healthy Classification Mix**
Varies by team and project stage:

**Early-stage project**:
- Features: 60-80%
- Bug Fixes: 10-20%
- Maintenance: 10-20%

**Mature product**:
- Features: 40-50%
- Bug Fixes: 30-40%
- Maintenance: 20-30%

**Platform/Infrastructure**:
- Features: 30-40%
- Bug Fixes: 20-30%
- Maintenance: 30-50%

### Section 6: Issue Tracking

**Purpose**: Analyze process adherence and identify untracked work.

**Example**:
```markdown
## Issue Tracking

### Coverage Analysis
- **Overall Coverage**: 78.4% (254 of 324 commits tracked)
- **Platform Distribution**:
  - JIRA: 148 commits (45.7%)
  - GitHub Issues: 82 commits (25.3%)
  - Linear: 24 commits (7.4%)

### Untracked Work Analysis
**70 commits (21.6%) without ticket references**:
- Features: 32 commits (45.7%) - New functionality without tickets
- Bug Fixes: 24 commits (34.3%) - Fixes without issue tracking
- Maintenance: 14 commits (20.0%) - Tech debt work
```

#### Interpreting Coverage Metrics

**Overall Coverage**
- **78.4% = Healthy**: Most work is tracked
- **Target**: 60-80% (industry benchmark)
- **100% not realistic**: Small fixes, dependency updates often untracked

**Platform Distribution**
- **Shows**: Which PM tools the team uses
- **Use for**: Understanding tool adoption, integration priorities

**Untracked Work Analysis**
- **Most valuable section**: Shows what's happening outside planned work
- **Watch for**:
  - 🔴 **Untracked Features**: Major work not aligned with planning
  - ⚠️ **Untracked Bug Fixes**: Firefighting not captured in sprint planning
  - ✅ **Untracked Maintenance**: Often acceptable (dependency updates, refactoring)

**Action**: Review untracked Features and Bug Fixes with team. Should they have tickets?

### Section 7: Recommendations

**Purpose**: Automated suggestions based on detected patterns.

**Example**:
```markdown
## Recommendations

Based on the analysis, consider these improvements:

1. **Improve Ticket Discipline**: 21.6% of commits lack ticket references. Target 80%+ coverage through:
   - Pre-commit hooks validating ticket references
   - Sprint planning review of untracked work
   - Developer education on commit message standards

2. **Balance Workload**: Sarah Chen carries 42% of Frontend work. Consider:
   - Knowledge sharing sessions
   - Pair programming to distribute expertise
   - Onboarding additional frontend developers

3. **Address Bug Fix Concentration**: 34.6% of commits are bug fixes. Investigate:
   - Root cause analysis of frequent issues
   - Additional testing coverage
   - Refactoring high-churn areas
```

#### Acting on Recommendations

**Priority Recommendations**
1. **Ticket Coverage < 60%**: Immediate action, process breakdown
2. **Gini > 0.5**: High risk, address workload imbalance
3. **> 50% Bug Fixes**: Quality crisis, investigate root causes

**Pick 1-2 Per Sprint**
- Don't try to fix everything at once
- Focus on highest-impact improvements
- Track progress in next report

## Qualitative Report (Trend Analysis)

The qualitative report focuses on **week-by-week trends** and **classification insights**.

**Key Sections**:

### Executive Summary
Similar to narrative report, but emphasizes trends:
- Classification % changes week-over-week
- Velocity trends (commits per week)
- Ticket coverage trends

### Team Analysis (Per Developer)
Weekly breakdown showing:
- Classification mix changes
- Activity level changes
- Ticket coverage trends

**Use for**: Identifying patterns (e.g., "Bug Fixes increasing each week")

### Weekly Trends
Line-by-line weekly progression:
```markdown
Week 22 (Jul 28-03): Bug Fixes 33%, Features 44%, Maintenance 22% | 18/18 tickets (100%)
Week 23 (Aug 04-10): Bug Fixes 65% (+32%), Features 35% (-9%) | 20/20 tickets (100%)
Week 24 (Aug 11-17): Bug Fixes 100% (+35%), Features 0% (-35%) | 1/1 tickets (100%)
```

**Watch for**:
- ⚠️ **Bug Fixes increasing**: Quality degradation
- ⚠️ **Features decreasing**: Less new development
- ⚠️ **Ticket Coverage declining**: Process breakdown

## CSV Reports (For Dashboards)

CSV exports provide **raw data** for custom analysis and dashboards.

### Key CSV Files

**weekly_metrics_YYYYMMDD.csv**
- Columns: Week, Commits, Developers, Lines Changed, Ticket Coverage, Story Points
- **Use for**: Velocity trends, week-over-week tracking

**developer_focus_YYYYMMDD.csv**
- Columns: Developer, Total Commits, Projects, Focus Score, Work Style, Activity Score
- **Use for**: Team balance analysis, focus monitoring

**activity_distribution_YYYYMMDD.csv**
- Columns: Developer, Project, Commits, Lines Changed, % of Developer Work, % of Project Work
- **Use for**: Heatmaps showing developer × project matrix

**summary_YYYYMMDD.csv**
- Columns: Total Commits, Active Developers, Ticket Coverage, Gini Coefficient, etc.
- **Use for**: Executive dashboards, KPI tracking

See **[Dashboard Guide](dashboard-guide.md)** for import and visualization instructions.

## Common Patterns and What They Mean

### Healthy Patterns ✅

| Pattern | What It Shows | Action |
|---------|---------------|--------|
| Ticket Coverage 60-80% | Strong process adherence | Maintain standards |
| Gini < 0.3 | Balanced team workload | Monitor for changes |
| 40-50% Features | Balanced new development | Healthy product evolution |
| Commit messages 40+ words | Detailed documentation | Share best practices |
| Multiple active projects | Diversified portfolio | No action needed |

### Warning Patterns ⚠️

| Pattern | What It Shows | Action |
|---------|---------------|--------|
| Ticket Coverage 40-60% | Some untracked work | Review processes, improve discipline |
| Gini 0.3-0.5 | Moderate concentration | Monitor key contributors, knowledge sharing |
| Bug Fixes 40-50% | Quality issues emerging | Investigate root causes, add testing |
| Commit messages 10-40 words | Minimal documentation | Update commit message standards |
| Extended Hours pattern | Potential overwork | Review workload with individual |

### Critical Patterns 🔴

| Pattern | What It Shows | Action |
|---------|---------------|--------|
| Ticket Coverage < 40% | Process breakdown | Immediate process review |
| Gini > 0.5 | Severe workload imbalance | Rebalance work, reduce bus factor risk |
| Bug Fixes > 60% | Quality crisis | Root cause analysis, pause features |
| Commit messages < 10 words | Poor documentation | Mandatory commit message training |
| Single contributor > 60% | Extreme bus factor risk | Cross-training, knowledge transfer |

## Taking Action on Reports

### Weekly Sprint Retrospectives (5 minutes)

**Workflow**:
1. Check **Ticket Coverage trend** (improving or declining?)
2. Review **Untracked Work** section (what needs tickets?)
3. Note **Velocity trend** (stable, growing, or declining?)
4. Identify **1-2 quick wins** from Recommendations

**Output**: Sprint planning adjustments, process improvements

### Monthly Team Health Checks (10 minutes)

**Workflow**:
1. Review **Gini coefficient** (team balance over time)
2. Check **Developer Activity Scores** (outliers = underutilization or burnout?)
3. Examine **Classification Breakdown** (feature vs bug ratio trends)
4. Review **Time Patterns** (who's working extended hours?)

**Output**: Workload rebalancing, wellness check-ins

### Quarterly Planning Reviews (15 minutes)

**Workflow**:
1. Compare reports from last **3 months** (trend lines)
2. Track **improvements** in ticket coverage, quality metrics
3. Identify **persistent patterns** (e.g., consistently high bug %)
4. Set **measurable goals** for next quarter

**Output**: OKRs, process improvement initiatives

## Next Steps

- **Create a dashboard**: See [Dashboard Guide](dashboard-guide.md)
- **Understand all metrics**: See [Metrics Reference](metrics-reference.md)
- **Answer common questions**: See [FAQ](faq.md)

---

**Need help?** Reference the [Metrics Reference](metrics-reference.md) for detailed definitions and benchmarks.
