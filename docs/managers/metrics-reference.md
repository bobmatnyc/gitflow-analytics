# Metrics Reference Guide

A plain-language glossary of all GitFlow Analytics metrics with definitions, healthy ranges, benchmarks, and interpretation guidance.

## Quick Reference Table

| Metric | Definition | Good Range | Warning Signs | Critical Threshold |
|--------|-----------|------------|---------------|-------------------|
| **Ticket Coverage** | % commits with ticket refs | 60-80% | 40-60% | < 40% |
| **Gini Coefficient** | Work distribution balance | < 0.3 | 0.3-0.5 | > 0.5 |
| **Activity Score** | Developer productivity percentile | Context-dependent | Bottom 20% + declining | N/A |
| **Focus Score** | % of work on primary project | 60-80% | < 50% or > 90% | N/A |
| **Commit Quality** | Avg words per message | 40+ words | 10-40 words | < 10 words |
| **Feature %** | % of commits building new capabilities | 40-70% | Declining trend | < 20% |
| **Bug Fix %** | % of commits fixing errors | 20-40% | 40-60% | > 60% |
| **Maintenance %** | % of commits on tech debt/refactor | 10-30% | < 5% or > 40% | 0% |
| **Ticketing Score** | Weighted sum of issue/comment/page events across platforms | Context-dependent | Consistently 0.0 with integrations configured | N/A |

## Team Health Metrics

### Ticket Coverage

**Definition**: Percentage of commits that reference a ticket ID from project management tools (JIRA, GitHub Issues, ClickUp, Linear).

**Calculation**: `(Commits with ticket references / Total commits) × 100`

**What It Measures**: Process adherence—how well the team links code changes to planned work.

**Industry Benchmarks**:
- ✅ **60-80%**: Healthy tracking, most work is planned
- ⚠️ **40-60%**: Moderate gaps, some untracked work
- 🔴 **< 40%**: Significant process breakdown, most work untracked
- 🔵 **80-100%**: Excellent discipline (rare, often over-constrained)

**Context Matters**:
- **Early-stage startups**: 40-60% acceptable (rapid iteration, less planning)
- **Enterprise teams**: 70-80% expected (formal processes, compliance)
- **Open source**: 30-50% common (community contributions)

**What "Good" Looks Like**:
- Ticket coverage **stable** or **improving** over time
- Untracked work is primarily **Maintenance** (dependency updates, refactoring)
- Untracked **Features** and **Bug Fixes** are minimal

**Red Flags**:
- 🔴 **Declining trend**: Process degradation, team not following standards
- 🔴 **Untracked Features**: Major work not aligned with planning
- 🔴 **< 40% overall**: Unable to track what's being built

**Action When Low**:
1. Review untracked work analysis (what's missing tickets?)
2. Update commit message standards
3. Consider pre-commit hooks to enforce ticket references
4. Educate team on importance of linking commits to work items

**Related Metrics**: Untracked Work Analysis, Platform Distribution

---

### Gini Coefficient (Work Distribution)

**Definition**: Statistical measure of work distribution equality across team members. Named after Italian statistician Corrado Gini.

**Scale**: 0 (perfect equality) to 1 (one person does all work)

**What It Measures**: Work concentration risk—how balanced is the team's workload?

**Interpretation**:
- ✅ **0.0-0.3**: Well-balanced team
  - Low bus factor risk
  - Knowledge distributed
  - Sustainable workload
- ⚠️ **0.3-0.5**: Moderate concentration
  - Some key contributors
  - Monitor for knowledge silos
  - Medium bus factor risk
- 🔴 **0.5-1.0**: High concentration
  - Work dominated by few people
  - High bus factor risk
  - Unsustainable for key contributors

**Team Size Context**:

| Team Size | Excellent (Green) | Acceptable (Yellow) | Concerning (Red) |
|-----------|------------------|---------------------|------------------|
| 2-3 developers | < 0.4 | 0.4-0.5 | > 0.5 |
| 5-8 developers | < 0.3 | 0.3-0.45 | > 0.45 |
| 10+ developers | < 0.25 | 0.25-0.4 | > 0.4 |

**Why Team Size Matters**: Smaller teams mathematically can't achieve perfect balance (Gini = 0).

**What "Good" Looks Like**:
- No single developer contributes > 30% of total commits
- Top 3 contributors account for < 60% of work
- Gini coefficient **stable** or **decreasing** over time

**Red Flags**:
- 🔴 **Single developer > 50%**: Extreme bus factor risk
- 🔴 **Gini increasing**: Work becoming more concentrated
- 🔴 **Top contributor working Extended Hours**: Unsustainable load

**Action When High**:
1. Identify top contributors (who's carrying the team?)
2. Review project assignments (can work be redistributed?)
3. Knowledge sharing sessions (cross-training)
4. Onboard additional developers to high-concentration areas
5. Consider pairing junior developers with top contributors

**Related Metrics**: Developer Activity Score, Focus Score, Time Patterns

---

### Developer Activity Score

**Definition**: Percentile ranking of a developer's commit volume relative to the team (0-100 scale).

**Calculation**: Developer's commits compared to team distribution, expressed as percentile.

**What It Measures**: Relative productivity—how active is this developer compared to teammates?

**Interpretation**:
- **90-100 (Top 10%)**: Highest commit volume
- **70-90 (Top 30%)**: Above-average activity
- **30-70 (Middle 40%)**: Average activity
- **10-30 (Bottom 30%)**: Below-average activity
- **0-10 (Bottom 10%)**: Lowest commit volume

**IMPORTANT**: Activity Score is **context-dependent**, not an absolute performance metric.

**What It Does NOT Mean**:
- ❌ Higher score = better developer
- ❌ Lower score = underperformer
- ❌ Activity = impact or value delivered

**What It DOES Mean**:
- ✅ Relative commit volume within the team
- ✅ Useful for identifying outliers (very high or very low)
- ✅ Starting point for conversations, not conclusions

**When to Investigate**:

| Pattern | Possible Causes | Action |
|---------|----------------|--------|
| **Bottom 20% + Stable** | Different role (architect, lead), Part-time, Onboarding | Context check—is this expected? |
| **Bottom 20% + Declining** | Blockers, Disengagement, Reassignment, Leaving team | 1-on-1 check-in |
| **Top 10% + Extended Hours** | Burnout risk, Understaffing, Unrealistic deadlines | Workload review |
| **Top 10% + Stable** | High performer, Specialist role, Large project ownership | Monitor for sustainability |

**Context Examples**:

**Example 1: Bottom 20% is Normal**
- **Scenario**: Senior architect with 5 commits/week in team averaging 20 commits/week
- **Context**: Architect spends time on design, mentoring, code reviews (not captured in commit volume)
- **Action**: No action needed—role-appropriate activity

**Example 2: Bottom 20% Needs Attention**
- **Scenario**: Mid-level developer with 5 commits/week in team averaging 20 commits/week, declining trend
- **Context**: Developer reports blockers, unclear requirements, waiting on dependencies
- **Action**: 1-on-1 to address blockers, provide clarity, unblock

**What "Good" Looks Like**:
- Activity scores **distributed** across the team (not everyone at Top 10% or Bottom 20%)
- Outliers have **contextual reasons** (role, project stage, part-time)
- Scores are **stable** over time (no sudden drops)

**Red Flags**:
- 🔴 **Bottom 20% + declining + no clear reason**: Investigate blockers or disengagement
- 🔴 **Top 10% + Extended Hours + increasing Gini**: Burnout risk + team imbalance
- ⚠️ **Entire team in Bottom 50%**: Low overall velocity (compare to previous periods)

**Related Metrics**: Gini Coefficient, Time Patterns, Focus Score

---

## Productivity Metrics

### Total Commits

**Definition**: Number of code changes (commits) submitted during the analysis period.

**What It Measures**: Volume of development activity.

**NOT a Quality Metric**:
- ❌ More commits ≠ better
- ❌ Fewer commits ≠ worse
- ✅ Shows activity level, not value delivered

**Context Matters**:
- **100 commits from 1 developer**: Possible micro-commits or rapid iteration
- **100 commits from 10 developers**: Lower activity, possibly larger changes per commit
- **Declining commits + stable lines changed**: Consolidating smaller commits (healthy refactoring)

**Use For**:
- Tracking **velocity trends** week-over-week
- Understanding **team activity patterns**
- Capacity planning (historical commit rates)

**What "Good" Looks Like**:
- Commits **stable** or **growing** (indicates consistent velocity)
- Commit rate **aligns** with sprint capacity
- No sudden spikes or drops (indicates unstable workflow)

**Red Flags**:
- 🔴 **Sudden drop + same team size**: Blockers, process issues, or shifting priorities
- ⚠️ **Spike followed by silence**: Crunch → recovery cycle (unsustainable)

**Related Metrics**: Lines Changed, Activity Score, Velocity Trend

---

### Lines Changed

**Definition**: Total lines of code added, modified, or deleted during the analysis period.

**What It Measures**: Code churn—how much code is being touched.

**NOT a Quality Metric**:
- ❌ More lines ≠ more value
- ❌ Large changes aren't inherently good or bad
- ✅ Shows scale of changes, not impact

**Context Matters**:
- **100,000 lines + 5 commits**: Large refactor, dependency update, or generated code
- **100,000 lines + 500 commits**: High-churn project, rapid iteration
- **1,000 lines + 200 commits**: Micro-commits, small incremental changes

**Use For**:
- Identifying **large refactors** (high lines, low commits)
- Understanding **project volatility** (code churn trends)
- Correlation with **bug rates** (high churn may correlate with bugs)

**What "Good" Looks Like**:
- Lines changed **proportional** to commits (consistent change size)
- No extreme outliers (e.g., 1 commit with 100K lines)
- Trend aligns with **sprint goals** (feature sprints = more lines, stabilization sprints = fewer)

**Red Flags**:
- 🔴 **High churn + high bug fix %**: Quality issues from rapid changes
- ⚠️ **Single commit >> 10K lines**: Review for accidental inclusions (node_modules, generated files)

**Related Metrics**: Average Commit Size, Change Scope

---

### Story Points Delivered

**Definition**: Total story points extracted from commit messages or PR descriptions (if configured).

**What It Measures**: Estimated effort delivered (if using story point estimation).

**Requires Configuration**: Team must include story point references in commits (e.g., "SP-5", "points:3").

**Use For**:
- Tracking **sprint velocity** (points delivered per sprint)
- Comparing **estimated vs actual** effort
- Capacity planning for future sprints

**What "Good" Looks Like**:
- Story points **increasing** or **stable** over time (consistent velocity)
- Correlation between **points and commit volume** (sanity check)
- Points delivered **aligns** with sprint planning

**Limitations**:
- **Not all teams use story points**: Metric may be 0 or N/A
- **Accuracy depends on commit discipline**: Requires team to reference points consistently

**Related Metrics**: Ticket Coverage, Velocity Trend

---

## Work Pattern Metrics

### Focus Score

**Definition**: Percentage of a developer's work concentrated on their primary project (the project with most commits).

**Calculation**: `(Commits on primary project / Total commits) × 100`

**What It Measures**: Project concentration—how specialized vs generalized is the developer?

**Interpretation**:
- **80-100% (Highly Focused)**: Specialist, deep expertise on one project
- **60-80% (Focused)**: Primary project + some cross-team work
- **40-60% (Balanced)**: Multi-project contributor
- **0-40% (Scattered)**: Generalist, many small contributions across projects

**What "Good" Looks Like**:
- **Team average 60-80%**: Healthy specialization
- **Mix of focus levels**: Some specialists, some generalists
- **Focus aligns with role**: Frontend specialists highly focused on frontend projects

**Context Matters**:

| Scenario | Focus Score | Interpretation |
|----------|-------------|----------------|
| **Frontend specialist on growing project** | 85% | ✅ Healthy specialization |
| **Frontend specialist on declining project** | 85% | ⚠️ Risk—need to expand skills or reassign |
| **Tech lead across 5 projects** | 35% | ✅ Expected for leadership role |
| **Junior developer across 5 projects** | 35% | ⚠️ Possible context-switching overhead |

**Red Flags**:
- 🔴 **Highly Focused (>80%) on declining/stale project**: Bus factor risk + stale skills
- 🔴 **Scattered (<40%) + Bottom 20% Activity**: Spread too thin, ineffective
- ⚠️ **Team average >90%**: Over-specialized team, knowledge silos

**Action When Concerning**:
1. Review **project health**: Is the primary project active and strategic?
2. Check **developer satisfaction**: Does the developer want more variety or deeper focus?
3. Consider **rotation programs**: Move highly focused developers to new projects for skill development

**Related Metrics**: Work Style, Project Trends, Activity Score

---

### Work Style

**Definition**: Categorization of developer based on focus score and work patterns.

**Categories**:
- **Highly Focused**: > 80% of work on one project
- **Focused**: 60-80% on one project
- **Multi-project**: < 60% on one project
- **Large batch changes**: Few commits with many lines changed
- **Incremental contributor**: Many small commits

**What It Measures**: Developer's working approach and project assignment pattern.

**What "Good" Looks Like**:
- **Mix of work styles** on the team (specialists + generalists)
- **Work style aligns** with role (senior IC = focused, tech lead = multi-project)
- **Stable work style** over time (not switching randomly)

**Red Flags**:
- 🔴 **Entire team "Highly Focused"**: Knowledge silos, no cross-team coverage
- 🔴 **Entire team "Multi-project"**: Fragmented, context-switching overhead
- ⚠️ **Work style changing frequently**: Unstable assignments or unclear priorities

**Related Metrics**: Focus Score, Project Distribution

---

### Time Patterns

**Definition**: When during the day the developer typically commits code.

**Categories**:
- **Midday developer**: 9 AM - 5 PM (typical working hours)
- **Afternoon developer**: 12 PM - 8 PM (later shift)
- **Morning developer**: 6 AM - 2 PM (early shift)
- **Extended Hours**: Commits outside typical 8-hour window

**What It Measures**: Work schedule and potential overwork patterns.

**What "Good" Looks Like**:
- **Midday or Afternoon**: Standard working hours
- **Consistent pattern**: Developer has stable schedule
- **Mix across team**: Different time zones or schedules (if distributed team)

**Red Flags**:
- 🔴 **Extended Hours consistently**: Burnout risk, unrealistic deadlines
- 🔴 **Multiple developers with Extended Hours**: Team-wide workload issue
- ⚠️ **Pattern suddenly shifts**: Investigate cause (timezone change, workload spike, personal issues)

**Action When Concerning**:
1. **1-on-1 check-in**: Ask about workload and deadlines
2. **Review sprint capacity**: Are estimates realistic?
3. **Redistribute work**: If workload is too high, rebalance team
4. **Set boundaries**: Encourage healthy work-life balance

**Related Metrics**: Activity Score, Gini Coefficient

---

## Quality Metrics

### Commit Message Quality

**Definition**: Average number of words per commit message.

**What It Measures**: Documentation thoroughness—how well do commit messages explain changes?

**Interpretation**:
- ✅ **40+ words**: Detailed context (what, why, how)
- ⚠️ **10-40 words**: Minimal info (what only, no why)
- 🔴 **< 10 words**: Poor documentation (e.g., "fix", "update", "wip")

**Why It Matters**:
- **Future debugging**: Detailed messages help understand why changes were made
- **Onboarding**: New developers can learn from commit history
- **Code archaeology**: Investigating bugs or regressions

**What "Good" Looks Like**:
- **Team average > 40 words**: Detailed, contextual messages
- **Improving trend**: Team adopting better commit practices
- **Examples of good messages**:
  ```
  Fix authentication timeout in production (67 words)

  Users reported session timeouts after 5 minutes instead of
  30 minutes. Root cause: Redis TTL was set in seconds, not
  milliseconds. Changed session.timeout config from 1800 to
  1800000. Tested in staging with 30-minute idle session.

  Fixes: AUTH-1234
  ```

**Red Flags**:
- 🔴 **Team average < 10 words**: Widespread poor documentation
- 🔴 **Many "fix", "update", "wip" messages**: No context, hard to maintain
- ⚠️ **Declining trend**: Team rushing or forgetting standards

**Action When Low**:
1. Update **commit message standards** (document expectations)
2. Share **examples** of good commit messages
3. Consider **commit message templates** (git commit.template)
4. Educate team on **benefits** of detailed messages

**Related Metrics**: Ticket Coverage (commit messages link to tickets)

---

### Classification Confidence

**Definition**: Average confidence score (0-100%) of the ML classification model when categorizing commits.

**What It Measures**: How certain the ML model is about its categorizations.

**Interpretation**:
- ✅ **> 85% average**: Highly reliable categorization (trust the classifications)
- ⚠️ **70-85% average**: Moderately reliable (spot-check classifications)
- 🔴 **< 70% average**: Low confidence (manual review recommended)

**Why It Matters**:
- **High confidence** = You can trust Feature %, Bug Fix %, etc.
- **Low confidence** = Classifications may be inaccurate, trends unreliable

**What "Good" Looks Like**:
- **Team average > 85%**: ML model understands your commit patterns
- **Improving over time**: Model learns from your team's style
- **High confidence on critical classifications** (Features, Bug Fixes)

**Action When Low**:
1. Review **low-confidence commits** (listed in qualitative report)
2. Improve **commit message detail** (helps ML accuracy)
3. Use **standardized commit prefixes** (e.g., "feat:", "fix:", "refactor:")
4. Report patterns to GitFlow Analytics team (helps improve model)

**Related Metrics**: Classification Mix (Feature %, Bug Fix %, etc.)

---

## Process Health Metrics

### Classification Mix

**Definition**: Percentage breakdown of commits by type (Features, Bug Fixes, Maintenance, etc.).

**Categories**:
- **Features**: New functionality, enhancements
- **Bug Fixes**: Error corrections, patches
- **Refactor**: Code restructuring (no behavior change)
- **Maintenance**: Dependency updates, config changes, tech debt
- **Documentation**: README, comments, docs
- **Testing**: Test additions, test refactoring
- **Build/CI**: Pipeline, build config, deployments
- **Performance**: Optimization work

**What It Measures**: Type of work the team is doing.

**Healthy Ranges by Project Stage**:

**Early-Stage Project**:
- ✅ Features: 60-80%
- ✅ Bug Fixes: 10-20%
- ✅ Maintenance: 10-20%

**Mature Product**:
- ✅ Features: 40-50%
- ✅ Bug Fixes: 30-40%
- ✅ Maintenance: 20-30%

**Platform/Infrastructure**:
- ✅ Features: 30-40%
- ✅ Bug Fixes: 20-30%
- ✅ Maintenance: 30-50%

**What "Good" Looks Like**:
- **Mix aligns** with project stage and team goals
- **Stable percentages** (no wild swings week-to-week)
- **Refactoring/Maintenance > 10%**: Team managing tech debt

**Red Flags**:
- 🔴 **Bug Fixes > 60%**: Quality crisis, firefighting mode
- 🔴 **Features declining + Bug Fixes increasing**: Quality degradation
- 🔴 **Maintenance 0%**: Tech debt accumulating, future problems
- ⚠️ **Features > 80% in mature product**: May be ignoring quality/tech debt

**Action When Concerning**:

| Pattern | Action |
|---------|--------|
| **Bug Fixes > 60%** | Root cause analysis, add testing, pause features to stabilize |
| **Maintenance 0%** | Schedule tech debt sprints, allocate % of capacity to refactoring |
| **Features declining** | Review sprint goals—is team shifting to stabilization or distracted? |

**Related Metrics**: Classification Confidence, Untracked Work Analysis

---

### Platform Distribution (Ticket Tracking)

**Definition**: Breakdown of which project management tools are referenced in commits.

**Platforms Detected**:
- JIRA (e.g., "PROJ-123")
- GitHub Issues (e.g., "#456")
- ClickUp (e.g., "CU-789")
- Linear (e.g., "ENG-123")

**What It Measures**: Tool adoption and usage patterns.

**Use For**:
- Understanding **which PM tools** the team uses
- Identifying **fragmented tracking** (multiple tools)
- Planning **integration priorities**

**What "Good" Looks Like**:
- **Primary tool > 70%**: Team standardized on one platform
- **Platform distribution stable**: No sudden tool switching

**Red Flags**:
- ⚠️ **Highly fragmented** (4+ tools all < 30%): No standard, hard to track work
- ⚠️ **Platform distribution shifting**: Tool migration or team confusion

**Related Metrics**: Ticket Coverage, Untracked Work Analysis

---

## Ticketing Activity

GitFlow Analytics fetches actual activity from ticketing and collaboration platforms and computes a per-developer `ticketing_score`. This score is blended into `raw_activity_score` by `ActivityScorer` (default weight: 15%).

### Supported Platforms

| Platform | Auto-enabled when… |
|----------|-------------------|
| **GitHub Issues** | `github_issues:` block is present in config |
| **Confluence** | `confluence:` block is present in config |
| **JIRA** | `jira:` credentials block is present (no extra config needed) |

### Point Values

Each event type contributes a fixed number of points to a developer's `ticketing_score`:

| Event | Points |
|-------|--------|
| GitHub issue opened | 1.0 |
| GitHub issue closed | 1.5 |
| GitHub comment | 0.5 |
| Confluence page created | 2.0 |
| Confluence page edited | 1.0 |
| JIRA issue opened | 1.5 |
| JIRA issue closed | 2.0 |
| JIRA comment | 0.5 |

### Output Files

- **`ticketing_activity_summary.json`** — combined per-developer `ticketing_score` across all platforms
- **`github_issues_summary.json`** — issue breakdown, resolution times, top contributors
- **`confluence_activity_summary.json`** — page edits by space and author
- **`developer_activity_summary_*.csv`** — includes `ticketing_score` column alongside other activity metrics

### Interpretation

- A `ticketing_score` of `0.0` with integrations configured indicates no tracked activity on those platforms during the analysis window — not that the integration is broken.
- Setting `activity_scoring.ticketing_weight: 0.0` is a strict no-op: the `ticketing_score` column still appears in CSV output but does not influence `raw_activity_score`. This preserves backward compatibility for installations without ticketing integrations.

**Related Metrics**: Developer Activity Score, Platform Distribution (Ticket Tracking)

---

## Advanced Metrics

### Velocity Trend

**Definition**: Week-over-week change in commit volume or story points delivered.

**Categories**:
- **Growing**: Commits increasing week-over-week
- **Stable**: Commits consistent (±10%)
- **Declining**: Commits decreasing week-over-week

**What It Measures**: Team capacity trend.

**What "Good" Looks Like**:
- **Stable**: Consistent, predictable velocity
- **Growing**: Team ramping up, onboarding complete, or project scaling
- **Declining (expected)**: Stabilization phase, holiday season, planned transition

**Red Flags**:
- 🔴 **Declining + no clear reason**: Blockers, attrition, or shifting priorities
- 🔴 **Volatile** (spike + drop cycles): Crunch → recovery pattern (unsustainable)

**Related Metrics**: Total Commits, Story Points Delivered

---

### Untracked Work Analysis

**Definition**: Breakdown of commits without ticket references, categorized by type.

**What It Measures**: What work is happening outside planned/tracked tasks.

**Healthy Untracked Work**:
- ✅ **Maintenance**: Dependency updates, config changes (often not ticketed)
- ✅ **Documentation**: README updates, comments
- ✅ **Small refactors**: Minor cleanups

**Concerning Untracked Work**:
- 🔴 **Features**: Major new functionality without planning
- 🔴 **Bug Fixes**: Firefighting not captured in sprint planning
- ⚠️ **High percentage overall** (> 40%): Process breakdown

**Action When High**:
1. Review **specific untracked commits**: Should they have had tickets?
2. Update **process**: Ensure all features and bugs get tracked
3. Educate team on **importance** of linking commits to work items

**Related Metrics**: Ticket Coverage, Classification Mix

---

## Benchmarking and Context

### Team Size Adjustments

Many metrics scale with team size. Use these adjustments:

| Team Size | Gini Threshold | Expected Avg Commits/Dev | Focus Score |
|-----------|---------------|-------------------------|-------------|
| 2-3 developers | 0.4 | 15-30/week | 70-90% |
| 5-8 developers | 0.3 | 10-25/week | 60-80% |
| 10-15 developers | 0.25 | 8-20/week | 50-70% |
| 15+ developers | 0.20 | 5-15/week | 40-60% |

### Industry Benchmarks

| Metric | Startup | Mid-Size | Enterprise |
|--------|---------|----------|------------|
| **Ticket Coverage** | 40-60% | 60-75% | 70-85% |
| **Commit Quality** | 20-40 words | 30-50 words | 40-60 words |
| **Feature %** | 60-80% | 40-60% | 30-50% |
| **Bug Fix %** | 10-20% | 25-40% | 30-45% |
| **Maintenance %** | 10-20% | 15-25% | 20-35% |

---

## Next Steps

- **Interpret your first report**: See [Report Interpretation Guide](interpreting-reports.md)
- **Create dashboards**: See [Dashboard Guide](dashboard-guide.md)
- **Common questions**: See [FAQ](faq.md)
