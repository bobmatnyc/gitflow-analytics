# Manager FAQ

Frequently asked questions about GitFlow Analytics for engineering managers and team leads.

## Getting Started

### Q: Do I need to install anything to use GitFlow Analytics?

**A: No.** GitFlow Analytics runs on your technical team's machines. As a manager, you receive the generated reports (markdown files and CSVs). You don't need to install Python, Git, or any command-line tools.

**Delegation Model**:
1. Share the [Installation Guide](../getting-started/installation.md) with your technical lead
2. Have them run analysis weekly or monthly
3. Receive reports in the `./reports/` directory
4. Review the narrative report and import CSVs to spreadsheets

See **[Quick Start Guide](quickstart.md)** for the complete delegation workflow.

---

### Q: How often should I run GitFlow Analytics?

**A: It depends on your needs.** Here are common cadences:

| Cadence | Use Case | Analysis Period | Best For |
|---------|----------|-----------------|----------|
| **Weekly** | Sprint retrospectives | Last 2-4 weeks | Agile teams, fast-moving projects |
| **Bi-weekly** | Sprint planning | Last 4-6 weeks | Standard 2-week sprints |
| **Monthly** | Team health checks | Last 8-12 weeks | Monthly reviews, quarterly planning prep |
| **Quarterly** | Business reviews | Last 12-24 weeks | Executive summaries, long-term trends |

**Recommendation**: Start with **bi-weekly or monthly**, then adjust based on value.

**Pro Tip**: Run with longer periods (8-12 weeks) to smooth out volatility and see clearer trends.

---

### Q: What information do I need to provide my technical team?

**A: Minimal information.** Your technical team needs:

1. **GitHub organization or repository URLs** (if using GitHub)
   - Example: `github.com/your-org` or specific repos
2. **GitHub token** (for accessing private repositories)
   - Your team can generate this: [GitHub Token Guide](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
3. **Analysis period** (how many weeks back to analyze)
   - Example: "Last 8 weeks"

**Optional**:
- JIRA, ClickUp, or Linear configuration (if you want enhanced ticket tracking)
- Developer alias mappings (if developers use multiple email addresses)

See **[Configuration Guide](../configuration/configuration.md)** for technical details (for your team).

---

## Understanding Reports

### Q: What if ticket coverage is low or 0%?

**A: This is common and fixable.** Low ticket coverage means commits don't reference work items (JIRA, GitHub, ClickUp, Linear).

**Common Causes**:
1. **Team doesn't link commits to tickets**: Most common cause
2. **Ticket format not recognized**: GitFlow Analytics looks for patterns like "PROJ-123", "#456", "CU-789"
3. **No PM tool in use**: Team doesn't use JIRA/Linear/etc.

**What to Do**:

**Short Term (Immediate)**:
- **Review untracked work**: Check the "Untracked Work Analysis" section of the report
- **Identify critical gaps**: Are untracked commits Features or Bug Fixes? (Concerning) Or Maintenance? (Often acceptable)
- **Prioritize**: Focus on tracking Features and Bug Fixes, not necessarily dependency updates

**Medium Term (1-2 Sprints)**:
1. **Update commit message standards**: Require ticket references (e.g., "Fix auth bug [PROJ-123]")
2. **Pre-commit hooks**: Validate ticket references before commits (optional, requires technical setup)
3. **Sprint planning review**: Review previous sprint's untracked work to identify patterns

**Long Term (1-2 Months)**:
- **Process improvement**: Make ticket linking part of Definition of Done
- **Education**: Help team understand why tracking matters (reporting, planning, accountability)
- **Target 60-80%**: Set realistic goal (100% is rarely achievable or necessary)

**Context**: 0% ticket coverage is normal for:
- Personal projects
- Prototypes/experiments
- Open source contributions (no formal PM tool)

For production teams, target **60-80%** over time.

---

### Q: How do I compare multiple teams?

**A: Run GitFlow Analytics separately for each team's repositories.**

**Workflow**:
1. **Create separate configurations** for each team
   - `team-a-config.yaml`
   - `team-b-config.yaml`
2. **Run analysis for each team**:
   ```bash
   gitflow-analytics -c team-a-config.yaml --weeks 8
   gitflow-analytics -c team-b-config.yaml --weeks 8
   ```
3. **Import summary CSVs to spreadsheet**:
   - Create comparison table with key metrics side-by-side
4. **Normalize for team size**:
   - Use **per-developer metrics** (commits/developer, not total commits)
   - Compare **Gini coefficient** (work distribution)
   - Compare **ticket coverage %** (process adherence)

**Metrics for Cross-Team Comparison**:

| Metric | Why It's Comparable |
|--------|-------------------|
| **Ticket Coverage %** | Process adherence (independent of team size) |
| **Gini Coefficient** | Work distribution balance (adjusts for team size) |
| **Commits per Developer** | Normalized velocity (accounts for team size) |
| **Classification Mix** | Work type distribution (Features %, Bug Fix %) |
| **Avg Focus Score** | Specialization level |

**Avoid Comparing**:
- ❌ **Total commits** (larger teams = more commits, not meaningful)
- ❌ **Activity Scores** (percentile within team, not cross-team)
- ❌ **Absolute velocity** (context-dependent: team size, project complexity)

**Example Comparison Dashboard**:

| Team | Devs | Commits/Dev | Ticket Coverage | Gini | Feature % | Bug Fix % |
|------|------|-------------|-----------------|------|-----------|-----------|
| Frontend | 5 | 38.2 | 72.3% | 0.28 | 58% | 32% |
| Backend | 8 | 41.5 | 81.4% | 0.22 | 45% | 38% |
| Platform | 3 | 29.7 | 65.1% | 0.35 | 35% | 28% |

**Insights**: Backend has best process (81% coverage, low Gini). Frontend building most features. Platform is small team (higher Gini is normal).

---

### Q: What does "Gini coefficient" mean in plain language?

**A: Work distribution balance.** It measures how evenly work is distributed across your team.

**Simple Analogy**: Imagine slicing a pie among team members.
- **Gini = 0**: Everyone gets exactly equal slices (perfect balance)
- **Gini = 1**: One person gets the entire pie (extreme imbalance)

**Practical Interpretation**:

| Gini Value | What It Means | Real-World Example |
|------------|---------------|-------------------|
| **0.0-0.2** | Very balanced | 5 developers: 22%, 21%, 20%, 19%, 18% of work |
| **0.2-0.3** | Balanced (healthy) | 5 developers: 28%, 24%, 20%, 18%, 10% of work |
| **0.3-0.5** | Moderate concentration | 5 developers: 40%, 25%, 15%, 12%, 8% of work |
| **0.5-0.7** | High concentration | 5 developers: 55%, 20%, 12%, 8%, 5% of work |
| **0.7-1.0** | Extreme concentration | One person does 80%+ of work |

**Why It Matters**:
- **Low Gini (< 0.3)**: ✅ Low bus factor risk, sustainable team
- **High Gini (> 0.5)**: 🔴 High risk if key person leaves, potential burnout

**Context by Team Size**:
- **2-3 developers**: Gini 0.3-0.4 is normal (mathematical limit)
- **5-8 developers**: Gini < 0.3 is achievable
- **10+ developers**: Gini < 0.25 indicates excellent balance

**Action if High**:
1. Identify top contributors (who's carrying >40% of work?)
2. Review project assignments (can work be redistributed?)
3. Knowledge sharing (pair programming, cross-training)

See **[Metrics Reference](metrics-reference.md#gini-coefficient-work-distribution)** for detailed explanation.

---

### Q: What's a healthy classification mix (Features vs Bug Fixes)?

**A: It depends on your project stage.** There's no universal "good" ratio, but here are guidelines:

**Early-Stage Project (First 6-12 Months)**:
- ✅ **Features**: 60-80% (building new capabilities)
- ✅ **Bug Fixes**: 10-20% (minimal bugs, new code)
- ✅ **Maintenance**: 10-20% (light refactoring)

**Mature Product (1-3 Years Old)**:
- ✅ **Features**: 40-50% (balanced development)
- ✅ **Bug Fixes**: 30-40% (stabilization, edge cases)
- ✅ **Maintenance**: 20-30% (tech debt management)

**Legacy System (3+ Years)**:
- ✅ **Features**: 30-40% (slower feature development)
- ✅ **Bug Fixes**: 30-40% (ongoing fixes)
- ✅ **Maintenance**: 30-40% (significant refactoring, updates)

**Platform/Infrastructure**:
- ✅ **Features**: 30-40% (new capabilities)
- ✅ **Bug Fixes**: 20-30% (stability)
- ✅ **Maintenance**: 30-50% (upkeep, updates, optimization)

**Warning Signs**:

| Pattern | What It Means | Action |
|---------|---------------|--------|
| 🔴 **Bug Fixes > 60%** | Quality crisis, firefighting | Root cause analysis, pause features to stabilize |
| 🔴 **Maintenance 0%** | Tech debt accumulating | Schedule refactoring sprints |
| 🔴 **Features declining over time** | Project stagnating or quality issues | Investigate blockers or quality problems |
| ⚠️ **Features > 80% in mature product** | Ignoring quality/tech debt | Balance with maintenance work |

**How to Use This**:
1. Compare your mix to the appropriate category above
2. Track trends (is Bug Fix % increasing? Concerning.)
3. Adjust sprint planning (allocate % of capacity to maintenance if 0%)

See **[Report Interpretation Guide](interpreting-reports.md#classification-mix)** for examples.

---

## Taking Action

### Q: What actions should I take based on findings?

**A: Focus on 1-2 high-impact improvements per sprint.** Here are prioritized actions by finding:

### 🔴 Critical (Immediate Action)

| Finding | Action | Timeline |
|---------|--------|----------|
| **Ticket Coverage < 40%** | Review untracked work with team, update commit standards | This sprint |
| **Gini > 0.5** | Identify top contributor, redistribute work, cross-train | This sprint |
| **Bug Fixes > 60%** | Root cause analysis, pause features, add testing | This sprint |
| **Extended Hours (multiple developers)** | 1-on-1 check-ins, workload review, rebalance | This week |

### ⚠️ Important (Address Soon)

| Finding | Action | Timeline |
|---------|--------|----------|
| **Ticket Coverage 40-60%** | Process review, educate team on linking commits | Next 2 sprints |
| **Gini 0.3-0.5** | Monitor key contributors, knowledge sharing sessions | Next month |
| **Maintenance 0%** | Schedule tech debt sprint, allocate 10-20% capacity | Next quarter |
| **Bottom 20% Activity + Declining** | 1-on-1 to identify blockers, provide support | Next sprint |

### ✅ Monitor (Track Over Time)

| Finding | Action | Timeline |
|---------|--------|----------|
| **Ticket Coverage 60-80%** | Maintain standards, celebrate success | Ongoing |
| **Gini < 0.3** | No action, healthy balance | Ongoing |
| **Velocity stable** | Track trends, use for sprint planning | Ongoing |

**Prioritization Framework**:
1. **Safety first**: Address Extended Hours patterns (burnout risk)
2. **Risk second**: High Gini (bus factor), low Ticket Coverage (process breakdown)
3. **Quality third**: High Bug Fix %, low Maintenance %
4. **Process improvements**: Ticket coverage, commit quality

**Don't Try to Fix Everything**: Pick 1-2 items per sprint. Track improvement in next report.

---

### Q: How do I use reports in sprint retrospectives?

**A: Focus on 3 key areas in 10 minutes.**

**Sprint Retrospective Workflow** (10 minutes):

**1. Velocity Check (2 minutes)**
- **Open**: Weekly Metrics section or velocity chart
- **Review**: Did we commit more/less than planned?
- **Discuss**: "We had 42 commits this sprint vs 38 last sprint. Velocity is stable."

**2. Ticket Coverage Review (3 minutes)**
- **Open**: Untracked Work Analysis section
- **Identify gaps**: "15 commits (24%) were untracked. 8 were features—why didn't they have tickets?"
- **Action**: "Let's create tickets for large features before coding next sprint."

**3. Pattern Identification (5 minutes)**
- **Open**: Classification Breakdown
- **Discuss trends**: "Bug Fixes increased from 30% to 45% this sprint. What caused that?"
- **Action**: "Let's investigate the root cause and add test coverage."

**Example Retrospective Discussion**:

**Facilitator**: "Looking at GitFlow Analytics, our ticket coverage dropped from 75% to 62% this sprint. Untracked work analysis shows 12 feature commits without tickets. What happened?"

**Developer**: "We got urgent requests from Product and coded them immediately without creating tickets first."

**Action**: "Going forward, all urgent requests get a ticket created in JIRA before coding. Even if it's retroactive, create the ticket and reference it in the commit."

**Outcome**: Process improvement captured for next sprint.

**What to Bring to Retrospectives**:
- ✅ Narrative report (Executive Summary, Classification sections)
- ✅ Velocity chart (week-by-week trends)
- ✅ Untracked work breakdown (specific gap identification)
- ❌ Don't bring: Individual Activity Scores (use for 1-on-1s, not team retros)

---

### Q: Can I use this for performance reviews?

**A: Yes, but with important caveats.**

**✅ DO Use GitFlow Analytics For**:
- **Contextual data**: Activity patterns, project contributions, work styles
- **Team-level trends**: Overall velocity, process adherence
- **Discussion starters**: "I noticed you're highly focused on Project X—how's that going?"
- **Workload assessment**: "Your Activity Score is Top 10% + Extended Hours pattern—are you overloaded?"

**❌ DO NOT Use As**:
- **Performance metric**: "Your Activity Score is Bottom 20%, you're underperforming" ❌
- **Sole evaluation**: Commits don't measure impact, code quality, mentoring, design work
- **Comparison tool**: "Developer A has more commits than Developer B" ❌
- **Punitive measure**: Low metrics ≠ bad performance

**Why Activity Metrics Are Insufficient**:
- Senior architects may have low commit volume but high impact (design, mentoring)
- Junior developers may have high commit volume but need more code review
- Platform engineers may have low commit volume but critical infrastructure work
- Bug fixes on critical systems may be more valuable than features on side projects

**How to Use Responsibly**:

**Good Example**:
- "I see you've been working extended hours consistently. Let's talk about workload and how I can help."
- "Your focus is 90% on Project X. Are you interested in exploring other projects for skill development?"
- "Ticket coverage for your commits is low. Let's review commit message practices."

**Bad Example**:
- "Your Activity Score is Bottom 20%. You need to work harder." ❌
- "You have fewer commits than your peers. Explain yourself." ❌
- "Your Bug Fix % is high. You're making too many mistakes." ❌

**Recommendation**: Use GitFlow Analytics as **one data point** alongside:
- 1-on-1 conversations
- Peer feedback
- Code review quality
- Project outcomes
- Stakeholder feedback
- Technical skills assessment

**Performance Review Framework**:
1. **Impact**: Did they deliver high-quality work that moved projects forward?
2. **Collaboration**: Did they help teammates, share knowledge, improve processes?
3. **Technical Growth**: Did they learn new skills, tackle complex challenges?
4. **GitFlow Analytics**: Provides context on work patterns, not conclusions

---

## Technical Questions

### Q: Do I need JIRA or other PM tools to use GitFlow Analytics?

**A: No.** GitFlow Analytics works **without** JIRA, Linear, ClickUp, or any PM tool.

**What GitFlow Analytics Provides Without PM Tools**:
- ✅ Commit volume and velocity trends
- ✅ Developer activity and work distribution
- ✅ ML-powered commit classification (Features, Bug Fixes, Maintenance)
- ✅ Code churn and project activity
- ✅ Developer focus scores and work patterns

**What You Get WITH PM Tool Integration**:
- ➕ **Ticket Coverage**: % of commits linked to planned work
- ➕ **Story Point Tracking**: Estimated effort vs actual commits
- ➕ **Platform Distribution**: Which PM tools the team uses
- ➕ **Untracked Work Analysis**: What's happening outside tickets

**Recommendation**:
- **Start without PM integration**: Get insights from Git history alone
- **Add PM tools later**: If you want ticket coverage and story point tracking
- **Optional, not required**: Core value is Git analysis, PM tools enhance it

See **[PM Platform Setup Guide](../guides/pm-platform-setup.md)** for integration details (optional).

---

### Q: How accurate is the ML classification?

**A: 85-95% accurate for most teams.**

**How It Works**:
GitFlow Analytics uses machine learning to categorize commits based on:
- Commit message content (keywords, patterns)
- File changes (which files were modified)
- Code patterns (added lines, deleted lines, refactoring patterns)

**Accuracy Metrics**:
- ✅ **> 85% team average**: Highly reliable (trust the classifications)
- ⚠️ **70-85% average**: Moderately reliable (spot-check if critical decisions)
- 🔴 **< 70% average**: Low confidence (manual review recommended)

**Check Your Accuracy**:
Look for "Classification Confidence" in the Commit Classification Analysis section of the report.

**Example**:
```
Classification Confidence: 87.3% of commits classified with >80% confidence
```

This means the ML model is **87% confident** in its categorizations—highly reliable.

**What Affects Accuracy**:
- ✅ **Detailed commit messages**: "Fix auth timeout bug" → High confidence Bug Fix
- ✅ **Conventional commit prefixes**: "feat:", "fix:", "refactor:" → Very high confidence
- ❌ **Vague messages**: "update", "fix" → Low confidence
- ❌ **Mixed changes**: Feature + bug fix in one commit → Lower confidence

**Improving Accuracy**:
1. **Use detailed commit messages** (40+ words helps)
2. **Adopt conventional commits** (e.g., "feat: add user login")
3. **Separate concerns**: One commit = one type of change (feature OR bug fix, not both)

**When to Manually Review**:
- Critical decisions based on classification (e.g., "50% bugs means quality crisis")
- Low confidence scores (< 70% average)
- Unexpected patterns (e.g., sudden spike in Maintenance %)

See **[ML Categorization Guide](../guides/ml-categorization.md)** for technical details.

---

### Q: Can GitFlow Analytics access private repositories?

**A: Yes, with a GitHub token.**

**Requirements**:
1. **GitHub Personal Access Token** with `repo` permissions
2. **Team configures token** in GitFlow Analytics configuration
3. **Token kept secure** (not committed to Git, use `.env` file)

**Security**:
- ✅ Token stored locally (not sent to external servers)
- ✅ GitFlow Analytics runs on your team's machines (not SaaS)
- ✅ Data stays within your infrastructure
- ✅ No external API calls for analysis (only GitHub API for fetching repos)

**Privacy**:
GitFlow Analytics includes **data anonymization** features for external sharing:
- Developer names → "Developer A", "Developer B"
- Repository names → "Project 1", "Project 2"
- Commit messages → Redacted (only classifications kept)

See **[Security Documentation](../SECURITY.md)** for details.

---

## Process & Best Practices

### Q: What's the difference between Activity Score and commits?

**A: Activity Score is percentile ranking, commits are absolute volume.**

| Metric | Definition | Example |
|--------|-----------|---------|
| **Commits** | Absolute number of commits | "Sarah has 42 commits" |
| **Activity Score** | Percentile rank within team (0-100) | "Sarah is at 85/100 (Top 10%)" |

**Activity Score Calculation**:
1. Rank all developers by commit volume
2. Convert to percentile (0 = lowest, 100 = highest)
3. Categorize: Top 10%, Top 30%, Middle 40%, Bottom 30%, Bottom 10%

**Why Use Activity Score**:
- ✅ **Comparable across periods**: "Top 10%" is meaningful even if total commits change
- ✅ **Identifies outliers**: Easier to spot "Bottom 20%" than "18 commits vs 42 commits"
- ✅ **Context-independent**: Works for small and large teams

**Why Use Commits**:
- ✅ **Absolute volume**: "We had 324 commits this sprint"
- ✅ **Velocity tracking**: "Commits increased from 38/week to 42/week"
- ✅ **Capacity planning**: "Historical average is 40 commits/week"

**Use Both Together**:
- **Commits**: Track team velocity and capacity
- **Activity Score**: Identify individual outliers (very high or very low)

**Example**:
- **Developer A**: 42 commits, Activity Score 85/100 (Top 10%)
- **Developer B**: 38 commits, Activity Score 72/100 (Top 30%)
- **Developer C**: 12 commits, Activity Score 15/100 (Bottom 20%)

**Insight**: Developer C is an outlier. Check if this is expected (part-time, onboarding, blockers) or needs attention.

---

### Q: How do I balance features vs bug fixes vs maintenance?

**A: Allocate capacity based on project stage and goals.**

**Recommended Allocation**:

| Project Stage | Features | Bug Fixes | Maintenance |
|---------------|----------|-----------|-------------|
| **Early-Stage (MVP)** | 70-80% | 10-15% | 10-15% |
| **Growth Phase** | 50-60% | 20-30% | 20-30% |
| **Mature Product** | 40-50% | 30-40% | 20-30% |
| **Stabilization** | 20-30% | 40-50% | 30-40% |

**How to Use GitFlow Analytics**:
1. **Check current mix**: See Classification Breakdown in report
2. **Compare to target**: Are you aligned with project stage?
3. **Adjust sprint planning**: Allocate capacity accordingly

**Example Scenario**:

**Current State (from report)**:
- Features: 62%
- Bug Fixes: 35%
- Maintenance: 3%

**Project Stage**: Mature product (2 years old)

**Analysis**:
- ✅ Feature % is healthy (within 40-50% target)
- ✅ Bug Fix % is expected for mature product
- 🔴 Maintenance is too low (3% vs 20-30% target)

**Action**:
- Schedule "Tech Debt Sprint" (allocate 50% capacity to maintenance)
- Long-term: Reserve 20% of each sprint for refactoring/updates
- Track Maintenance % in next report (should increase)

**Sprint Planning Template**:

**Week 1-2 Sprint**:
- 50% capacity: New features (planned work)
- 30% capacity: Bug fixes (backlog + new issues)
- 20% capacity: Maintenance (tech debt, refactoring, dependency updates)

**Stabilization Sprint (e.g., pre-release)**:
- 20% capacity: Critical features only
- 50% capacity: Bug fixes (polish, edge cases)
- 30% capacity: Performance, optimization, docs

---

## Troubleshooting

### Q: Report shows 0 commits or incomplete data

**A: This usually means repository access or configuration issues.**

**Common Causes**:
1. **GitHub token missing or invalid**
2. **Repository path incorrect**
3. **Time period has no activity** (e.g., holiday weeks)
4. **Branch not specified** (analyzing wrong branch)

**How to Debug** (for your technical team):
1. Check configuration file (`config.yaml`):
   - GitHub token present?
   - Repository URLs correct?
   - Branch name specified? (default is `main`)
2. Run with verbose logging:
   ```bash
   gitflow-analytics -c config.yaml --weeks 8 --verbose
   ```
3. Check error messages in output

**Common Fixes**:
- **Token issue**: Regenerate GitHub token with `repo` permissions
- **Branch issue**: Specify branch in config: `branch: main` or `branch: master`
- **Time period**: Extend analysis period: `--weeks 12` instead of `--weeks 4`

See **[Troubleshooting Guide](../guides/troubleshooting.md)** for detailed debugging.

---

### Q: Classifications seem wrong (e.g., feature labeled as bug fix)

**A: ML classification can be improved with better commit messages.**

**Why Misclassification Happens**:
1. **Vague commit messages**: "update", "fix" → Hard to classify
2. **Mixed changes**: One commit with feature + bug fix
3. **Unusual patterns**: Team's commit style differs from ML training data

**How to Improve**:
1. **Use conventional commit prefixes**:
   - `feat: add user authentication` → Definitely Feature
   - `fix: resolve timeout in API` → Definitely Bug Fix
   - `refactor: extract validation logic` → Definitely Refactor
2. **Write detailed messages**: Include context (what, why)
3. **Separate concerns**: One commit = one type of change

**Quick Fix for Reports**:
If classifications are critical for decision-making:
1. Manually review low-confidence commits (listed in qualitative report)
2. Focus on overall trends, not individual commits
3. Use classification as directional signal, not absolute truth

**Long-Term Improvement**:
Adopt **Conventional Commits** standard: https://www.conventionalcommits.org/

---

## Advanced Topics

### Q: Can I track multiple teams or organizations?

**A: Yes.** Run GitFlow Analytics separately for each team/org.

**Workflow**:
1. Create separate configuration files:
   - `team-frontend.yaml`
   - `team-backend.yaml`
   - `team-platform.yaml`
2. Run analysis for each:
   ```bash
   gitflow-analytics -c team-frontend.yaml --weeks 8
   gitflow-analytics -c team-backend.yaml --weeks 8
   gitflow-analytics -c team-platform.yaml --weeks 8
   ```
3. Compare summary CSVs across teams

**Organization-Wide Analysis**:
You can also analyze entire GitHub organizations:
```yaml
github:
  organization: "your-org"
  token: "${GITHUB_TOKEN}"
```

This discovers all repositories and generates a consolidated report.

See **[Quick Start](quickstart.md)** for configuration examples.

---

### Q: How do I anonymize data for external sharing?

**A: GitFlow Analytics includes built-in anonymization.**

**Use Case**: Share reports with consultants, executives outside engineering, or external auditors without exposing developer names or sensitive code.

**Anonymization Features** (Technical team configures):
1. **Developer names** → "Developer A", "Developer B", etc.
2. **Repository names** → "Project 1", "Project 2", etc.
3. **Commit messages** → Redacted (only classifications kept)
4. **File paths** → Generalized or removed

**How to Use**:
Ask your technical team to run with anonymization flag (see [CLI Reference](../reference/cli-commands.md)).

**What's Preserved**:
- ✅ Metrics (ticket coverage, Gini, activity scores)
- ✅ Trends (week-by-week patterns)
- ✅ Classifications (Feature %, Bug Fix %)
- ❌ No names, repos, or code content

**Security**: Use for external sharing only. Internal reviews benefit from full context.

---

## Getting More Help

### Q: Where can I find more documentation?

**Manager Resources**:
- **[Quick Start Guide](quickstart.md)** - 5-minute onboarding
- **[Report Interpretation](interpreting-reports.md)** - Detailed report walkthrough
- **[Metrics Reference](metrics-reference.md)** - All metrics explained
- **[Dashboard Guide](dashboard-guide.md)** - Create visual dashboards
- **[This FAQ](faq.md)** - Common questions

**Technical Documentation** (for your team):
- [User Guide](../getting-started/) - Installation and setup
- [Configuration Guide](../configuration/configuration.md) - YAML configuration
- [CLI Reference](../reference/cli-commands.md) - Command-line options
- [Troubleshooting](../guides/troubleshooting.md) - Error resolution

---

### Q: I have a question not answered here. Where should I ask?

**Internal Questions** (about your reports):
1. Ask your technical lead who runs GitFlow Analytics
2. Reference this FAQ or [Metrics Reference](metrics-reference.md) together
3. Review [Report Interpretation Guide](interpreting-reports.md)

**Product Questions** (about GitFlow Analytics itself):
1. Check [GitHub Issues](https://github.com/bobmatnyc/gitflow-analytics/issues)
2. Search existing documentation (this FAQ, User Guide)
3. Open a new GitHub issue with your question

**Process/Methodology Questions**:
1. Reference [Metrics Reference](metrics-reference.md) for definitions
2. See industry benchmarks in [Metrics Reference](metrics-reference.md#benchmarking-and-context)
3. Consult with your engineering leadership on team-specific context

---

**Still have questions?** Open an issue on [GitHub](https://github.com/bobmatnyc/gitflow-analytics/issues) or ask your technical lead to reference the [User Guide](../getting-started/).
