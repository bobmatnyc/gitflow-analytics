# Dashboard Guide: Visualizing Your Data

Learn how to import GitFlow Analytics CSV exports into Excel and Google Sheets to create executive dashboards and track trends over time.

## Quick Start (5 Minutes)

### What You'll Need
- GitFlow Analytics CSV files (in `./reports/` directory)
- Excel, Google Sheets, or similar spreadsheet tool

### The Process
1. Import CSV files to your spreadsheet tool
2. Create pivot tables and charts
3. Set up automatic weekly refresh (optional)

**Result**: Visual dashboards for team reviews, executive summaries, and quarterly planning.

## CSV Files Overview

GitFlow Analytics generates multiple CSV files, each optimized for different views:

| CSV File | Purpose | Key Columns | Best For |
|----------|---------|-------------|----------|
| **summary_YYYYMMDD.csv** | Project-wide statistics | Total Commits, Ticket Coverage, Gini, Active Devs | Executive KPI cards |
| **weekly_metrics_YYYYMMDD.csv** | Week-by-week trends | Week, Commits, Developers, Ticket Coverage, Story Points | Velocity charts, trend lines |
| **developer_focus_YYYYMMDD.csv** | Developer profiles | Developer, Focus Score, Activity Score, Work Style | Team balance analysis |
| **activity_distribution_YYYYMMDD.csv** | Developer × Project matrix | Developer, Project, Commits, % of Work | Heatmaps, contribution matrices |
| **qualitative_insights_YYYYMMDD.csv** | Classification insights | Week, Feature %, Bug Fix %, Maintenance % | Work type trends |

**Start with**: `summary_YYYYMMDD.csv` and `weekly_metrics_YYYYMMDD.csv` for your first dashboard.

## Google Sheets Quick Start

### Step 1: Import CSV (2 minutes)

1. Open Google Sheets
2. **File → Import → Upload**
3. Select `weekly_metrics_YYYYMMDD.csv`
4. Import settings:
   - **Import location**: "Replace spreadsheet"
   - **Separator type**: "Comma"
   - **Convert text to numbers**: "Yes"

**Result**: Weekly metrics data loaded into Sheet1.

### Step 2: Create Summary Table (3 minutes)

Create a summary table for key metrics:

**A. Insert Summary Section**:
- Click cell A1
- Insert → Table (or manually create)
- Add these rows:

| Metric | Current Value | Previous Week | Change |
|--------|---------------|---------------|--------|
| Total Commits | =SUM(B:B) | | |
| Active Developers | =MAX(C:C) | | |
| Ticket Coverage | =AVERAGE(E:E) | | |
| Avg Commits/Week | =AVERAGE(B:B) | | |

**B. Use Formulas**:
Replace column letters with actual column names from your CSV:
- Total Commits: `=SUM(Commits)`
- Active Developers: `=MAX(Active_Developers)`
- Ticket Coverage: `=AVERAGE(Ticket_Coverage)`

### Step 3: Create Velocity Chart (2 minutes)

**A. Select Data**:
- Highlight columns: `Week`, `Commits`
- **Insert → Chart**

**B. Chart Settings**:
- **Chart type**: Line chart
- **X-axis**: Week
- **Y-axis**: Commits
- **Title**: "Weekly Commit Velocity"

**Customize**:
- Add **trendline** (Series → Trendline)
- Add **data labels** for recent weeks
- Set **colors** (blue for velocity, red for trendline)

**Result**: Visual trend showing team velocity over time.

### Step 4: Create Ticket Coverage Trend (2 minutes)

**A. Select Data**:
- Highlight: `Week`, `Ticket Coverage`
- **Insert → Chart**

**B. Chart Settings**:
- **Chart type**: Line chart
- **Add benchmark line**:
  - Right-click chart → Series → Add series
  - Create helper column with value 0.7 (70% benchmark)
  - Add as second series, dashed line

**Customize**:
- **Color**: Green for coverage, gray dashed for benchmark
- **Y-axis**: Format as percentage (0% - 100%)
- **Title**: "Ticket Coverage Trend (Target: 60-80%)"

**Result**: Visual showing process adherence improvement.

### Step 5: Add Conditional Formatting (1 minute)

Highlight cells based on thresholds:

**A. Ticket Coverage Column**:
- Select all Ticket Coverage values
- **Format → Conditional formatting**
- **Rules**:
  - Green: >= 0.6 (60%)
  - Yellow: 0.4 - 0.6 (40-60%)
  - Red: < 0.4 (< 40%)

**Result**: At-a-glance health indicators.

## Excel Quick Start

### Step 1: Import CSV (2 minutes)

1. Open Excel
2. **Data → From Text/CSV**
3. Select `weekly_metrics_YYYYMMDD.csv`
4. Import settings:
   - **File origin**: "Unicode (UTF-8)"
   - **Delimiter**: "Comma"
   - Click **Load**

**Result**: Data loaded into new worksheet.

### Step 2: Create PivotTable for Developer Focus (3 minutes)

**A. Insert PivotTable**:
- Open `developer_focus_YYYYMMDD.csv`
- Select all data (Ctrl+A)
- **Insert → PivotTable**

**B. PivotTable Settings**:
- **Rows**: Developer
- **Values**:
  - Total Commits (Sum)
  - Focus Score (Average)
  - Activity Score (Average)

**C. Sort**:
- Right-click values → Sort → Largest to Smallest

**Result**: Developer summary with key metrics.

### Step 3: Create Activity Heatmap (5 minutes)

**A. Import Data**:
- Open `activity_distribution_YYYYMMDD.csv`

**B. Create Matrix**:
- **Insert → PivotTable**
- **Rows**: Developer
- **Columns**: Project
- **Values**: Commits (Sum)

**C. Conditional Formatting**:
- Select all values in PivotTable
- **Home → Conditional Formatting → Color Scales**
- Choose: White → Yellow → Red (low → high)

**Result**: Heatmap showing developer × project contributions.

### Step 4: Create Dashboard Sheet (5 minutes)

**A. Create New Sheet**:
- Insert new worksheet named "Executive Dashboard"

**B. Add KPI Cards**:
Create text boxes with large numbers for key metrics:
- Total Commits (link to summary.csv)
- Active Developers
- Ticket Coverage
- Velocity Trend (↑ or ↓)

**C. Add Charts**:
- Copy velocity chart from Sheet1
- Copy ticket coverage chart
- Copy classification pie chart (from qualitative_insights.csv)

**D. Layout**:
```
+------------------+------------------+------------------+
| Total Commits    | Active Devs      | Ticket Coverage  |
|     324          |       8          |     78.4%        |
+------------------+------------------+------------------+
| [Velocity Chart ----------------------]               |
+-------------------------------------------------------+
| [Ticket Coverage Trend ---------------]               |
+-------------------------------------------------------+
| [Classification Pie] | [Developer Matrix]            |
+----------------------+-------------------------------+
```

**Result**: Single-page executive dashboard.

## Sample Dashboards

### 1. Executive Summary Dashboard

**Purpose**: High-level KPIs for stakeholders.

**Data Sources**:
- `summary_YYYYMMDD.csv`
- `weekly_metrics_YYYYMMDD.csv`

**Components**:

**KPI Cards** (Top Row):
- **Total Commits**: 324 (↑ 12% vs last period)
- **Active Developers**: 8
- **Ticket Coverage**: 78.4% (✅ Above benchmark)
- **Velocity**: 40.5 commits/week (↑ Stable)

**Charts**:
1. **Line Chart**: Weekly commit velocity (8 weeks)
2. **Line Chart**: Ticket coverage trend with 60-80% benchmark bands
3. **Pie Chart**: Classification breakdown (Features, Bug Fixes, Maintenance)
4. **Bar Chart**: Top 5 contributors

**Update Frequency**: Weekly

**Target Audience**: Directors, VPs, Product Managers

---

### 2. Team Health Dashboard

**Purpose**: Monitor work distribution, focus, and balance.

**Data Sources**:
- `developer_focus_YYYYMMDD.csv`
- `activity_distribution_YYYYMMDD.csv`

**Components**:

**KPI Cards**:
- **Gini Coefficient**: 0.24 (✅ Balanced)
- **Avg Focus Score**: 68.3% (✅ Healthy specialization)
- **Extended Hours Developers**: 0 (✅ No burnout risk)

**Charts**:
1. **Heatmap**: Developer × Project contribution matrix
2. **Scatter Plot**: Activity Score (X) vs Focus Score (Y)
   - Quadrants: High Activity/High Focus, High Activity/Low Focus, etc.
3. **Bar Chart**: Work style distribution (Highly Focused, Focused, Multi-project)
4. **Table**: Developer profiles with Activity Score, Focus Score, Time Pattern

**Update Frequency**: Monthly

**Target Audience**: Engineering Managers, Team Leads

---

### 3. Process Health Dashboard

**Purpose**: Track process adherence and quality metrics.

**Data Sources**:
- `weekly_metrics_YYYYMMDD.csv`
- `qualitative_insights_YYYYMMDD.csv`

**Components**:

**KPI Cards**:
- **Ticket Coverage**: 78.4% (Target: 60-80%)
- **Commit Quality**: 45.2 words/message (✅ Detailed)
- **Untracked Work**: 21.6% (Mostly maintenance)

**Charts**:
1. **Line Chart**: Ticket coverage trend (12 weeks)
2. **Stacked Area Chart**: Classification trends over time
   - Features (blue)
   - Bug Fixes (red)
   - Maintenance (green)
3. **Bar Chart**: Untracked work breakdown (Features, Bug Fixes, Maintenance)
4. **Table**: Platform distribution (JIRA, GitHub, ClickUp, Linear)

**Update Frequency**: Weekly (for sprint retrospectives)

**Target Audience**: Scrum Masters, Engineering Managers

---

### 4. Velocity & Planning Dashboard

**Purpose**: Support sprint planning and capacity estimates.

**Data Sources**:
- `weekly_metrics_YYYYMMDD.csv`
- `developer_focus_YYYYMMDD.csv`
- `story_point_correlation_YYYYMMDD.csv` (if available)

**Components**:

**KPI Cards**:
- **Avg Velocity**: 40.5 commits/week
- **Story Points/Week**: 23 (if tracked)
- **Velocity Trend**: ↑ Growing (12% vs last month)

**Charts**:
1. **Line Chart**: Weekly velocity with 4-week moving average
2. **Bar Chart**: Velocity by developer (shows capacity distribution)
3. **Scatter Plot**: Story Points vs Commits (correlation analysis)
4. **Forecast**: Projected velocity for next 4 weeks (based on trend)

**Update Frequency**: Weekly (before sprint planning)

**Target Audience**: Scrum Masters, Product Managers

---

## Chart Recommendations by Metric

| Metric | Best Chart Type | Why | Example Setup |
|--------|----------------|-----|---------------|
| **Commit Velocity** | Line chart | Shows trends over time | X: Week, Y: Commits, Add trendline |
| **Developer Distribution** | Horizontal bar chart | Easy name comparison | X: Commits, Y: Developer, Sort descending |
| **Classification Mix** | Pie or donut chart | Shows proportions clearly | Values: Feature %, Bug %, Maintenance % |
| **Work Balance (Gini)** | Gauge or bullet chart | Single value vs threshold | Value: Gini, Threshold: 0.3 |
| **Ticket Coverage Trend** | Line chart with bands | Shows target range | Y: Coverage %, Add 60% and 80% reference lines |
| **Developer × Project** | Heatmap | Multi-dimensional view | Rows: Developers, Cols: Projects, Color: Commits |
| **Activity Score** | Scatter plot | Shows clusters/outliers | X: Activity Score, Y: Focus Score |
| **Weekly Trends** | Stacked area chart | Shows composition over time | X: Week, Y: Commits, Stack: Classification |

## Advanced Techniques

### 1. Automatic Weekly Refresh (Google Sheets)

**Use Case**: Import latest CSV automatically each week.

**Setup**:
1. Upload CSV to Google Drive (same location each week)
2. In Google Sheets, use `IMPORTDATA()`:
   ```
   =IMPORTDATA("https://drive.google.com/your-csv-url")
   ```
3. Share Drive folder with team (view-only)
4. Update CSV file weekly (overwrite with same name)

**Result**: Dashboard auto-refreshes when CSV updates.

### 2. Comparison Dashboard (This Month vs Last Month)

**Setup**:
1. Import two periods: `weekly_metrics_current.csv` and `weekly_metrics_previous.csv`
2. Create calculated columns:
   - `Change = Current - Previous`
   - `% Change = (Current - Previous) / Previous`
3. Add **conditional formatting**:
   - Green: Positive change (e.g., ticket coverage improving)
   - Red: Negative change (e.g., velocity declining)

**Result**: Side-by-side comparison with trend indicators.

### 3. Executive One-Pager (PowerPoint/Keynote)

**Use Case**: Quarterly review presentation.

**Setup**:
1. Copy charts from Excel/Sheets
2. Paste into PowerPoint/Keynote
3. Add context text boxes:
   - "Ticket coverage improved 15% this quarter"
   - "Team velocity stable at 40 commits/week"
4. Highlight **key insights** with callout boxes

**Result**: Executive-ready presentation slide.

### 4. Rolling 4-Week Average (Smoothing Volatility)

**Use Case**: Remove weekly spikes for clearer trends.

**Excel Formula**:
```excel
=AVERAGE(B2:B5)  // In cell C5 (rolling avg of weeks 2-5)
```

Drag formula down for each week.

**Google Sheets Formula**:
```
=AVERAGE(B2:B5)
```

**Result**: Smoothed velocity trend showing true pattern.

## Tools & Platforms Comparison

### Google Sheets (Free)

**Pros**:
- ✅ Free, collaborative
- ✅ Cloud-based (access anywhere)
- ✅ Easy sharing with team
- ✅ `IMPORTDATA()` for auto-refresh

**Cons**:
- ❌ Limited advanced features (vs Excel)
- ❌ Performance issues with large datasets (>10K rows)

**Best For**: Small teams, simple dashboards, collaborative tracking

---

### Microsoft Excel (Paid)

**Pros**:
- ✅ Powerful features (PivotTables, advanced charts)
- ✅ Offline access
- ✅ Handles large datasets well
- ✅ Familiar to most users

**Cons**:
- ❌ License cost ($70-150/year)
- ❌ Not collaborative (unless using Office 365)
- ❌ Manual CSV import (no auto-refresh without macros)

**Best For**: Offline analysis, advanced users, large datasets

---

### Tableau (Paid)

**Pros**:
- ✅ Beautiful, interactive dashboards
- ✅ Advanced analytics (forecasting, clustering)
- ✅ Publish to web for stakeholders
- ✅ Real-time data connections

**Cons**:
- ❌ Expensive ($70+/month per user)
- ❌ Steep learning curve
- ❌ Overkill for simple dashboards

**Best For**: Enterprise dashboards, executive presentations, advanced analytics

---

### Looker/Metabase (Varies)

**Pros**:
- ✅ Database-connected (auto-refresh)
- ✅ Shareable dashboards
- ✅ SQL-based (customizable)

**Cons**:
- ❌ Requires database setup (can't use CSVs directly)
- ❌ Technical setup needed
- ❌ Metabase free, Looker expensive

**Best For**: Engineering-integrated BI, teams with data infrastructure

---

### Power BI (Microsoft)

**Pros**:
- ✅ Powerful like Tableau
- ✅ Free tier available (Power BI Desktop)
- ✅ Integrates with Microsoft ecosystem

**Cons**:
- ❌ Windows-only (Desktop version)
- ❌ Steeper learning curve than Excel

**Best For**: Microsoft shops, Windows users, enterprise teams

## Dashboard Automation Tips

### Weekly Report Cadence

**Workflow**:
1. Run GitFlow Analytics every Monday:
   ```bash
   gitflow-analytics -c config.yaml --weeks 4
   ```
2. Import new CSVs to Google Drive (overwrites previous)
3. Dashboard auto-refreshes (if using `IMPORTDATA()`)
4. Share dashboard link in weekly team meeting

**Time Savings**: 15 minutes/week (vs manual import + chart updates)

### Monthly Health Check

**Workflow**:
1. Run analysis with longer period (12 weeks):
   ```bash
   gitflow-analytics -c config.yaml --weeks 12
   ```
2. Compare to previous month's report
3. Create comparison dashboard (this month vs last)
4. Present in monthly all-hands

**Focus**: Trends, not daily/weekly volatility

### Quarterly Business Review (QBR)

**Workflow**:
1. Run analysis for full quarter (12-16 weeks)
2. Create executive one-pager:
   - Key metrics summary
   - Major trend charts
   - Recommendations for next quarter
3. Export to PowerPoint/PDF
4. Present to leadership

**Output**: Data-driven insights for quarterly planning

## Troubleshooting

### Issue: CSV Import Fails

**Cause**: Encoding issues or malformed CSV.

**Fix**:
- **Google Sheets**: Use "UTF-8" encoding on import
- **Excel**: Data → From Text/CSV → Set encoding to "Unicode (UTF-8)"

### Issue: Charts Show Incorrect Data

**Cause**: Column headers not recognized.

**Fix**:
- Ensure first row is headers (Week, Commits, etc.)
- Re-create chart and manually select correct columns

### Issue: Dates Not Sorting Correctly

**Cause**: Dates imported as text.

**Fix**:
- **Google Sheets**: Format → Number → Date
- **Excel**: Right-click column → Format Cells → Date

### Issue: Percentage Metrics Show as Decimals

**Cause**: Ticket Coverage imported as 0.78 instead of 78%.

**Fix**:
- **Google Sheets**: Format → Number → Percent
- **Excel**: Right-click column → Format Cells → Percentage

## Next Steps

### Beginner Path
1. ✅ Import `weekly_metrics.csv` to Google Sheets
2. ✅ Create velocity line chart (5 minutes)
3. ✅ Add ticket coverage chart with benchmark line
4. ✅ Share dashboard link with team

### Intermediate Path
1. ✅ Create Executive Summary Dashboard with multiple CSVs
2. ✅ Set up conditional formatting for health indicators
3. ✅ Add PivotTables for developer summaries
4. ✅ Implement auto-refresh with `IMPORTDATA()`

### Advanced Path
1. ✅ Create Team Health Dashboard with heatmaps
2. ✅ Build comparison dashboard (this month vs last)
3. ✅ Export to Tableau/Power BI for advanced analytics
4. ✅ Integrate with your existing BI infrastructure

## Related Documentation

- **[Report Interpretation Guide](interpreting-reports.md)** - Understand the data before visualizing
- **[Metrics Reference](metrics-reference.md)** - Definitions and benchmarks
- **[Quick Start Guide](quickstart.md)** - Getting your first reports

---

**Questions?** See the [FAQ](faq.md) or ask your technical lead for CSV file locations.
