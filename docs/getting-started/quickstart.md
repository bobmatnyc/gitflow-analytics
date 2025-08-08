# Quick Start Tutorial

Get GitFlow Analytics running in 5 minutes with this hands-on tutorial.

## 🎯 What You'll Accomplish

By the end of this tutorial, you'll have:
- ✅ Created your first configuration file
- ✅ Analyzed a sample repository 
- ✅ Generated your first reports
- ✅ Understood the key output files

**Time required**: ~5 minutes

## 📋 Prerequisites

Before starting:
- ✅ GitFlow Analytics is [installed](installation.md)
- ✅ You have a GitHub personal access token
- ✅ You have command-line access

## 🚀 Step 1: Create Configuration

Create a simple configuration file to analyze a public repository:

```bash
# Create your first config file
cat > quickstart-config.yaml << EOF
github:
  token: "\${GITHUB_TOKEN}"
  repositories:
    - owner: "octocat"
      name: "Hello-World"
      local_path: "./hello-world"

analysis:
  # Analyze the last 4 weeks
  weeks: 4
  
# Save reports in current directory  
reports:
  output_directory: "./reports"
EOF
```

## 🔑 Step 2: Set GitHub Token

```bash
# Set your GitHub token (replace with your actual token)
export GITHUB_TOKEN="ghp_your_token_here"

# Or create a .env file
echo "GITHUB_TOKEN=ghp_your_token_here" > .env
```

## ⚡ Step 3: Run Your First Analysis

```bash
# Run the analysis (this will take 1-2 minutes)
gitflow-analytics -c quickstart-config.yaml

# Watch for output like:
# 🔍 Analyzing repositories...
# 📊 Processing commits...
# 📝 Generating reports...
# ✅ Analysis complete!
```

## 📊 Step 4: Explore the Results

After the analysis completes, you'll have several files in `./reports/`:

```bash
# View generated files
ls -la reports/

# Expected output:
# weekly_metrics_YYYYMMDD.csv      # Weekly developer metrics
# developers_YYYYMMDD.csv          # Developer profiles
# summary_YYYYMMDD.csv             # Project summary
# narrative_report_YYYYMMDD.md     # Comprehensive markdown report
```

### Key Files to Review

**1. Narrative Report** (most important)
```bash
# View the comprehensive markdown report
cat reports/narrative_report_*.md
```
This contains:
- Executive summary
- Team composition analysis  
- Development patterns
- Key insights and recommendations

**2. Developer Metrics**
```bash
# View developer-specific metrics
head -5 reports/developers_*.csv
```
Shows commits, projects, and activity patterns per developer.

**3. Weekly Trends**
```bash
# View weekly activity trends
head -5 reports/weekly_metrics_*.csv  
```
Tracks development velocity over time.

## 🎉 Success! What You Just Did

Congratulations! You've successfully:

1. **Configured GitFlow Analytics** with a YAML file
2. **Connected to GitHub** using personal access token
3. **Analyzed repository history** for the past 4 weeks
4. **Generated comprehensive reports** with insights

## 🔍 Understanding the Output

### Executive Summary
The narrative report starts with key metrics:
- Total commits analyzed
- Active developers identified  
- Primary programming languages
- Development velocity trends

### Developer Insights
For each developer, you'll see:
- Commit volume and percentage of total work
- Primary projects and focus areas
- Work patterns (focused vs. distributed)
- Contribution trends over time

### Project Analysis
The analysis reveals:
- Which projects are most active
- Developer distribution across projects
- Code change patterns and impact
- Collaboration indicators

## 🚨 Troubleshooting Quick Issues

**"No commits found"**
- The Hello-World repository is quite old; try analyzing a more recent repository
- Adjust the `weeks` parameter to analyze a longer time period

**"Authentication failed"**
- Double-check your GitHub token is set correctly
- Verify the token has necessary permissions (`repo` scope)

**"Repository not found"**  
- Ensure the repository owner/name are correct
- Check that you have access to private repositories (if analyzing private repos)

## 🎯 Key Takeaways

1. **Simple Configuration**: YAML files make setup straightforward
2. **Comprehensive Analysis**: Rich insights from just Git history
3. **Multiple Output Formats**: CSV for data, Markdown for readability
4. **No External Dependencies**: Works without JIRA, Linear, or other PM tools

## 🔄 What's Next?

Now that you've completed the quick start:

### Analyze Your Own Repository
```bash
# Edit the configuration to use your repository
vim quickstart-config.yaml

# Update the repository section:
repositories:
  - owner: "your-username"
    name: "your-repository"
    local_path: "./your-repo"
```

### Try Advanced Features
- **[ML Categorization](../guides/ml-categorization.md)** - Enable automatic commit classification
- **[Organization Analysis](../guides/organization-setup.md)** - Analyze multiple repositories
- **[Custom Reports](../guides/report-customization.md)** - Customize output formats

### Learn More
- **[Configuration Guide](../guides/configuration.md)** - Complete configuration reference
- **[Your First Analysis](first-analysis.md)** - Deeper dive into understanding output
- **[Examples](../examples/)** - Real-world configuration examples

## 🆘 Need Help?

- **Common Issues**: Check [Troubleshooting Guide](../guides/troubleshooting.md)
- **Configuration Questions**: See [Configuration Guide](../guides/configuration.md)
- **Bug Reports**: [GitHub Issues](https://github.com/bobmatnyc/gitflow-analytics/issues)

Great job completing the quick start! You're ready to explore more advanced GitFlow Analytics features.