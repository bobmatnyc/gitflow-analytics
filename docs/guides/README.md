# User Guides

Comprehensive guides for configuring and using GitFlow Analytics effectively.

## 🎯 Configuration & Setup

### [Complete Configuration Guide](configuration.md)
Master the YAML configuration format with detailed examples covering:
- GitHub authentication and repository setup
- Organization-wide repository discovery
- Identity resolution and developer mapping
- Advanced analysis options and filters
- Report customization and output formats

### [PM Platform Setup](pm-platform-setup.md)
Configure multiple project management platforms for ticket tracking:
- **JIRA** - Enterprise project management integration
- **Linear** - Modern issue tracking setup
- **ClickUp** - All-in-one productivity platform
- **GitHub Issues** - Native GitHub integration
- Multi-platform configuration and credential management
- Story point extraction and ticket coverage analysis

### [ML Categorization Setup](ml-categorization.md)
Enable machine learning features for intelligent commit classification:
- Install and configure spaCy language models
- Set up ML-powered commit categorization (85-95% accuracy)
- Customize classification categories and thresholds
- Monitor and improve model performance

### [Organization-Wide Setup](organization-setup.md)
Scale GitFlow Analytics across multiple repositories:
- Automatic repository discovery from GitHub organizations
- Bulk configuration for enterprise environments
- Cross-repository reporting and insights
- Performance optimization for large-scale analysis

## 🛠️ Advanced Features

### [Report Customization](report-customization.md)
Tailor reports to your team's needs:
- Choose output formats (CSV, JSON, Markdown, HTML)
- Custom report templates and branding
- Filter and focus analysis on specific areas
- Integration with other tools and dashboards

### [ChatGPT Integration](chatgpt-setup.md)
Enhance insights with AI-powered qualitative analysis:
- Set up OpenAI API integration
- Configure qualitative analysis parameters
- Generate deeper insights from commit patterns
- Balance cost with analysis depth

## 🔧 Maintenance & Operations  

### [Troubleshooting Guide](troubleshooting.md)
Solutions to common issues and problems:
- Installation and dependency issues
- GitHub API authentication problems
- Configuration validation errors
- Performance and memory optimization
- Error message explanations and fixes

## 🎯 Quick Navigation by Goal

### I want to...

**Analyze a single repository**
→ [Configuration Guide](configuration.md) → [Repository Setup Section](configuration.md#repository-configuration)

**Analyze my entire organization**
→ [Organization Setup](organization-setup.md) → [Organization Discovery](organization-setup.md#automatic-discovery)

**Track tickets from JIRA, Linear, or ClickUp**
→ [PM Platform Setup](pm-platform-setup.md) → [Platform Configuration](pm-platform-setup.md#platform-configuration)

**Get better commit categorization**
→ [ML Categorization](ml-categorization.md) → [Model Setup](ml-categorization.md#installation)

**Customize report formats**
→ [Report Customization](report-customization.md) → [Output Formats](report-customization.md#output-formats)

**Fix configuration issues**
→ [Troubleshooting](troubleshooting.md) → [Configuration Errors](troubleshooting.md#configuration-issues)

**Add AI insights**
→ [ChatGPT Setup](chatgpt-setup.md) → [API Configuration](chatgpt-setup.md#setup)

**Solve authentication problems**
→ [Troubleshooting](troubleshooting.md) → [GitHub API Issues](troubleshooting.md#github-authentication)

**Optimize for large repositories**
→ [Configuration Guide](configuration.md) → [Performance Settings](configuration.md#performance-optimization)

## 📚 Guide Difficulty Levels

- 🟢 **Beginner**: Configuration Guide, PM Platform Setup, Troubleshooting
- 🟡 **Intermediate**: Organization Setup, Report Customization
- 🔴 **Advanced**: ML Categorization, ChatGPT Integration

## 💡 Pro Tips

1. **Start Simple**: Begin with basic configuration, add advanced features gradually
2. **Use Examples**: Each guide includes working configuration examples
3. **Test Incrementally**: Use `--validate-only` flag to test configuration changes
4. **Monitor Performance**: Large analyses benefit from caching and batch processing
5. **Customize Gradually**: Default settings work well; customize based on specific needs

## 🔄 Related Documentation

- **[Getting Started](../getting-started/)** - New user onboarding
- **[Examples](../examples/)** - Real-world usage scenarios  
- **[Reference](../reference/)** - Technical specifications
- **[Architecture](../architecture/)** - System design details

Need help choosing where to start? Check the [Getting Started](../getting-started/) section first!