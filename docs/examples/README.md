# Usage Examples

Real-world configuration examples and usage scenarios for GitFlow Analytics.

## 🎯 Example Categories

### [Basic Analysis](basic-analysis.md)
Simple single-repository analysis perfect for:
- Individual developers analyzing their work
- Small teams with one main repository  
- Quick health checks and productivity insights
- Learning GitFlow Analytics fundamentals

### [Enterprise Setup](enterprise-setup.md)  
Large-scale organization analysis featuring:
- Multi-repository organization discovery
- Advanced identity resolution strategies
- Performance optimization for 100+ repositories
- Enterprise reporting and data export

### [CI Integration](ci-integration.md)
Continuous integration workflows including:
- GitHub Actions automated analysis
- GitLab CI pipeline integration  
- Jenkins job configuration
- Automated report generation and distribution

### [Custom Workflows](custom-workflows.md)
Advanced usage patterns for specialized needs:
- Custom commit categorization rules
- Specialized report formats
- Integration with external tools
- Advanced filtering and analysis focus

## 🚀 Quick Start Examples

### Minimal Configuration
```yaml
# quickstart-config.yaml
github:
  token: "${GITHUB_TOKEN}"
  repositories:
    - owner: "myorg"
      name: "myrepo"
      local_path: "./myrepo"
```

### Organization Discovery  
```yaml
# org-config.yaml
github:
  token: "${GITHUB_TOKEN}"
  organization: "myorg"  # Discovers all repositories
  
analysis:
  weeks: 8
```

### ML-Enhanced Analysis
```yaml
# ml-config.yaml
github:
  token: "${GITHUB_TOKEN}"
  organization: "myorg"

analysis:
  weeks: 12
  enable_ml_categorization: true
  
ml_categorization:
  model_name: "en_core_web_sm"
  confidence_threshold: 0.7
```

## 📁 Configuration File Templates

All examples include complete, working configuration files located in `/examples/config/`:

- **config-sample.yaml** - Basic single-repository setup
- **config-organization.yaml** - Organization-wide analysis  
- **config-ml-enhanced.yaml** - ML categorization enabled
- **config-enterprise.yaml** - Large-scale deployment
- **config-minimal.yaml** - Simplest possible configuration

## 🎯 Choose Your Example

### By Team Size

**Individual Developer (1 person)**
→ [Basic Analysis](basic-analysis.md) - Perfect for personal productivity insights

**Small Team (2-10 people)**  
→ [Basic Analysis](basic-analysis.md) with multiple repositories

**Medium Team (10-50 people)**
→ [Enterprise Setup](enterprise-setup.md) with organization discovery

**Large Organization (50+ people)**
→ [Enterprise Setup](enterprise-setup.md) with performance optimization

### By Use Case

**First-time setup**
→ [Basic Analysis](basic-analysis.md) - Start simple and expand

**Regular team health checks**
→ [CI Integration](ci-integration.md) - Automated recurring analysis  

**Quarterly planning and reviews**
→ [Enterprise Setup](enterprise-setup.md) - Comprehensive insights

**Custom reporting needs**
→ [Custom Workflows](custom-workflows.md) - Specialized configurations

### By Technical Needs

**Simple CSV reports**
→ [Basic Analysis](basic-analysis.md) - Standard output formats

**JSON data export**  
→ [Custom Workflows](custom-workflows.md) - API integration patterns

**ML commit categorization**
→ [Enterprise Setup](enterprise-setup.md) - Advanced analysis features

**Automated workflows**
→ [CI Integration](ci-integration.md) - Pipeline integration

## 💡 Example Usage Patterns

### Development Workflow
1. **Start with Basic**: Use basic analysis to understand your repository
2. **Add Features**: Gradually enable ML categorization and advanced options  
3. **Scale Up**: Move to organization-wide analysis as needed
4. **Automate**: Integrate with CI/CD for regular insights

### Analysis Cadence
```yaml
# Weekly sprint reviews
analysis:
  weeks: 2
  
# Monthly team health  
analysis:
  weeks: 4
  
# Quarterly planning
analysis:
  weeks: 12
  
# Annual reviews
analysis:
  weeks: 52
```

## 🔧 Customization Examples

### Focus on Specific Areas
```yaml
analysis:
  # Only analyze Python and JavaScript
  include_file_patterns:
    - "*.py"
    - "*.js" 
    - "*.ts"
    
  # Exclude generated files
  exclude_file_patterns:
    - "*.min.js"
    - "package-lock.json"
    - "__pycache__/*"
```

### Custom Report Formats
```yaml
reports:
  formats: ["csv", "json", "markdown"]
  output_directory: "./reports"
  
  # Custom naming
  filename_template: "team_analysis_{date}"
  
  # Include additional data
  include_untracked_analysis: true
  include_detailed_metrics: true
```

## 📊 Expected Outputs

Each example shows expected report structure and key insights you'll receive:

- **CSV Reports**: Structured data for analysis and integration
- **Markdown Reports**: Human-readable insights and recommendations
- **JSON Exports**: Complete data for custom tooling and dashboards

## 🆘 Common Adaptations

### Adapting Examples for Your Environment

1. **Repository URLs**: Replace example repositories with your own
2. **Authentication**: Set up your GitHub token and credentials
3. **Time Periods**: Adjust `weeks` parameter for your analysis needs
4. **Output Locations**: Customize paths for your directory structure
5. **Team Structure**: Modify identity mappings for your developers

### Testing New Configurations

```bash
# Validate configuration without running analysis
gitflow-analytics -c your-config.yaml --validate-only

# Test with shorter time period first
gitflow-analytics -c your-config.yaml --weeks 2

# Clear cache if making major changes
gitflow-analytics -c your-config.yaml --clear-cache
```

## 🔄 Related Documentation

- **[Getting Started](../getting-started/)** - Installation and first steps
- **[Configuration Guide](../guides/configuration.md)** - Complete YAML reference
- **[Troubleshooting](../guides/troubleshooting.md)** - Common issues and solutions
- **[Reference Documentation](../reference/)** - Technical specifications

## 📝 Contributing Examples

Have a useful configuration or workflow? Consider contributing it:

1. Create a new example following existing patterns
2. Include complete working configuration
3. Document expected outputs and key insights
4. Test with real data before submitting
5. See [Contributing Guide](../developer/contributing.md) for details

Start with an example that matches your situation and customize from there!