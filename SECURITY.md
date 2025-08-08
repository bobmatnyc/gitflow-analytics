# Security Policy

## Reporting Security Vulnerabilities

We take the security of GitFlow Analytics seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities using one of these methods:
- **GitHub Security Advisories**: Use the "Security" tab in the GitHub repository to privately report vulnerabilities
- **Email**: Contact the maintainers directly through their GitHub profiles
- **Subject Line**: `[SECURITY] GitFlow Analytics - [Brief Description]`

### What to Include

Please include as much of the following information as possible:

1. **Type of vulnerability** (e.g., code injection, authentication bypass, data exposure)
2. **Full paths** to source files related to the vulnerability
3. **Location** of the affected code (file path, line numbers, or git commit)
4. **Step-by-step instructions** to reproduce the vulnerability
5. **Impact** assessment - what data or systems could be affected
6. **Proof of concept** or exploit code (if available)
7. **Your assessment** of the severity

### Response Timeline

- **Initial Response**: We aim to acknowledge security reports within 48 hours
- **Status Update**: We will provide a more detailed response within 7 days
- **Resolution**: We will work to address critical vulnerabilities within 30 days

### Security Best Practices for Users

When using GitFlow Analytics, please follow these security recommendations:

#### 1. Credential Management

**✅ Do:**
- Store API tokens in `.env` files (never commit to version control)
- Use environment variables for sensitive configuration
- Add `.env` to your `.gitignore` file
- Use GitHub Personal Access Tokens with minimal required scopes
- Rotate API tokens regularly

**❌ Don't:**
- Hard-code tokens or passwords in YAML configuration files
- Commit `.env` files to version control  
- Use overly broad API token permissions
- Share configuration files containing credentials

**Example secure configuration:**
```yaml
# config.yaml - safe to commit
github:
  token: "${GITHUB_TOKEN}"  # References environment variable
  organization: "myorg"

# .env - NEVER commit this file
GITHUB_TOKEN=ghp_secureTokenValue123
```

#### 2. Data Privacy

**Repository Access:**
- Ensure your GitHub tokens only have access to repositories you intend to analyze
- Use organization-scoped tokens when possible
- Review repository access permissions regularly

**Report Anonymization:**
- Use the `--anonymize` flag when sharing reports externally
- Be cautious about committing generated reports to public repositories
- Review reports for sensitive information before sharing

**Example:**
```bash
# Generate anonymized reports for external sharing
gitflow-analytics -c config.yaml --anonymize --output ./public-reports
```

#### 3. Network Security

**API Communications:**
- All API communications use HTTPS by default
- Verify SSL certificates are properly validated
- Consider using VPN or secure networks when analyzing sensitive repositories

#### 4. Local Data Security

**Cache and Database Files:**
- Cache files (`.gitflow-cache/`) contain analyzed repository data
- Ensure proper file permissions on cache directories
- Consider encrypting cache directories for sensitive projects
- Regularly clean up old cache files

**File Permissions:**
```bash
# Secure cache directory permissions (Unix/Linux/macOS)
chmod 700 .gitflow-cache/
chmod 600 .gitflow-cache/*.db
```

#### 5. Configuration Validation

**Input Validation:**
- The tool validates YAML configuration but always review configuration files
- Be cautious with regex patterns that could cause performance issues
- Validate repository paths before analysis

#### 6. Third-party Dependencies

**Dependency Security:**
- Keep GitFlow Analytics updated to the latest version
- Monitor security advisories for dependencies
- Use tools like `pip-audit` to check for vulnerable dependencies:

```bash
# Check for security vulnerabilities in dependencies
pip install pip-audit
pip-audit
```

### Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | ✅ Yes             |
| < 1.0   | ❌ No              |

### Known Security Considerations

#### 1. API Rate Limiting

- GitHub API has rate limits that could be triggered by analysis
- Malicious repository configurations could cause excessive API calls
- The tool implements rate limiting and backoff strategies

#### 2. Repository Data Exposure

- The tool analyzes commit messages and metadata which may contain sensitive information
- Generated reports include repository statistics and developer information
- Use anonymization features when sharing reports

#### 3. ML Model Dependencies

- The ML categorization feature uses spaCy models
- Models are downloaded from spaCy's official repositories
- Consider verifying model checksums in high-security environments

#### 4. Local File Access

- The tool requires read access to local Git repositories
- Ensure repository paths are properly validated and sandboxed
- Be cautious when analyzing untrusted repositories

### Security Disclosure Policy

- We will acknowledge valid security reports within 48 hours
- We will provide regular updates on the status of security fixes
- We will credit security researchers (with their permission) in our security advisories
- We may request that you keep the vulnerability confidential until we have had a chance to address it

### Bug Bounty Program

We do not currently offer a formal bug bounty program, but we greatly appreciate security research and responsible disclosure.

### Security Updates

Security updates will be:
- Released as patch versions (e.g., 1.0.5 → 1.0.6)
- Documented in release notes and security advisories
- Communicated through GitHub security advisories
- Available immediately via PyPI

### Contact Information

For security-related questions or concerns:
- **Security Issues**: Report via email (see "How to Report" section above)  
- **General Questions**: Use GitHub Discussions for non-security questions
- **Documentation**: Check README.md and CONTRIBUTING.md first

### Acknowledgments

We thank the security research community for helping keep GitFlow Analytics safe and secure.

---

**Note**: This security policy may be updated from time to time. Please check back periodically for any changes.