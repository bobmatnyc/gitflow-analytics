# Documentation Archive

This directory contains archived documentation files that are no longer actively used but are preserved for historical reference and context.

## Purpose

The `_archive/` directory serves to:
- Preserve temporary documentation created during development
- Maintain historical context for major features and bug fixes
- Keep the root directory clean while preserving file history
- Provide reference material for understanding project evolution

## Organization Policy

### What Goes Here

Files that should be archived include:
- **Temporary implementation summaries** - Created during feature development
- **Bug resolution reports** - Detailed investigation and fix documentation
- **Feature test reports** - Testing documentation for completed features
- **Code analysis reports** - One-time code structure analyses
- **Migration guides** - Completed migration documentation
- **Deprecated guides** - Documentation for removed or replaced features

### What Stays Active

Files that should remain in active documentation:
- **User guides** - Ongoing usage documentation (docs/guides/)
- **API references** - Current API specifications (docs/reference/)
- **Architecture docs** - Current system design (docs/architecture/)
- **Getting started guides** - Onboarding documentation (docs/getting-started/)

## Retention Policy

- **Permanent retention**: All archived files are kept indefinitely in version control
- **No automatic deletion**: Files are never automatically removed from archive
- **Manual cleanup**: Archive review occurs during major version milestones
- **Git history preservation**: All files moved using `git mv` to maintain history

## Archive Structure

```
docs/_archive/
├── README.md                    # This file
├── temp-files/                  # Temporary documentation files
│   ├── README.md               # Index of temporary files
│   ├── code_analysis_report.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   └── ...
├── features/                    # Archived feature documentation
├── bugs/                        # Archived bug reports
└── migrations/                  # Archived migration guides
```

## Finding Archived Content

### By File Name
```bash
find docs/_archive -name "*.md" -type f
```

### By Content
```bash
grep -r "search term" docs/_archive/
```

### By Git History
```bash
git log --all --full-history -- "docs/_archive/filename.md"
```

## Archiving New Files

To archive a file while preserving git history:

```bash
# 1. Identify the appropriate archive subdirectory
# 2. Use git mv to preserve history
git mv path/to/file.md docs/_archive/subdirectory/

# 3. Update the archive index
# Edit docs/_archive/subdirectory/README.md

# 4. Commit the change
git commit -m "chore: archive filename.md"
```

## Cross-References

For current documentation organization, see:
- [Documentation Structure](../STRUCTURE.md) - Overall documentation organization
- [Main Documentation Index](../README.md) - Active documentation navigation
- [Project Organization](../reference/PROJECT_ORGANIZATION.md) - File organization standards

## Version History

- **2025-10-20**: Initial archive structure created with temp-files subdirectory
