# Research Agent Memory - gitflow-analytics

<!-- MEMORY LIMITS: 16KB max | 10 sections max | 15 items per section -->
<!-- Last Updated: 2025-08-05 15:36:32 | Auto-updated by: research -->

## Project Architecture (Max: 15 items)
- Service-oriented architecture with clear module boundaries
- Three-tier agent hierarchy: project → user → system
- Agent definitions use standardized JSON schema validation

## Coding Patterns Learned (Max: 15 items)
- Always use PathResolver for path operations, never hardcode paths
- SubprocessRunner utility for external command execution
- LoggerMixin provides consistent logging across all services

## Implementation Guidelines (Max: 15 items)
- Check docs/STRUCTURE.md before creating new files
- Follow existing import patterns: from claude_mpm.module import Class
- Use existing utilities instead of reimplementing functionality

## Domain-Specific Knowledge (Max: 15 items)
<!-- Agent-specific knowledge accumulates here -->

## Effective Strategies (Max: 15 items)
<!-- Successful approaches discovered through experience -->

## Common Mistakes to Avoid (Max: 15 items)
- Don't modify Claude Code core functionality, only extend it
- Avoid duplicating code - check utils/ for existing implementations
- Never hardcode file paths, use PathResolver utilities

## Integration Points (Max: 15 items)
<!-- Key interfaces and integration patterns -->

## Performance Considerations (Max: 15 items)
<!-- Performance insights and optimization patterns -->

## Current Technical Context (Max: 15 items)
- EP-0001: Technical debt reduction in progress
- Target: 80% test coverage (current: 23.6%)
- Integration with Claude Code 1.0.60+ native agent framework

## Recent Learnings (Max: 15 items)
<!-- Most recent discoveries and insights -->
