# Commit Message for Version 1.2.1

fix: critical bug fixes for two-step process and LLM classification

## Fixed Issues
- Fixed commits not being stored in CachedCommit table during fetch step
  - Added code to store individual commits when creating daily batches
  - Fixed schema mismatch (removed non-existent project_key field)

- Fixed narrative report generation when CSV is disabled
  - Added canonical_id to commits loaded from database
  - Updated identity resolver to set canonical_id on commits
  - Fixed None value handling for complexity_delta field

- Fixed LLM classification functionality
  - Added missing classify_commits_batch method to LLMCommitClassifier
  - Fixed timezone-aware datetime comparisons in batch classifier
  - Properly loads API keys from .env files in config directory

## New Features
- Added token tracking and cost display for LLM classification
  - Shows model, API calls, total tokens used, and costs
  - Displays cache hit statistics
  - Integrated with batch classification output

## Testing Completed
- ✅ Verified commits are stored in database (32 commits stored)
- ✅ Verified narrative reports generate without CSV files
- ✅ Verified LLM classification works with real API key from .env
- ✅ Verified token tracking displays correctly ($0.0001 for 474 tokens)
- ✅ Verified cache functionality (2 cached predictions)

## Version Update
- Updated version from 1.2.0 to 1.2.1
- Updated CHANGELOG.md with all fixes and improvements

This patch release ensures the two-step process (fetch then classify) works correctly
with proper data persistence, report generation, and LLM classification with cost tracking.