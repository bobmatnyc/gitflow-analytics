# Manual Test Script for GitFlow Analytics Install Wizard

This document provides step-by-step manual test procedures for components that require interactive user input.

---

## Test 2: GitHub Setup Tests

### Test 2.1: Valid GitHub Token ✅ READY TO TEST

**Command**:
```bash
cd /Users/masa/Projects/managed/gitflow-analytics/test-qa-install
gitflow-analytics install --output-dir test-valid-github
```

**Steps**:
1. When prompted for GitHub token, enter a valid PAT
2. Wait for validation
3. Verify authenticated username is displayed
4. Continue through remaining prompts (use minimal setup)

**Expected Results**:
- ✅ Token validation succeeds
- ✅ Shows authenticated username
- ✅ Proceeds to next step
- ✅ No credential leakage in output

**Verification**:
```bash
# Check files were created
ls -la test-valid-github/
cat test-valid-github/config.yaml | grep "GITHUB_TOKEN"  # Should show ${GITHUB_TOKEN}
cat test-valid-github/.env | grep "GITHUB_TOKEN="  # Should have actual token
stat -f "%OLp" test-valid-github/.env  # Should be 600
```

---

### Test 2.2: Invalid GitHub Token ✅ READY TO TEST

**Command**:
```bash
gitflow-analytics install --output-dir test-invalid-github
```

**Steps**:
1. Enter invalid token: `invalid_token_123456`
2. Observe error handling
3. When asked to retry, select "n" (no)

**Expected Results**:
- ✅ Validation fails with clear error
- ✅ No raw exception displayed
- ✅ Offers retry (max 3 attempts)
- ✅ Installation fails gracefully after decline

**Security Check**:
- ❌ FAIL if actual token appears in error message
- ❌ FAIL if raw exception displayed

---

### Test 2.3: GitHub Token Retry Flow ✅ READY TO TEST

**Command**:
```bash
gitflow-analytics install --output-dir test-retry-github
```

**Steps**:
1. First attempt: invalid token
2. When asked to retry, select "y" (yes)
3. Observe retry delay (should be 1 second)
4. Second attempt: valid token

**Expected Results**:
- ✅ Retry delay shown (1 second for first retry)
- ✅ User can correct mistake
- ✅ Successful on second attempt

---

## Test 3: Repository Configuration Tests

### Test 3.1: Organization Mode ✅ READY TO TEST

**Command**:
```bash
gitflow-analytics install --output-dir test-org-mode
```

**Steps**:
1. Complete GitHub setup with valid token
2. Select mode "A" (Organization)
3. Enter organization name (e.g., "anthropics" or known org)
4. Continue through setup

**Expected Results**:
- ✅ Organization validated against GitHub API
- ✅ Shows repository count estimate
- ✅ config.yaml includes `organization: <org_name>`
- ✅ No `repositories` section in config

**Verification**:
```bash
grep "organization:" test-org-mode/config.yaml
grep -c "repositories:" test-org-mode/config.yaml  # Should be 0
```

---

### Test 3.2: Manual Repository Mode ✅ READY TO TEST

**Command**:
```bash
gitflow-analytics install --output-dir test-manual-repos
```

**Steps**:
1. Complete GitHub setup
2. Select mode "B" (Manual)
3. Enter valid repository path (use test repos)
4. When asked for another, decline (n)

**Expected Results**:
- ✅ Path validated for existence
- ✅ Checks for .git directory
- ✅ config.yaml includes `repositories:` list
- ✅ No `organization` field in config

**Verification**:
```bash
grep "repositories:" test-manual-repos/config.yaml
grep -c "organization:" test-manual-repos/config.yaml  # Should be 0
```

---

### Test 3.3: Invalid Repository Path (Security) ✅ TESTED AUTOMATICALLY

**Status**: ✅ PASSED in automated tests
**Result**: Path traversal properly blocked for `/etc/passwd`

---

## Test 4: JIRA Integration Tests

### Test 4.1: JIRA Enabled with Valid Credentials ⚠️ REQUIRES JIRA ACCESS

**Command**:
```bash
gitflow-analytics install --output-dir test-jira-valid
```

**Steps**:
1. Complete GitHub setup
2. Complete repository setup (use manual mode, skip repos)
3. Enable JIRA integration (y)
4. Enter valid JIRA credentials:
   - Base URL: https://your-instance.atlassian.net
   - Email: your.email@company.com
   - API Token: (from https://id.atlassian.com/manage-profile/security/api-tokens)

**Expected Results**:
- ✅ JIRA validation succeeds
- ✅ Story point fields discovered
- ✅ config.yaml includes `pm.jira` section with `${JIRA_*}` placeholders
- ✅ .env includes JIRA credentials

**Verification**:
```bash
grep "jira:" test-jira-valid/config.yaml
grep "JIRA_BASE_URL=" test-jira-valid/.env
grep "JIRA_ACCESS_USER=" test-jira-valid/.env
grep "JIRA_ACCESS_TOKEN=" test-jira-valid/.env
```

---

### Test 4.2: JIRA Disabled ✅ READY TO TEST

**Command**:
```bash
gitflow-analytics install --output-dir test-no-jira
```

**Steps**:
1. Complete GitHub setup
2. Complete repository setup
3. Decline JIRA integration (n)

**Expected Results**:
- ✅ JIRA section skipped
- ✅ No JIRA credentials in .env
- ✅ Installation proceeds normally

**Verification**:
```bash
grep -c "jira:" test-no-jira/config.yaml  # Should be 0
grep -c "JIRA" test-no-jira/.env  # Should be 0
```

---

### Test 4.3: JIRA Invalid Credentials ⚠️ REQUIRES TESTING

**Command**:
```bash
gitflow-analytics install --output-dir test-jira-invalid
```

**Steps**:
1. Enable JIRA
2. Enter invalid credentials (bad base_url or token)

**Expected Results**:
- ✅ Validation fails with generic error
- ✅ No credential exposure in error
- ✅ Offers retry (max 3)
- ✅ Can skip after failures

**Security Check**:
- ❌ FAIL if credentials appear in error messages
- ❌ FAIL if authorization headers logged

---

## Test 5: AI Integration Tests

### Test 5.1: OpenRouter Enabled ⚠️ REQUIRES API KEY

**Command**:
```bash
gitflow-analytics install --output-dir test-openrouter
```

**Steps**:
1. Complete GitHub/repo setup
2. Skip JIRA (n)
3. Enable AI (y)
4. Enter OpenRouter API key (starts with "sk-or-")

**Expected Results**:
- ✅ API key detected as OpenRouter
- ✅ Validation succeeds
- ✅ config.yaml includes `chatgpt.api_key: ${OPENROUTER_API_KEY}`
- ✅ .env includes `OPENROUTER_API_KEY=<key>`

**Verification**:
```bash
grep "api_key: \${OPENROUTER_API_KEY}" test-openrouter/config.yaml
grep "OPENROUTER_API_KEY=" test-openrouter/.env
```

---

### Test 5.2: OpenAI Key (non-OpenRouter) ⚠️ REQUIRES API KEY

**Command**:
```bash
gitflow-analytics install --output-dir test-openai
```

**Steps**:
1. Enable AI
2. Enter OpenAI API key (starts with "sk-" but not "sk-or-")

**Expected Results**:
- ✅ Detected as OpenAI (not OpenRouter)
- ✅ Uses correct validation endpoint
- ✅ Config and .env properly generated

---

### Test 5.3: AI Disabled ✅ READY TO TEST

**Command**:
```bash
gitflow-analytics install --output-dir test-no-ai
```

**Steps**:
1. Decline AI (n)

**Expected Results**:
- ✅ AI section skipped
- ✅ No AI credentials in .env

---

## Test 9: End-to-End Integration Test

### Test 9.1: Complete Install and Run Analysis ⚠️ COMPREHENSIVE TEST

**Command**:
```bash
gitflow-analytics install --output-dir test-e2e
```

**Steps**:
1. Complete full installation:
   - GitHub: valid token
   - Repositories: organization mode OR manual with test repos
   - JIRA: enabled with valid credentials (optional)
   - AI: enabled with valid key (optional)
   - Analysis: 4 weeks, default directories
2. Validate configuration:
   ```bash
   gitflow-analytics analyze -c test-e2e/config.yaml --validate-only
   ```
3. Run actual analysis (if validation passes):
   ```bash
   gitflow-analytics analyze -c test-e2e/config.yaml --weeks 4
   ```

**Expected Results**:
- ✅ Installation completes successfully
- ✅ Config validation passes
- ✅ Analysis runs without errors
- ✅ Reports generated in output directory

**Verification**:
```bash
ls -la test-e2e/
ls -la test-e2e/reports/
```

---

## Test 10: Error Handling Tests

### Test 10.1: Keyboard Interrupt (Ctrl+C) ✅ READY TO TEST

**Command**:
```bash
gitflow-analytics install --output-dir test-interrupt
```

**Steps**:
1. Start installation
2. During any prompt, press Ctrl+C

**Expected Results**:
- ✅ Graceful exit with cleanup message
- ✅ No partial files left behind
- ✅ No error traceback

---

### Test 10.2: Invalid Input Types ✅ READY TO TEST

**Command**:
```bash
gitflow-analytics install --output-dir test-invalid-input
```

**Steps**:
1. Complete to analysis configuration
2. When asked for "Analysis period (weeks)", enter "abc" (invalid)
3. Observe error handling

**Expected Results**:
- ✅ Type validation with re-prompting
- ✅ Clear error message
- ✅ No crashes

---

### Test 10.3: Network Timeouts (SIMULATION) ⏭️ SKIP

**Note**: Difficult to simulate without network manipulation. Can test by:
1. Disconnecting network
2. Attempting GitHub validation
3. Observing timeout behavior

**Expected Results**:
- ✅ Timeout error caught
- ✅ User-friendly message (no raw exception)
- ✅ Offers retry
- ✅ Can abort cleanly

---

## Test Execution Checklist

### Core Functionality
- [✅] Test 1: Command availability
- [MANUAL] Test 2.1: Valid GitHub token
- [MANUAL] Test 2.2: Invalid GitHub token
- [MANUAL] Test 2.3: Retry flow
- [MANUAL] Test 3.1: Organization mode
- [MANUAL] Test 3.2: Manual repository mode
- [MANUAL] Test 4.2: JIRA disabled
- [MANUAL] Test 5.3: AI disabled

### Security
- [✅] Test 6.1: File permissions
- [✅] Test 3.3: Path traversal protection
- [✅] Test 8.1: Exception sanitization
- [✅] Test 8.2: Memory clearing
- [✅] Test 8.3: Logging suppression (code review)

### Integration
- [✅] Test 6.2: .gitignore update
- [✅] Test 7.1: Config validation
- [MANUAL] Test 9.1: End-to-end workflow

### Robustness
- [MANUAL] Test 10.1: Keyboard interrupt
- [MANUAL] Test 10.2: Invalid inputs

---

## Test Execution Notes

**Automated Tests Status**: ✅ PASSED (13/13 tests)

**Manual Tests Required**:
- Basic flows (GitHub, org/manual mode, skip options): ~10 minutes
- With credentials (JIRA, AI): ~5 minutes if credentials available
- End-to-end: ~5 minutes with valid setup

**Total Estimated Time**: 20-25 minutes for full manual testing

**Recommendation**:
- Run automated tests first (completed)
- Execute basic manual flows without external dependencies
- Optional: Test with real credentials if available
- Document any issues found
