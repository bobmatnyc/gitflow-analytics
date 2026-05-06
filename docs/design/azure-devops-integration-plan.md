# Azure DevOps PM Integration — Implementation Plan

**Status:** Draft (planning hive output, awaiting user decisions on §2 open questions)
**Target version:** `3.16.0` (originally reserved `3.15.0`, but that was consumed by Bob's PR #56 / story-points float widening on 2026-05-05)
**Estimated effort:** 8–11 engineering days (1 engineer); 6–7 days with parallel doc/test work
**Prior art:** `src/gitflow_analytics/pm_framework/adapters/jira_adapter.py` (713 LOC) + converters mixin (714 LOC) + cache (495 LOC) ≈ **1,920 LOC of reference implementation**

This plan synthesizes outputs from four parallel planning agents (API research, architecture, phasing, risk review). Where they disagreed, the disagreement is called out and a recommendation given.

---

## 1. Executive summary

Add `AzureDevOpsAdapter` parallel to the existing `JIRAAdapter` under `pm_framework/adapters/`, wire it into the orchestrator, config schema, ticket extractor, setup wizard, identity sync, and reports. Ship behind an opt-in config block; zero behavior change for existing JIRA users.

**Strong consensus across agents:**
- PAT-only auth for v1 (HTTP Basic, empty username, base64-encoded `:<PAT>`).
- Hand-rolled `requests` + `urllib3.Retry`, mirroring JIRA. Do **not** vendor the official `azure-devops` SDK.
- Three-file split: `azure_devops_adapter.py` + `azure_devops_cache.py` + `azure_devops_converters.py` (mixin).
- Two-stage fetch: WIQL (IDs only) → `POST /_apis/wit/workitemsbatch` (200 IDs/chunk).
- Default ticket regex: `r"AB#(\d+)"`. **Do not** use bare `r"#(\d+)"` — collides with GitHub PR/issue numbers.
- Map work-item status by **State Category** (`Proposed/InProgress/Resolved/Completed/Removed`), not state name. Custom processes (CMMI, inherited) rename states.
- Defer to v1.1 / v2: OAuth, Azure DevOps Server (on-prem), full identity sync, write operations, ADO test plans.
- Cloud only for v1 (`https://dev.azure.com/{org}` and the legacy `{org}.visualstudio.com`).

**The single highest-leverage finding** (review agent): adding ADO only to `pm_framework/adapters/` is **not enough**. There are ~40 hardcoded `"jira"` strings in legacy enrichment paths (`core/data_fetcher*`, `pipeline_collect.py`, `cli_fetch.py`, `core/data_fetcher_processing.py:328` literally has `platform="jira"  # Assuming JIRA for now`). If we ship ADO without addressing this, ADO data will be silently dropped on the legacy path. **Phase 1 must include a refactor of the platform-tag plumbing** before any ADO-specific code lands.

---

## 2. Open decisions (blocking — please confirm before Phase 1 starts)

These are the questions where agents diverged or where user input is required.

| # | Decision | Recommendation | Why it matters |
|---|----------|---------------|----------------|
| 1 | **Auth method** | PAT only for v1 | OAuth doubles complexity (callback server, token refresh). |
| 2 | **On-prem (ADO Server / TFS)** | Out of scope for v1; reject with clear error | NTLM/Kerberos + version skew + SSL self-signed = weeks of work. |
| 3 | **HTTP client** | Hand-rolled `requests` (mirror JIRA) | Official SDK lags releases and pulls ~25 transitive deps. |
| 4 | **Ticket regex default** | `r"AB#(\d+)"` only; document `(?:AB#|#)(\d+)` as opt-in override | Bare `#1234` collides with GitHub PR/issue refs in mixed-platform repos. |
| 5 | **Sprint enumeration** | Project-level **iteration classification node tree** (`/wit/classificationnodes/Iterations?$depth=10`), de-dup by full path | Iterations are team-scoped. Per-team enumeration multiplies API calls and creates duplicates; classification node tree is project-scoped and stable. (Architecture agent and reviewer disagreed; reviewer's project-tree approach wins on simplicity and parity.) |
| 6 | **Custom process support** | Ship Agile + Scrum + Basic + CMMI mapping tables out-of-the-box | Mapping tables are small and free at runtime. Inherited custom processes fall through to UNKNOWN with a config override hook. |
| 7 | **`UnifiedIssue.story_points` type** | **Landed in v3.15.0** (Bob's PR #56, commit `bea075f`). Model widened to `Optional[float]`, JIRA coercion fixed, SQL columns flipped to `REAL`. Phase 3 ADO converter inherits the corrected behavior. | ADO `Effort` (Scrum) and `StoryPoints` (Agile) are natively float; the precision bug is gone. |
| 8 | **Cache layout** | Keep per-platform SQLite for v1 (`azure_devops_tickets.db`), schedule unified `pm_tickets.db` for v2 | Reviewer flagged the per-platform DB sprawl as a design smell. Refactoring now would balloon scope; revisit before adding the 3rd platform. |
| 9 | **Native commit ↔ work-item links** | Use `POST /_apis/wit/artifactUriQuery` as a **secondary** correlation source on top of `AB#` message scanning. Both `correlation_method="ticket_reference"` and `correlation_method="native_link"` recorded; native wins on tie | Catches commits made via VS/VSCode "Link work item" UI without `AB#` in the message. |
| 10 | **Identity sync** | Defer full sync to v1.1; v1 ships an `ado-identity-doctor` diagnostic that lists unmatched assignees | ADO `uniqueName` (UPN, `alice@tenant.onmicrosoft.com`) frequently differs from git commit email. Auto-merge is its own project. |
| 11 | **Method naming** | Extend existing `Config.get_effective_ticket_platforms()` (`schema.py:778`); do not invent `get_pm_platforms` | Original brief used the wrong method name. |
| 12 | **Feature flag** | Ship behind `pm.azure_devops.enabled` (already required); do **not** add a second `azure_devops_v1_beta` flag | One opt-in is enough; double-gating creates support confusion. |

---

## 3. Architecture (condensed)

### 3.1 New files

| Path | Purpose |
|------|---------|
| `src/gitflow_analytics/pm_framework/adapters/azure_devops_adapter.py` | `AzureDevOpsAdapter` class, HTTP session, abstract method implementations (~600–750 LOC). |
| `src/gitflow_analytics/pm_framework/adapters/azure_devops_cache.py` | `AzureDevOpsTicketCache` (SQLite, mirror `JiraTicketCache`). |
| `src/gitflow_analytics/pm_framework/adapters/azure_devops_converters.py` | `AzureDevOpsConvertersMixin` — work-item → `UnifiedIssue`, identity ref normalization, type/state/priority mappers. |
| `src/gitflow_analytics/integrations/azure_devops_identity_sync.py` | **v1.1 stub** — logs "deferred to v2" until identity work lands. |
| `tests/pm_framework/adapters/test_azure_devops_adapter.py` | Auth, projects, WIQL, batch fetch, error paths. |
| `tests/pm_framework/adapters/test_azure_devops_converters.py` | Pure-function mapper tests, no network. |
| `tests/pm_framework/adapters/test_azure_devops_cache.py` | Cache TTL, eviction, schema. |
| `tests/pm_framework/adapters/test_azure_devops_get_issues.py` | End-to-end WIQL → batch flow with `responses`-mocked fixtures, including pagination. |
| `tests/pm_framework/adapters/test_azure_devops_links.py` | Native commit-link correlation. |
| `tests/config/test_azure_devops_config.py` | Schema loading, env-var resolution, regex defaults. |
| `tests/extractors/test_tickets_azure_devops.py` | `AB#` extraction, mixed-platform commit messages. |
| `tests/pm_framework/adapters/fixtures/azure_devops/` | JSON fixtures: `connection_data.json`, `projects_list.json`, `wiql_query_response.json`, `workitems_batch.json`, `workitems_batch_capped.json`, `workitems_batch_paginated_2.json`, `iterations_list.json`, `graph_users.json`, `comments.json`, `fields.json`, `relations.json`. |
| `docs/configuration/azure-devops.md` | User-facing config reference. |
| `docs/examples/azure-devops-configuration.md` | Sample YAML block. |

### 3.2 Class surface

```
class AzureDevOpsAdapter(AzureDevOpsConvertersMixin, BasePlatformAdapter):
    def _get_platform_name(self) -> str  # "azure_devops"
    def _get_capabilities(self) -> PlatformCapabilities  # see §3.4
    def authenticate(self) -> bool  # GET _apis/connectionData; checks PAT scopes
    def test_connection(self) -> dict  # diagnostic with org, projects, types, scopes
    def get_projects(self) -> list[UnifiedProject]
    def get_issues(self, project_id, since=None, issue_types=None) -> list[UnifiedIssue]
    def get_sprints(self, project_id) -> list[UnifiedSprint]  # via classification node tree
    def get_users(self, project_id) -> list[UnifiedUser]  # Graph API; degrades to assignee-extract on 403
    def get_issue_comments(self, issue_key) -> list[dict]
    def get_custom_fields(self, project_id) -> dict
    def get_commit_links(self, commit_shas: list[str]) -> list[tuple[str, str]]  # ADO-specific via artifactUriQuery
```

### 3.3 Config schema (new)

`AzureDevOpsConfig` dataclass added to `src/gitflow_analytics/config/schema.py` near `JIRAConfig` (~line 470):

```
@dataclass
class AzureDevOpsConfig:
    enabled: bool = True
    organization_url: str = ""              # https://dev.azure.com/{org}
    personal_access_token: str = ""         # ${AZURE_DEVOPS_PAT}
    project: Optional[str] = None           # default project; multi-project via list later
    work_item_types: Optional[list[str]] = None   # allowlist; None=canonical 6
    area_paths: list[str] = []              # AreaPath UNDER filters (OR-joined)
    story_point_fields: list[str] = ["Microsoft.VSTS.Scheduling.StoryPoints",
                                     "Microsoft.VSTS.Scheduling.Effort",
                                     "Microsoft.VSTS.Scheduling.Size"]   # Agile, Scrum, CMMI
    custom_fields: dict[str, str] = {}      # friendly_name -> reference_name
    api_version: str = "7.1"
    batch_size: int = 200                   # ADO hard cap
    rate_limit_delay: float = 0.2
    verify_ssl: bool = True
    cache_ttl_hours: int = 168              # 7 days
    dns_timeout: int = 10
    connection_timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 1.0
    state_category_overrides: dict[str, str] = {}    # "Approved" -> "Proposed"
    work_item_type_overrides: dict[str, str] = {}    # "Custom Type" -> "story"
    ticket_regex_override: Optional[str] = None
    is_on_premise: bool = False             # reserved; v1 rejects with error
```

YAML shape (canonical):
```yaml
pm:
  azure_devops:
    enabled: true
    organization_url: "https://dev.azure.com/myorg"
    project: "MyProject"
    personal_access_token: "${AZURE_DEVOPS_PAT}"
    work_item_types: ["User Story","Bug","Task","Feature","Epic"]
```

The loader auto-synthesizes the matching `pm_integration.platforms.azure_devops` block (mirror `_process_jira_pm_config` at `loader_sections.py:531`). **No top-level `azure_devops:` block** — that fragments the config space the way `jira:` and `jira_integration:` already do.

### 3.4 Capability flags

```
supports_projects        = True
supports_issues          = True
supports_sprints         = True
supports_time_tracking   = True   # OriginalEstimate / RemainingWork / CompletedWork
supports_story_points    = True
supports_custom_fields   = True
supports_issue_linking   = True
supports_comments        = True
supports_attachments     = False  # not used in v1
supports_workflows       = True   # state categories
supports_bulk_operations = True   # workitemsbatch
supports_cursor_pagination = True # x-ms-continuationtoken
rate_limit_requests_per_hour = 1500   # approximation; ADO uses TSTU
rate_limit_burst_size        = 200
max_results_per_page         = 200    # workitemsbatch hard limit
```

### 3.5 Mapping tables

**Work-item type → IssueType** (covers Agile, Scrum, Basic, CMMI):

| ADO type | IssueType |
|----------|-----------|
| Epic | EPIC |
| Feature | FEATURE |
| User Story / Product Backlog Item / Requirement / Issue (Basic) | STORY |
| Task | TASK |
| Bug | BUG |
| Issue (Agile/CMMI) | INCIDENT |
| Change Request | IMPROVEMENT |
| Risk / Review / Test Case / Test Plan | TASK with descriptive label |
| Impediment (Scrum) | TASK with `blocked` label |
| anything else | UNKNOWN (with `work_item_type_overrides` config hook) |

**State category → IssueStatus**:

| Category | IssueStatus |
|----------|-------------|
| Proposed | TODO |
| InProgress | IN_PROGRESS |
| Resolved | IN_REVIEW |
| Completed | DONE |
| Removed | CANCELLED |
| (null) | name-fallback heuristic, then UNKNOWN |

**Priority** (`Microsoft.VSTS.Common.Priority`, integer 1–4): 1→CRITICAL, 2→HIGH, 3→MEDIUM, 4→LOW, 0/null→UNKNOWN. Reuse `BasePlatformAdapter._map_priority` (already handles `"1"`–`"4"` strings).

---

## 4. Phased plan

### Phase 0 — Foundation & decisions (0.5 day)

**Goal:** lock open decisions in §2; prepare dependency/secret plumbing.

**Deliverables:**
- ADR document resolving §2 open questions, committed to `docs/adr/` if that dir exists, otherwise `docs/design/azure-devops-decisions.md`.
- `.env.example` updated with `AZURE_DEVOPS_ORG_URL`, `AZURE_DEVOPS_PAT`, optional `AZURE_DEVOPS_PROJECT`.
- `pyproject.toml` left unchanged (requests-only path).

**Acceptance:** ADR merged. No code yet.

**Dependencies:** none.

---

### Phase 1 — Refactor platform-tag plumbing + config schema + stub adapter (1.5 days)

**Goal:** make Azure DevOps registrable, parseable, and **silently fixable** in legacy paths before any real ADO code lands. This phase is the project's risk insurance.

**Files touched (refactor slice — fixes hardcoded `"jira"` hotspots):**
- `src/gitflow_analytics/core/data_fetcher_processing.py:170,183,328,343` — replace literal `platform="jira"` with platform tag from the orchestrator's adapter map.
- `src/gitflow_analytics/core/data_fetcher.py:17,126,287`, `data_fetcher_parallel.py:14,278,514` — generalize `jira_integration` parameter to `pm_orchestrator` (accepting `PMFrameworkOrchestrator`); add a deprecated alias keeping the old kwarg working.
- `src/gitflow_analytics/core/analyze_pipeline.py:284,293-296,360`, `pipeline_collect.py:66,73,78,143`, `cli_fetch.py:170,213-214,359` — same generalization.
- `src/gitflow_analytics/core/cache.py:172` — per-platform breakdown for diagnostics.
- `src/gitflow_analytics/core/schema_version.py:94` — schema fingerprint registry now keyed by platform list, accepts `azure_devops`.
- `src/gitflow_analytics/integrations/orchestrator.py:241,340` — remove JIRA special-case; route through registered adapters by name.
- `src/gitflow_analytics/reports/ticketing_activity_report.py:261,357`, `metrics/activity_scoring.py:207` — iterate `cfg.get_effective_ticket_platforms()` instead of hardcoded `"jira"`.

**Files touched (config + stub slice):**
- `src/gitflow_analytics/config/schema.py:386` — add `"azure_devops": r"AB#(\d+)"` to `TicketDetectionConfig.patterns` defaults.
- `src/gitflow_analytics/config/schema.py:~470` — add `AzureDevOpsConfig` dataclass.
- `src/gitflow_analytics/config/schema.py:~717` — add optional `azure_devops: Optional[AzureDevOpsConfig] = None` to `Config`.
- `src/gitflow_analytics/config/schema.py:778-815` — `get_effective_ticket_platforms()` includes `"azure_devops"` when the config block is present.
- `src/gitflow_analytics/config/loader_sections.py` — add `_process_azure_devops_pm_config` parallel to `_process_jira_pm_config:531`. Auto-synthesize `pm_integration.platforms.azure_devops`. Resolve `${AZURE_DEVOPS_PAT}` env var; raise `EnvironmentVariableError` on missing creds.
- `src/gitflow_analytics/pm_framework/adapters/azure_devops_adapter.py` (NEW) — `AzureDevOpsAdapter` stub; `__init__` reads config; every other method raises `NotImplementedError("Azure DevOps adapter: <method> — implemented in Phase X")`. All capability flags `False` for now.
- `src/gitflow_analytics/pm_framework/adapters/__init__.py` — export `AzureDevOpsAdapter`.
- `src/gitflow_analytics/pm_framework/orchestrator.py:115` — uncomment `register_adapter("azure_devops", AzureDevOpsAdapter)`.
- `tests/config/test_azure_devops_config.py` (NEW), regression sweep over JIRA tests.

**Acceptance:**
- All existing tests still pass after the refactor (zero JIRA regressions).
- Loading a config with `pm.azure_devops` parses without error.
- Orchestrator reports `azure_devops` as a registered adapter.
- `core/data_fetcher_processing.py` no longer has `platform="jira"  # Assuming JIRA` literal.
- A new test asserts that `get_effective_ticket_platforms()` includes `"azure_devops"` when configured.

**Estimated effort:** 1.5 days. (The refactor slice is the bulk; budget aggressively because legacy code paths have hidden test coverage.)

**Dependencies:** Phase 0.

---

### Phase 2 — Auth + read-only project listing (1 day)

**Goal:** prove HTTP/auth/session work end-to-end with the smallest possible read surface.

**Files touched:** `azure_devops_adapter.py` — implement `_create_session`, `_ensure_session`, `_build_url` (cloud-only; reject `tfs/` collection URLs at config load time with explicit "ADO Server unsupported in v1"), `authenticate`, `test_connection`, `get_projects`. Auth header: `Authorization: Basic <b64(":<pat>")>`.

**Tests:** `responses`-mocked unit tests covering happy path + 401/403/404/5xx-with-retry. Integration test gated by `GFA_AZURE_DEVOPS_INTEGRATION_TEST=1`.

**Acceptance:** `gitflow-analytics test-pm-connection azure_devops` (added in Phase 7) succeeds against a real org. PAT scope diagnostic prints required scopes (`vso.work`, optionally `vso.identity`/`vso.graph`).

**Estimated effort:** 1 day. **Dependencies:** Phase 1.

---

### Phase 3 — Work items via WIQL + batch fetch (2 days)

**Goal:** core feature parity with JIRA's `get_issues`.

**Files touched:**
- `azure_devops_adapter.py` — `get_issues`, `_build_wiql`, `_fetch_work_items_batch`, `_get_default_fields`, `_get_state_category` (with state metadata cache per work item type).
- `azure_devops_converters.py` (NEW) — `_convert_work_item`, `_convert_identity_ref` (handles both string and dict shapes), `_map_work_item_type`, `_map_state_category` (with name-fallback heuristic), `_map_priority_value`, `_extract_story_points_ado`, `_unified_issue_to_dict` / `_dict_to_unified_issue`.
- `azure_devops_cache.py` (NEW) — SQLite-backed `AzureDevOpsTicketCache`; same interface as `JiraTicketCache`.

**WIQL pagination:** WIQL has a result cap (research agent reported 20,000; review agent reported 1,000 — both can be true depending on which limit you hit first). **Implement defensive windowing regardless**: sort by `[System.ChangedDate]` ascending; if a window returns ≥ `WIQL_RESULT_LIMIT` (configurable, default 1000), binary-split the time window and recurse. Add an explicit assertion that the final union covers `[since, now]` with no gaps.

**Tests:** mapper tests for every row in §3.5. Pagination test with synthetic 1500-row fixture. Cache round-trip preserves all `UnifiedIssue` fields including `story_points: float`.

**Acceptance:** ≥ 85% line coverage on `azure_devops_adapter.py` + `azure_devops_converters.py`. Pulling 4 weeks of work items from a fixture produces well-formed `UnifiedIssue` objects.

**Estimated effort:** 2 days. **Dependencies:** Phase 2.

---

### Phase 4 — Sprints (Iterations) + users (1 day)

**Goal:** fill in capability flags so velocity reports populate.

**Files touched:**
- `azure_devops_adapter.py` — `get_sprints` via `GET /_apis/wit/classificationnodes/Iterations?$depth=10` (project-level; deduplicated by classification node path). `get_users` via Graph API on `vssps.dev.azure.com`; on 403 (PAT lacks `vso.graph`) degrades gracefully — extract unique assignees from the cached work-item set, log the degradation, drop `supports_users` flag to `False` for that session.

**Acceptance:** Sprint timeFrame (`past`/`current`/`future`) → (`is_active`, `is_completed`) round-trips. `get_users` returns `[]` cleanly when scope is missing.

**Estimated effort:** 1 day. **Dependencies:** Phase 3.

---

### Phase 5 — Comments, custom fields, native commit links (1 day)

**Goal:** match JIRA optional surfaces and add ADO's unique value-add (native links).

**Files touched:**
- `azure_devops_adapter.py` — `get_issue_comments` (`/comments?api-version=7.1-preview.4`), `get_custom_fields` (filtered to `Custom.*` and `WEF.*` reference names), `get_commit_links(commit_shas)` via `POST /_apis/wit/artifactUriQuery` batched at 50 SHAs/request.
- `pm_framework/orchestrator.py` — in `correlate_issues_with_commits`, add a `hasattr(adapter, 'get_commit_links')`-gated branch that contributes `correlation_method="native_link"` results with confidence 1.0.

**Acceptance:** Commits without `AB#` references but with ADO native artifact links produce correlations. JIRA correlation tests still pass (the new branch is duck-typed).

**Estimated effort:** 1 day. **Dependencies:** Phase 3.

---

### Phase 6 — Ticket extraction in commit messages (0.5 day)

**Goal:** make `AB#1234` resolve through the existing extractor pipeline.

**Files touched:**
- `src/gitflow_analytics/extractors/tickets.py:176-192` — register `"azure_devops"` regex (default `r"AB#(\d+)"`).
- `tickets.py:212` — explicit case-sensitivity rule for ADO (`AB#` is case-sensitive convention).
- `tickets.py:533` — extend platform-normalization branch (currently `"jira" or "linear"`) to include `"azure_devops"`.
- `tickets.py:708` — `_format_ticket_id` returns `f"AB#{ticket_id}"` for the `azure_devops` platform.
- `src/gitflow_analytics/extractors/tickets_analysis.py:150` — broaden platform inference.

**Tests:** Mixed-platform commit message test (`"Fix login PROJ-5 + AB#9"` extracts both with correct platform tags).

**Acceptance:** `_correlate_by_ticket_references` in the orchestrator produces ADO correlations from commits with `AB#` refs.

**Estimated effort:** 0.5 day. **Dependencies:** Phase 3.

---

### Phase 7 — Setup wizard + reports + diagnostic CLI (1 day)

**Goal:** UX surface parity.

**Files touched:**
- `src/gitflow_analytics/cli_wizards/install_wizard_pm.py` — `_setup_azure_devops`, `_validate_azure_devops`, `_store_azure_devops_config`, `_discover_azure_devops_fields` (mirror `_setup_jira:22`). Validate PAT scopes by hitting `connectionData` and probing `_apis/wit/workitems`.
- `src/gitflow_analytics/cli_wizards/install_wizard.py:43-79,184-212,366-380` — generalize the wizard flow to register ADO alongside JIRA.
- `src/gitflow_analytics/cli_wizards/install_wizard_output.py:277` — emit `ticket_platforms.append("azure_devops")` when configured.
- `src/gitflow_analytics/cli_setup.py:212,250,459` — extend `discover-fields` and the example block.
- **New CLI command:** `gitflow-analytics test-pm-connection [platform]` (vendor-neutral; works for JIRA today + ADO new). Wire through `PMFrameworkOrchestrator.test_connection`.
- `src/gitflow_analytics/integrations/azure_devops_identity_sync.py` — **stub only** that logs "deferred to v1.1"; full implementation is a follow-up. Add `gitflow-analytics ado-identity-doctor` diagnostic that lists ADO assignees missing from `developer_identities`.

**Acceptance:** Interactive `gitflow-analytics install` wizard offers Azure DevOps as a PM option, validates the PAT, and writes a working YAML.

**Estimated effort:** 1 day. **Dependencies:** Phases 2, 5.

---

### Phase 8 — Documentation (0.5 day)

**Files touched:**
- `docs/configuration/azure-devops.md` (NEW) — full config reference, required PAT scopes (`vso.work`, optional `vso.identity`/`vso.graph`), env-var resolution, troubleshooting (401/403 PAT scope errors, on-prem rejection, WIQL windowing behavior, identity matching caveats).
- `docs/examples/azure-devops-configuration.md` (NEW) — paste-ready YAML.
- `docs/configuration/configuration.md` — add ADO to platform reference table.
- `README.md` — list ADO under PM integrations.
- `CHANGELOG.md` — `[Unreleased] ### Added` entry.

**Acceptance:** `tests/docs/test_azure_devops_doc_example.py` (NEW) parses the doc YAML and validates against the schema.

**Estimated effort:** 0.5 day. **Dependencies:** Phase 7.

---

### Phase 9 — Test, validate, ship (1 day)

**Goal:** release-ready.

**Steps:**
1. Full `pytest -q` green (zero JIRA regressions; verified against `tests/integrations/test_jira_activity_integration.py` + `tests/test_pm_env_resolution.py` baseline).
2. Run `pytest -m integration` against a real ADO org with `GFA_AZURE_DEVOPS_INTEGRATION_TEST=1` env vars set.
3. Manual end-to-end: clone a repo with `AB#` commits → `gitflow-analytics analyze` → confirm correlations populate, no double-counting with co-existing JIRA refs.
4. Performance benchmark: 4-week / 500-issue pull completes in <30s cold, <5s warm.
5. `src/gitflow_analytics/_version.py` → `3.16.0` (3.15.0 was consumed by PR #56).
6. `CHANGELOG.md` — promote `[Unreleased]` to `[3.16.0] - <release-date>`.
7. Commits: `feat: add Azure DevOps PM platform integration` + `chore: bump version to 3.16.0`.

**Acceptance:** all gates in §6 below pass.

**Estimated effort:** 1 day. **Dependencies:** all prior phases.

---

## 5. Risk register (top 10)

| # | Risk | L | I | Mitigation | Phase |
|---|------|---|---|------------|-------|
| 1 | Dual-stack (`pm_framework/` vs `integrations/`) — ~40 hardcoded `"jira"` strings in legacy enrichment paths silently drop ADO data | H | H | Phase 1 refactor slice. **This is the project's biggest hidden risk.** | 1 |
| 2 | WIQL row cap (1k or 20k depending on which you hit first) silently truncates large pulls | H | H | Defensive `[ChangedDate]` windowing with binary-split + gap-detection assertion | 3 |
| 3 | Custom process templates (CMMI, inherited) rename types and states; naive name-mapping misclassifies >50% | H | H | Map by `StateCategory`, not state name. Ship Agile + Scrum + Basic + CMMI tables. Config overrides for inherited custom processes. | 3 |
| 4 | Ticket regex `#(\d+)` collides with GitHub PR/issue numbers | H | H | Default `r"AB#(\d+)"`. Validator warns if both ADO + GitHub enabled with bare `#` | 1, 6 |
| 5 | Rate-limit / TSTU exhaustion under burst loads (ADO TSTU model differs from JIRA hourly cap) | M | H | Custom retry middleware reading `Retry-After` and `X-RateLimit-Delay` headers; configurable max-backoff (default 5 min); log throttle events with `X-RateLimit-Resource` | 2 |
| 6 | On-prem TFS / ADO Server URL parsing silently breaking against Services-only assumptions | M | H | v1 explicitly rejects `tfs/` collection URLs at config-load time | 1 |
| 7 | ADO `uniqueName` (UPN) ≠ git commit email → identity merge failures | H | M | Defer full sync to v1.1. v1 ships `ado-identity-doctor` diagnostic + `identity_aliases` config block | 7 |
| 8 | Deleted/recycled work items return 404 mid-run; cache poisoning if soft-deleted item is later restored | M | M | Negative-cache 404s with separate short TTL (1h, configurable) | 3 |
| 9 | Team-scoped iterations either double-count or miss data depending on enumeration strategy | M | M | Use project-level classification node tree; document team-scoped capacity data as out-of-scope for v1 | 4 |
| 10 | Story points stored as float (`3.5`) lose precision via `int(float(value))` | — | — | **Resolved** in v3.15.0 (Bob's PR #56). No remaining mitigation work for ADO. | — (closed) |

---

## 6. Validation / acceptance gates

Before merging:

1. **Regression:** All existing tests pass; `tests/integrations/test_jira_activity_integration.py`, `tests/test_pm_env_resolution.py`, ticketing-activity-report tests unchanged.
2. **Unit coverage:** ≥85% for `azure_devops_adapter.py` + `azure_devops_converters.py`; ≥75% combined new-code coverage.
3. **Edge case suite:** All 25 cases in §7 covered by named tests (use `responses` library; fixtures scrubbed from real ADO responses).
4. **Integration:** `pytest -m integration` with `GFA_AZURE_DEVOPS_INTEGRATION_TEST=1` against a real org; `len(get_issues(project, since=...))` matches an independent WIQL count within 0–1 (allowing for in-flight modifications during the run).
5. **Performance:** 4-week / 500-issue pull ≤30s cold, ≤5s warm; cache hit-rate ≥80% on second run.
6. **Cross-platform correlation:** A repo with both `PROJ-123` (JIRA) and `AB#456` (ADO) refs produces two distinct correlations, neither shadowing the other.
7. **Dual-stack consistency:** After Phase 1 refactor, run analyze pipeline twice (legacy enrichment path + `pm_framework` path) and assert ticket counts match per platform.
8. **Diagnostic:** `gitflow-analytics test-pm-connection azure_devops` exits 0 on valid config with PAT scopes printed; non-zero with remediation message on failure.
9. **Migration safety:** Running v3.15.0 against an existing v3.14.x cache directory does not corrupt JIRA data; `core/schema_version.py` migration coexists.

---

## 7. Edge cases the test plan must cover

1. Empty project (0 work items) → `get_issues` returns `[]`.
2. Project with exactly 1000 work items → no truncation; 1001 → still no truncation (windowing kicks in).
3. Project with 50,000 work items + `since` → completes in <5 min, ≤200 batch GETs.
4. Work item with `null System.AssignedTo` → `assignee=None`.
5. `System.AssignedTo` as legacy string `"Alice <alice@x>"` vs modern dict → both normalize to `UnifiedUser`.
6. Custom `System.State="Awaiting Review"` + `StateCategory="InProgress"` → maps to `IN_PROGRESS`.
7. `StateCategory` null (legacy CMMI) → name-fallback, then `UNKNOWN`, never crashes.
8. Story points `3.5` (float) → preserved as `3.5` (requires `Optional[float]` model change).
9. Scrum `Effort` field but no `StoryPoints` → fallback resolves correctly.
10. CMMI `Size` field instead → still resolves.
11. 429 with `Retry-After: 5` → adapter sleeps 5s, retries, total elapsed matches.
12. 429 with `Retry-After: 120` (over default cap) → respects header up to configured max.
13. 401 mid-pagination (PAT expired) → fast-fail with explicit "PAT expired or revoked" message.
14. 203 with HTML body (login redirect) → raises auth error; HTML never passed to JSON parser.
15. Network timeout at page 5/10 → partial results discarded, exception with context.
16. WIQL with apostrophe in project name → properly escaped.
17. Circular parent-child link → `linked_issues` extraction terminates.
18. Iteration with same name across two teams → de-dup by full classification path.
19. Sprint with null start/end (planning-only) → `is_active=False, is_completed=False`.
20. Commit with `AB#1234` and `#5678` → ADO match `1234`, GitHub match `5678`, no double-counting.
21. Commit with `AB#1234` only, ADO not configured → no spurious correlation.
22. Mixed-platform commit `PROJ-1 + AB#1` → both correlations recorded with correct `platform`.
23. Cache hit of older schema → `_dict_to_unified_issue` upgrades or invalidates, never crashes.
24. ADO org with 25 projects, only 2 enabled → only those 2 fetched.
25. PAT scoped only to `Work Items (read)` → `test_connection` succeeds; sprint fetch fails with "PAT lacks Project & Team (read) scope" message rather than generic 403.

---

## 8. Out of scope for v1 (explicit non-goals)

- Write operations (create/update work items, comments, links).
- Azure DevOps Server / TFS on-prem (rejected at config load).
- Active Directory / NTLM / Kerberos auth.
- OAuth 2.0 / Azure AD app authentication.
- TFVC version control (Git only).
- Test plans, test runs, test results, test cases as first-class entities.
- Full ADO identity sync into `IdentityCache` (deferred to v1.1; diagnostic-only in v1).
- Team-scoped iteration capacity data.
- Multi-org configurations (single org per config in v1).
- Boards-Boards integration / cross-org work-item links.

---

## 9. Suggested next action

1. **User review of §2 open decisions.** Twelve decisions; each is a one-word answer in most cases. Without these, Phase 1 cannot start cleanly.
2. **Spawn a `coder` agent for Phase 1** (refactor + config + stub) as a self-contained PR. This phase is purely additive in user-facing behavior and gives us the safety net before any ADO-specific code.
3. After Phase 1 lands, parallelize Phases 2 + 3 + 4 (data path) on one engineer and Phases 6 + 7 (extractor + wizard) on another.

---

## 10. Reference file index

Files the implementing engineer will edit, grouped by category:

**PM framework (new):**
- `src/gitflow_analytics/pm_framework/adapters/azure_devops_adapter.py`
- `src/gitflow_analytics/pm_framework/adapters/azure_devops_cache.py`
- `src/gitflow_analytics/pm_framework/adapters/azure_devops_converters.py`
- `src/gitflow_analytics/pm_framework/adapters/__init__.py`
- `src/gitflow_analytics/pm_framework/orchestrator.py:115`

**Reference (read, do not modify):**
- `src/gitflow_analytics/pm_framework/base.py`
- `src/gitflow_analytics/pm_framework/models.py` (modify only for `Optional[float]` story_points decision)
- `src/gitflow_analytics/pm_framework/registry.py`
- `src/gitflow_analytics/pm_framework/adapters/jira_adapter.py`
- `src/gitflow_analytics/pm_framework/adapters/jira_adapter_converters.py`
- `src/gitflow_analytics/pm_framework/adapters/jira_cache.py`

**Config:**
- `src/gitflow_analytics/config/schema.py:386,470,717,778`
- `src/gitflow_analytics/config/loader.py:295`
- `src/gitflow_analytics/config/loader_sections.py:531`

**Extractor / correlation:**
- `src/gitflow_analytics/extractors/tickets.py:176,212,533,708`
- `src/gitflow_analytics/extractors/tickets_analysis.py:150`

**Legacy stack refactor (Phase 1):**
- `src/gitflow_analytics/core/data_fetcher.py:17,126,287`
- `src/gitflow_analytics/core/data_fetcher_parallel.py:14,278,514`
- `src/gitflow_analytics/core/data_fetcher_processing.py:14,170,183,328,343`
- `src/gitflow_analytics/core/analyze_pipeline.py:284,293-296,360`
- `src/gitflow_analytics/pipeline_collect.py:66,73,78,143`
- `src/gitflow_analytics/cli_fetch.py:170,213-214,359`
- `src/gitflow_analytics/core/cache.py:172`
- `src/gitflow_analytics/core/schema_version.py:94`
- `src/gitflow_analytics/integrations/orchestrator.py:241,340`
- `src/gitflow_analytics/reports/ticketing_activity_report.py:261,357`
- `src/gitflow_analytics/metrics/activity_scoring.py:207`

**CLI / wizard:**
- `src/gitflow_analytics/cli_wizards/install_wizard.py:43-79,184-212,366-380`
- `src/gitflow_analytics/cli_wizards/install_wizard_pm.py:22`
- `src/gitflow_analytics/cli_wizards/install_wizard_output.py:277`
- `src/gitflow_analytics/cli_setup.py:212,250,459`

**Identity (stub for v1):**
- `src/gitflow_analytics/integrations/azure_devops_identity_sync.py` (NEW, stub)

**Release plumbing:**
- `pyproject.toml`
- `.env.example`
- `src/gitflow_analytics/_version.py` → `3.15.0`
- `CHANGELOG.md`

**Docs:**
- `docs/configuration/azure-devops.md` (NEW)
- `docs/examples/azure-devops-configuration.md` (NEW)
- `docs/configuration/configuration.md`
- `README.md`
- `docs/design/azure-devops-decisions.md` (NEW — Phase 0 ADR)
