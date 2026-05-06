# Azure DevOps Integration — Phase 0 Decision Record

**Date:** 2026-05-05
**Status:** Accepted
**Scope:** Decisions blocking Phase 1 (refactor + config schema + stub adapter) of the Azure DevOps PM integration. Companion to [azure-devops-integration-plan.md](./azure-devops-integration-plan.md).

This document codifies four decisions reached during planning. It exists so Phase 1–9 contributors do not re-litigate the same trade-offs and so the plan's `§2 Open decisions` table has a durable rationale trail.

Decisions deferred to later phases (HTTP client, sprint enumeration, CMMI mapping coverage, cache layout, native commit-link enrichment, identity sync depth) are not in scope here — they will be recorded in their own ADRs as those phases begin.

---

## Decision 1 — Authentication method: PAT only for v1

### Context

Azure DevOps Services supports three auth modes for API clients:

1. Personal Access Tokens (HTTP Basic with empty username, base64-encoded `:<PAT>`).
2. Microsoft Entra ID (formerly AAD) OAuth 2.0 — device flow or auth-code flow.
3. Azure AD service principal (client_id + client_secret + tenant_id).

The `AzureDevOpsConfig` dataclass shape and the loader's env-var resolution behavior depend on which modes are supported.

### Decision

**v1 ships PAT-only.** Single config field `personal_access_token: str` resolved from `${AZURE_DEVOPS_PAT}` environment variable.

OAuth and Azure AD service principal are **deferred** to v1.1. When added, they will be selected by an additive `auth_method: Literal["pat", "oauth", "service_principal"] = "pat"` discriminator on the existing dataclass — no breaking change to v1 configs.

### Rationale

- gitflow-analytics is a single-user CLI tool today. OAuth's primary benefit (avoid handing a long-lived shared credential to every team member) does not apply.
- Every existing PM integration in this codebase uses an API token (JIRA `api_token`, Confluence `api_token`). PAT is consistent.
- OAuth (device flow) adds ~200–400 LOC: token cache file, refresh-token handling, and either a localhost callback listener or device-code polling. Out of proportion for v1.
- The default `"pat"` discriminator value makes v1.1's OAuth opt-in fully backward-compatible.

### Consequences

- Users in security-conscious orgs that prohibit long-lived PATs cannot use v1. They will need to wait for v1.1's OAuth path or rotate PATs manually.
- `AZURE_DEVOPS_PAT` becomes a documented required environment variable. The setup wizard prompts for it; the loader raises `EnvironmentVariableError("AZURE_DEVOPS_PAT", "AzureDevOps", config_path)` if missing — pattern parallels JIRA's `JIRA_API_TOKEN` handling at `loader_sections.py:113`.
- Required PAT scopes (documented in `docs/configuration/azure-devops.md`):
  - `vso.work` — read work items, queries, area/iteration paths (required)
  - `vso.identity` — read user identities (recommended; needed for `get_users`)
  - `vso.graph` — Graph API user enumeration (optional; degrades gracefully if absent)

### Schema impact (Phase 1)

```python
@dataclass
class AzureDevOpsConfig:
    enabled: bool = True
    organization_url: str = ""
    personal_access_token: str = ""   # resolved from ${AZURE_DEVOPS_PAT}
    # ... (other fields per plan §3.3)
```

No `auth_method` field in v1 — its absence is the v1 implicit default. v1.1 adds it without migrating existing configs.

---

## Decision 2 — On-premises Azure DevOps Server: rejected at config load

### Context

Azure DevOps Services (cloud) and Azure DevOps Server (on-premises, formerly TFS) share most of their REST API surface but diverge on:

- URL pattern: `https://dev.azure.com/{org}` or `https://{org}.visualstudio.com` (cloud) vs. `https://{server}/{collection}/...` (on-prem).
- API version compatibility: Server lags Services by 6–12 months; `api-version=7.1` requires ADO Server 2022+.
- Authentication: Server commonly uses NTLM/Kerberos in addition to PAT, especially behind reverse proxies.
- TLS: self-signed certs are common in enterprise on-prem deployments.

Silent acceptance of on-prem URLs would let users configure ADO Server, hit auth or API-version errors mid-fetch, and file confusing bug reports.

### Decision

**v1 rejects on-prem URLs at config load time** with an actionable error message.

The loader validates `organization_url` against a cloud-host allowlist (`dev.azure.com`, `*.visualstudio.com`). URLs containing `/tfs/`, `/DefaultCollection/`, or any other host are rejected with:

```
Azure DevOps Server (on-premises) is not supported in v1.
Only Azure DevOps Services (cloud) is supported. Configure
either https://dev.azure.com/{org} or https://{org}.visualstudio.com.
On-premises support is roadmapped for v1.2; see
docs/design/azure-devops-integration-plan.md §8.
```

`AzureDevOpsConfig.is_on_premise: bool = False` is **reserved as a forward-compatibility marker**. Setting it to `True` is rejected by the loader with the same message in v1; v1.2 will flip the rejection to a feature-gated implementation.

### Rationale

- Full on-prem support adds an estimated 3–5 engineering days to v1 plus a real ADO Server test environment. Out of scope for the v1 timeline.
- Optimistic acceptance ("let it fail at runtime") wastes user troubleshooting hours on cryptic NTLM/SSL/api-version errors that the loader can preempt in milliseconds.
- Reserving `is_on_premise` in the schema now means v1.2 can ship without a config migration.

### Consequences

- Enterprise customers running ADO Server are explicitly locked out of v1 with a clear message pointing at the v1.2 roadmap.
- The loader gains a small URL-pattern validator (~10 LOC).
- The setup wizard's PAT-validation step in Phase 7 includes a pre-check for on-prem URLs to catch user error before the PAT prompt.

---

## Decision 3 — Default ticket regex: strict `AB#` with per-config override

### Context

Azure Boards recognizes two commit-message reference forms:

- `AB#1234` — the explicit "Azure Boards" prefix Microsoft injects via the official GitHub-↔-Azure-Boards integration. Cross-platform-safe.
- `#1234` — bare numeric reference. Auto-detected by Azure Repos but **collides catastrophically** with GitHub PR/issue numbers in mixed-platform repos.

The default regex is wired into `Config.get_effective_ticket_platforms()` (`schema.py:813`), which runs even when no PM platform is explicitly configured. A loose default has cross-codebase blast radius — every gitflow-analytics user would see different ticket-extraction behavior on upgrade.

### Decision

- **Default regex:** `r"AB#(\d+)"` registered at `TicketDetectionConfig.patterns["azure_devops"]` (`schema.py:386`).
- **Per-config override:** `AzureDevOpsConfig.ticket_regex_override: Optional[str] = None`. When set, replaces the default for that deployment only.
- **Documented opt-in pattern** for Azure-Repos-only shops who want bare `#1234` support: `ticket_regex_override: "(?:AB#|#)(\\d+)"`. Documented with an explicit GitHub-collision warning in `docs/configuration/azure-devops.md`.
- **Case sensitivity:** `AB#` is case-sensitive by Microsoft convention. Phase 6 adds `azure_devops` to the case-sensitive platform list at `extractors/tickets.py:212` (currently `platform != "jira"`).

### Rationale

- `AB#` is the only cross-platform-unambiguous form. It is what Microsoft's own tooling emits.
- A loose default would silently reclassify every `#NNN` PR reference in every existing user's repos as an ADO work item — unacceptable side effect of an additive feature.
- The override exists for the legitimate Azure-Repos-only use case where users *do* want bare-`#` matching, accepting that it conflicts with GitHub.

### Consequences

- Users in Azure-Repos-only shops who type bare `#1234` see zero correlations until they set `ticket_regex_override`. The setup wizard prompts about this explicitly.
- Migration concern: existing JIRA-only users see no behavior change (ADO regex registers but is only consulted when ADO is configured or in the no-config fallback list, where it harmlessly fails to match anything in practice).
- Phase 6 work item: validate that adding `azure_devops` to the case-sensitive list does not regress any existing JIRA/Linear/ClickUp tests.

---

## Decision 4 — `UnifiedIssue.story_points` type: change to `Optional[float]`

**Status:** **Accepted — implemented and merged.** Bob landed PR #56 (commit `bea075f`, released as v3.15.0 on 2026-05-05) which widened the model to `Optional[float]`, fixed the JIRA adapter's `int(float(value))` coercion, and updated the SQLAlchemy / SQLite schemas. The implementation matches this ADR's design and extends it to additional sites we'd missed in our audit:
- `UnifiedSprint.planned_story_points` and `completed_story_points` also widened.
- `_DayStats.story_points` dataclass field widened.
- `pipeline_report.py` had its own `int()` truncation that was also corrected.
- A v12.0 `schema_version` migration helper was added (SQLite dynamic type affinity means no DDL rebuild needed).
- New test `tests/integrations/test_jira_adapter_story_points.py` covers 5 cases (3.5, 1.5 string, 0.5, integer-as-float, unparseable).

The ADO adapter (Phase 3) inherits the corrected behavior automatically; no follow-up work is needed in this codebase to consume the change.

### Context

`UnifiedIssue.story_points` is currently `Optional[int]` (`pm_framework/models.py:150`). The JIRA adapter's converter coerces values via `int(float(value))` at `jira_adapter_converters.py:261`, silently dropping fractional precision. Story-point estimates of `3.5` round to `3`.

Azure DevOps natively stores story points as `double`:
- `Microsoft.VSTS.Scheduling.StoryPoints` — Agile process
- `Microsoft.VSTS.Scheduling.Effort` — Scrum process
- `Microsoft.VSTS.Scheduling.Size` — CMMI process

Modified Fibonacci scales (`0.5, 1, 2, 3, 5, 8, 13`) are common in real teams. Shipping ADO with `int` coercion would silently mangle these.

### Decision

**Change `UnifiedIssue.story_points` from `Optional[int]` to `Optional[float]` in Phase 1.**

Same change applies to `BasePlatformAdapter._extract_story_points` return type (`base.py:333`) and any cache schema columns persisting story points (`INTEGER` → `REAL` in SQLite).

The JIRA adapter's `int(float(value))` coercion is **fixed in the same PR** as part of the model change — `int(...)` becomes `float(...)`. This eliminates the existing precision bug.

### Rationale

- Doing the change now, while only one adapter exists, is the cleanest moment. Deferring to Phase 3 means revisiting the JIRA adapter's coercion code after committing to a downstream behavior.
- Python's duck typing means most aggregations (`sum()`, division for ratios) silently accept the wider type without consumer changes.
- A separate `story_points_decimal` field (the rejected option C from the planning conversation) creates two sources of truth and is strongly discouraged.

### Consequences

- Phase 1 includes a `grep -rn "story_points" src/gitflow_analytics` audit. Any consumer with explicit `int` typing is updated to `float`. SQLite `INTEGER` columns become `REAL`.
- A new test asserts `UnifiedIssue(story_points=3.5).story_points == 3.5` round-trips through the JIRA adapter's converter without precision loss. This test would fail today.
- Reports formatting story points as `f"{n}"` continue to work (Python prints `3.5` correctly). Reports formatting as `f"{n:d}"` (forcing integer) would break — none found in initial grep but the audit must confirm.
- This is a behavior change for existing JIRA users with fractional story points: they will now see `3.5` in reports where they previously saw `3`. Documented in `CHANGELOG.md` under "Changed" for v3.15.0.

---

## Cross-cutting Phase 1 implications

These decisions together imply the following Phase 1 deliverables (recap from plan §4):

1. `AzureDevOpsConfig` dataclass with PAT-only auth field, reserved `is_on_premise` flag, and `ticket_regex_override` field.
2. URL validator in the loader rejecting on-prem patterns.
3. `r"AB#(\d+)"` default registered at `schema.py:386`.
4. ~~`UnifiedIssue.story_points` type change~~ — **Landed independently** in Bob's PR #56 / v3.15.0 (see Decision 4 status). Phase 1 inherits it from main; no Phase 1 model edits required.
5. The legacy-stack hardcoded-`"jira"` cleanup (independent of the four decisions but Phase 1's primary safety-net work — see plan §4 Phase 1 file list).
6. Stub `AzureDevOpsAdapter` registered with the orchestrator; every data method raises `NotImplementedError("Azure DevOps adapter: <method> — implemented in Phase X")`.
7. New tests: `tests/config/test_azure_devops_config.py`, JIRA-adapter no-regression sweep. (Story-point precision regression test deferred with Decision 4.)

## Decisions deferred to later phases

Recorded here so contributors don't ask twice:

| Phase | Decision | Recommendation (not yet locked) |
|-------|----------|--------------------------------|
| 2 | HTTP client (requests vs official SDK) | Hand-rolled `requests` + `urllib3.Retry`, mirror JIRA pattern |
| 3 | Custom process mapping coverage | Ship Agile + Scrum + Basic + CMMI tables; inherited custom processes fall through to UNKNOWN with config override |
| 3 | Cache layout (per-platform DB vs unified) | Per-platform `azure_devops_tickets.db` for v1; unified `pm_tickets.db` revisited before adding 3rd platform |
| 4 | Sprint enumeration scope | Project-level classification node tree (`/wit/classificationnodes/Iterations`); team-scoped capacity out of scope for v1 |
| 5 | Native commit-link correlation | Use `POST /_apis/wit/artifactUriQuery` as secondary correlation source on top of `AB#` message scanning |
| 7 | Identity sync depth | Defer full sync to v1.1; v1 ships `ado-identity-doctor` diagnostic only |

---

## Change log

- 2026-05-05 — Initial draft. All four decisions accepted in planning conversation.
- 2026-05-06 — Decision 4 status flipped from "Deferred" to "Accepted — implemented and merged" after Bob shipped PR #56 / v3.15.0 with the int→float widening. Phase 1 implications updated to reflect the model change is no longer in our scope.
