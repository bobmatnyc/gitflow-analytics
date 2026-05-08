"""Pipeline stage commands (collect, classify, report) for GitFlow Analytics CLI."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .cli_utils import setup_logging
from .config import ConfigLoader

logger = logging.getLogger(__name__)


@click.command(name="collect")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--weeks",
    "-w",
    type=int,
    default=4,
    show_default=True,
    help="Number of complete weeks to collect",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force re-fetch even for weeks already in the cache",
)
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="none",
    help="Enable logging with specified level",
)
def collect_command(config_path: Path, weeks: int, force: bool, log: str) -> None:
    """Stage 1: Collect raw commit data from repositories into the weekly cache.

    \b
    Fetches commits from every repository listed in the configuration file and
    stores them in the local SQLite cache.  Weeks that are already cached are
    skipped unless --force is given.

    \b
    EXAMPLES:
      # Collect 4 weeks of data
      gfa collect -c config.yaml --weeks 4

      # Force a fresh fetch, ignoring cached weeks
      gfa collect -c config.yaml --weeks 4 -f

    \b
    NEXT STEP:
      gfa classify -c config.yaml
    """
    setup_logging(log, __name__)

    try:
        from .pipeline import run_collect

        cfg = ConfigLoader.load(config_path)
        click.echo(f"Stage 1: Collecting data ({weeks} weeks)...")

        result = run_collect(
            cfg=cfg,
            weeks=weeks,
            force=force,
            progress_callback=lambda msg: click.echo(f"  {msg}"),
        )

        if result.errors:
            for err in result.errors:
                click.echo(f"  Warning: {err}", err=True)

        click.echo(
            f"\nCollect complete: "
            f"{result.total_commits} commits from "
            f"{result.repos_fetched + result.repos_cached} repositories "
            f"({result.repos_cached} cached, {result.repos_fetched} fetched"
            + (f", {result.repos_failed} failed" if result.repos_failed else "")
            + ")"
        )

    except (FileNotFoundError, Exception) as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(1)


@click.command(name="classify")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--weeks",
    "-w",
    type=int,
    default=4,
    show_default=True,
    help="Number of weeks to classify (must match the collect --weeks value)",
)
@click.option(
    "--week",
    "target_weeks",
    multiple=True,
    metavar="YYYY-Www",
    help=(
        "Target a specific ISO week for classification (e.g. 2026-W07). "
        "May be repeated for multiple discrete weeks. "
        "Mutually exclusive with --weeks, --from, and --to."
    ),
)
@click.option(
    "--from",
    "from_week",
    default=None,
    metavar="YYYY-Www",
    help=(
        "Start of an inclusive ISO week range (e.g. 2026-W01). "
        "Must be used together with --to. "
        "Mutually exclusive with --weeks and --week."
    ),
)
@click.option(
    "--to",
    "to_week",
    default=None,
    metavar="YYYY-Www",
    help=(
        "End of an inclusive ISO week range (e.g. 2026-W18). "
        "Must be used together with --from. "
        "Mutually exclusive with --weeks and --week."
    ),
)
@click.option(
    "--reclassify",
    is_flag=True,
    help="Re-classify commits that were already classified",
)
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="none",
    help="Enable logging with specified level",
)
@click.option(
    "--show-jira-signals",
    is_flag=True,
    help=(
        "Log every commit short-circuited by the JIRA project-key mapping "
        "(see 'jira_project_mappings' in config.yaml; issue #62)."
    ),
)
@click.option(
    "--validate-coverage",
    is_flag=True,
    help=(
        "Issue #65: exit non-zero when any repository's classification "
        "coverage is below --coverage-threshold. Useful for CI pipelines "
        "that want to fail when commits are silently falling to maintenance."
    ),
)
@click.option(
    "--coverage-threshold",
    type=float,
    default=20.0,
    show_default=True,
    help=(
        "Per-repo classification coverage percent below which a warning is "
        "emitted (and --validate-coverage exits non-zero). Coverage = % of "
        "commits NOT classified as maintenance/KTLO/other/unknown."
    ),
)
def classify_command(
    config_path: Path,
    weeks: int,
    target_weeks: tuple[str, ...],
    from_week: Optional[str],
    to_week: Optional[str],
    reclassify: bool,
    log: str,
    show_jira_signals: bool,
    validate_coverage: bool,
    coverage_threshold: float,
) -> None:
    """Stage 2: Classify collected commits using batch LLM classification.

    \b
    Reads commits from the local cache (written by 'gfa collect') and runs
    batch classification on them.  Commits that were already classified are
    skipped unless --reclassify is given.

    \b
    EXAMPLES:
      # Classify the last 4 weeks
      gfa classify -c config.yaml

      # Force re-classification of all commits
      gfa classify -c config.yaml --reclassify

    \b
    PREREQUISITE:
      gfa collect -c config.yaml

    \b
    NEXT STEP:
      gfa report -c config.yaml
    """
    setup_logging(log, __name__)

    # Mutual-exclusivity validation for ISO-week targeting flags (#70).
    # We approximate "was --weeks explicitly set?" by comparing to the
    # default. Click does not expose source-of-value cleanly without ctx.
    weeks_default = 4
    targeting_flags = bool(target_weeks) or (from_week is not None) or (to_week is not None)
    weeks_explicitly_set = weeks != weeks_default

    if targeting_flags and weeks_explicitly_set:
        raise click.UsageError("--week/--from/--to cannot be combined with --weeks.")
    if bool(target_weeks) and (from_week is not None or to_week is not None):
        raise click.UsageError("--week cannot be combined with --from/--to.")
    if (from_week is None) != (to_week is None):
        raise click.UsageError("--from and --to must be used together.")

    # Compute explicit_date_range (Mon..Sun span) when targeting flags used.
    from datetime import date as _date

    from .utils.iso_week import iso_week_range, parse_iso_week

    explicit_date_range: Optional[tuple[_date, _date]] = None
    if target_weeks:
        # Union of all requested weeks: min(mondays) -> max(sundays).
        starts, ends = zip(*[parse_iso_week(w) for w in target_weeks])
        explicit_date_range = (min(starts), max(ends))
    elif from_week is not None and to_week is not None:
        explicit_date_range = iso_week_range(from_week, to_week)

    try:
        from .pipeline import run_classify

        cfg = ConfigLoader.load(config_path)
        if explicit_date_range is not None:
            from_iso = explicit_date_range[0].strftime("%G-W%V")
            to_iso = explicit_date_range[1].strftime("%G-W%V")
            click.echo(f"Stage 2: Classifying commits ({from_iso} to {to_iso})...")
        else:
            click.echo(f"Stage 2: Classifying commits ({weeks} weeks)...")

        result = run_classify(
            cfg=cfg,
            weeks=weeks,
            reclassify=reclassify,
            progress_callback=lambda msg: click.echo(f"  {msg}"),
            show_jira_signals=show_jira_signals,
            coverage_threshold=coverage_threshold,
            explicit_date_range=explicit_date_range,
        )

        if result.errors:
            for err in result.errors:
                click.echo(f"  Warning: {err}", err=True)
            if any("Run 'gfa collect' first" in e for e in result.errors):
                click.echo("\nHint: run 'gfa collect -c config.yaml' before classify", err=True)
                sys.exit(1)

        click.echo(
            f"\nClassify complete: "
            f"{result.processed_batches} batches, "
            f"{result.total_commits} commits"
            + (f" ({result.skipped_batches} skipped)" if result.skipped_batches else "")
        )

        # Issue #65: when --validate-coverage is set, surface a prominent
        # warning block and exit non-zero so CI pipelines fail when any
        # repository's classification coverage is below threshold.
        if validate_coverage:
            low_coverage = {
                path: pct
                for path, pct in result.coverage_by_repo.items()
                if pct < coverage_threshold
            }
            if low_coverage:
                click.echo("")
                click.echo("=" * 72, err=True)
                click.echo(
                    f"WARNING: classification coverage validation failed "
                    f"(threshold: {coverage_threshold:.1f}%)",
                    err=True,
                )
                click.echo("=" * 72, err=True)
                # Map repo_path back to the friendly name for output.
                repo_names = {str(repo.path): repo.name for repo in cfg.repositories}
                for path, pct in sorted(low_coverage.items(), key=lambda x: x[1]):
                    name = repo_names.get(path, path)
                    fallthrough = 100.0 - pct
                    click.echo(
                        f"  - {name}: {pct:.1f}% coverage "
                        f"({fallthrough:.1f}% fell to maintenance/KTLO)",
                        err=True,
                    )
                click.echo("=" * 72, err=True)
                click.echo(
                    "Hint: configure 'jira_project_mappings' in config.yaml "
                    "or adopt conventional commit prefixes "
                    "(feat:/fix:/docs:/refactor:/test:/...).",
                    err=True,
                )
                sys.exit(1)

    except SystemExit:
        # Allow sys.exit(1) above to propagate without being swallowed by
        # the broad except below (which was hiding the validate-coverage
        # exit code under "Error: 1").
        raise
    except Exception as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(1)


@click.command(name="report")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--weeks",
    "-w",
    type=int,
    default=4,
    show_default=True,
    help="Number of weeks to include in reports",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for reports (overrides config file)",
)
@click.option(
    "--generate-csv",
    is_flag=True,
    help="Generate CSV reports in addition to the narrative markdown report",
)
@click.option(
    "--anonymize",
    is_flag=True,
    help="Anonymize developer information in reports",
)
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="none",
    help="Enable logging with specified level",
)
def report_command(
    config_path: Path,
    weeks: int,
    output_path: Optional[Path],
    generate_csv: bool,
    anonymize: bool,
    log: str,
) -> None:
    """Stage 3: Generate reports from classified commit data.

    \b
    Reads classified commits from the cache (written by 'gfa collect' and
    'gfa classify') and generates report files.  No git operations are
    performed.

    \b
    EXAMPLES:
      # Generate the narrative markdown report
      gfa report -c config.yaml

      # Generate CSV reports as well
      gfa report -c config.yaml --generate-csv

      # Write reports to a custom directory
      gfa report -c config.yaml -o /tmp/my-reports --generate-csv

    \b
    PREREQUISITE:
      gfa collect -c config.yaml
      gfa classify -c config.yaml
    """
    setup_logging(log, __name__)

    try:
        from .pipeline import run_report

        cfg = ConfigLoader.load(config_path)

        if output_path is None:
            output_path = cfg.output.directory if cfg.output.directory else Path("./reports")

        click.echo(f"Stage 3: Generating reports ({weeks} weeks) → {output_path}")

        result = run_report(
            cfg=cfg,
            weeks=weeks,
            output_dir=output_path,
            generate_csv=generate_csv,
            anonymize=anonymize,
            progress_callback=lambda msg: click.echo(f"  {msg}"),
        )

        if result.errors:
            for err in result.errors:
                click.echo(f"  Warning: {err}", err=True)

        click.echo(f"\nReport complete: {len(result.generated_reports)} files in {output_path}")
        for name in result.generated_reports:
            click.echo(f"  {name}")

    except Exception as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(1)
