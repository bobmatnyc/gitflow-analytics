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
def classify_command(config_path: Path, weeks: int, reclassify: bool, log: str) -> None:
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

    try:
        from .pipeline import run_classify

        cfg = ConfigLoader.load(config_path)
        click.echo(f"Stage 2: Classifying commits ({weeks} weeks)...")

        result = run_classify(
            cfg=cfg,
            weeks=weeks,
            reclassify=reclassify,
            progress_callback=lambda msg: click.echo(f"  {msg}"),
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
