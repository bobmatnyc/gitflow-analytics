"""Analysis commands (analyze, fetch, collect, classify, report) for GitFlow Analytics CLI."""

import logging
from pathlib import Path
from typing import Optional

import click

from .cli_analysis_orchestrator import analyze
from .cli_fetch import fetch
from .cli_pipeline_commands import classify_command, collect_command, report_command

logger = logging.getLogger(__name__)


def register_analysis_commands(cli: click.Group) -> None:
    """Register analysis-related commands onto the CLI group."""
    cli.add_command(analyze_subcommand, name="analyze")
    cli.add_command(fetch, name="fetch")
    cli.add_command(collect_command, name="collect")
    cli.add_command(classify_command, name="classify")
    cli.add_command(report_command, name="report")


@click.command(name="analyze")
@click.option(
    "--config",
    "-c",
    type=click.Path(path_type=Path),
    default="config.yaml",
    help="Path to YAML configuration file (default: config.yaml)",
)
@click.option(
    "--weeks", "-w", type=int, default=12, help="Number of weeks to analyze (default: 12)"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for reports (overrides config file)",
)
@click.option("--anonymize", is_flag=True, help="Anonymize developer information in reports")
@click.option("--no-cache", is_flag=True, help="Disable caching (slower but always fresh)")
@click.option(
    "--validate-only", is_flag=True, help="Validate configuration without running analysis"
)
@click.option("--clear-cache", is_flag=True, help="Clear cache before running analysis")
@click.option(
    "--enable-qualitative",
    is_flag=True,
    help="Enable qualitative analysis (requires additional dependencies)",
)
@click.option(
    "--qualitative-only", is_flag=True, help="Run only qualitative analysis on existing commits"
)
@click.option(
    "--enable-pm", is_flag=True, help="Enable PM platform integration (overrides config setting)"
)
@click.option(
    "--pm-platform",
    multiple=True,
    help="Enable specific PM platforms (e.g., --pm-platform jira --pm-platform azure)",
)
@click.option(
    "--disable-pm", is_flag=True, help="Disable PM platform integration (overrides config setting)"
)
@click.option(
    "--no-rich",
    is_flag=True,
    default=True,
    help="Disable rich terminal output (use simple text progress instead)",
)
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="none",
    help="Enable logging with specified level (default: none)",
)
@click.option("--skip-identity-analysis", is_flag=True, help="Skip automatic identity analysis")
@click.option(
    "--apply-identity-suggestions",
    is_flag=True,
    help="Apply identity analysis suggestions without prompting",
)
@click.option(
    "--warm-cache", is_flag=True, help="Pre-warm cache with all commits for faster subsequent runs"
)
@click.option("--validate-cache", is_flag=True, help="Validate cache integrity and consistency")
@click.option(
    "--generate-csv",
    is_flag=True,
    help="Generate CSV reports (disabled by default, only narrative report is generated)",
)
@click.option(
    "--use-batch-classification/--use-legacy-classification",
    default=True,
    help=(
        "Use batch LLM classification on pre-fetched data (Step 2 of 2) - now the default behavior"
    ),
)
@click.option(
    "--force-fetch",
    "-f",
    is_flag=True,
    help=(
        "Force fetch fresh data even if cached data exists. "
        "Clears per-week cache entries and re-fetches all weeks. "
        "Alias: -f"
    ),
)
@click.option(
    "--progress-style",
    type=click.Choice(["rich", "simple", "auto"], case_sensitive=False),
    default="simple",
    help="Progress display style: rich (beautiful terminal UI), simple (tqdm), auto (detect)",
)
@click.option(
    "--cicd-metrics/--no-cicd-metrics",
    default=True,
    help="Enable CI/CD pipeline metrics collection (enabled by default, use --no-cicd-metrics to disable)",
)
@click.option(
    "--cicd-platforms",
    multiple=True,
    type=click.Choice(["github-actions"], case_sensitive=False),
    default=["github-actions"],
    help="CI/CD platforms to integrate (default: github-actions)",
)
@click.option(
    "--security-only",
    is_flag=True,
    help="Run only security analysis (skip productivity metrics)",
)
def analyze_subcommand(
    config: Path,
    weeks: int,
    output: Optional[Path],
    anonymize: bool,
    no_cache: bool,
    validate_only: bool,
    clear_cache: bool,
    enable_qualitative: bool,
    qualitative_only: bool,
    enable_pm: bool,
    pm_platform: tuple[str, ...],
    disable_pm: bool,
    no_rich: bool,
    log: str,
    skip_identity_analysis: bool,
    apply_identity_suggestions: bool,
    warm_cache: bool,
    validate_cache: bool,
    generate_csv: bool,
    use_batch_classification: bool,
    force_fetch: bool,
    progress_style: str,
    cicd_metrics: bool,
    cicd_platforms: tuple[str, ...],
    security_only: bool,
) -> None:
    """Run the complete analysis pipeline: collect → classify → report.

    \b
    This is the all-in-one command that internally runs all three pipeline
    stages in sequence:
      1. Collect  — fetch raw commits from git repositories into the cache
      2. Classify — run batch LLM classification on cached commits
      3. Report   — read classified commits and generate report files

    \b
    For finer-grained control you can run each stage independently:
      gfa collect -c config.yaml --weeks 4
      gfa classify -c config.yaml
      gfa report   -c config.yaml

    \b
    EXAMPLES:
      # Basic analysis of last 4 weeks (uses config.yaml by default)
      gitflow-analytics analyze --weeks 4

      # Use a custom configuration file
      gitflow-analytics analyze -c myconfig.yaml --weeks 4

      # Generate CSV reports with fresh data
      gitflow-analytics analyze --generate-csv --clear-cache

      # Quick validation of configuration
      gitflow-analytics analyze --validate-only

      # Analyze with qualitative insights
      gitflow-analytics analyze --enable-qualitative

      # Run only security analysis (requires security config)
      gitflow-analytics analyze --security-only

    \b
    OUTPUT FILES:
      - developer_metrics_YYYYMMDD.csv: Individual developer statistics
      - weekly_metrics_YYYYMMDD.csv: Week-by-week team metrics
      - narrative_report_YYYYMMDD.md: Executive summary and insights
      - comprehensive_export_YYYYMMDD.json: Complete data export

    \b
    PERFORMANCE TIPS:
      - Use --no-cache for latest data but slower performance
      - Use --clear-cache when configuration changes
      - Smaller --weeks values analyze faster
      - Enable caching for repeated analyses
    """
    analyze(
        config=config,
        weeks=weeks,
        output=output,
        anonymize=anonymize,
        no_cache=no_cache,
        validate_only=validate_only,
        clear_cache=clear_cache,
        enable_qualitative=enable_qualitative,
        qualitative_only=qualitative_only,
        enable_pm=enable_pm,
        pm_platform=pm_platform,
        disable_pm=disable_pm,
        no_rich=no_rich,
        log=log,
        skip_identity_analysis=skip_identity_analysis,
        apply_identity_suggestions=apply_identity_suggestions,
        warm_cache=warm_cache,
        validate_cache=validate_cache,
        generate_csv=generate_csv,
        use_batch_classification=use_batch_classification,
        force_fetch=force_fetch,
        progress_style=progress_style,
        cicd_metrics=cicd_metrics,
        cicd_platforms=cicd_platforms,
        security_only=security_only,
    )
