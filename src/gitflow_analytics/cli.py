"""Command-line interface for GitFlow Analytics."""

import logging
import sys

import click

from ._version import __version__

logger = logging.getLogger(__name__)


class AnalyzeAsDefaultGroup(click.Group):
    """Custom Click group that defaults to analyze when no explicit subcommand is provided.

    This allows 'gitflow-analytics -c config.yaml' to run analysis by default.
    """

    def parse_args(self, ctx: click.Context, args: list) -> list:
        """Override parse_args to default to analyze unless explicit subcommand provided."""
        if args and args[0] in self.list_commands(ctx):
            return super().parse_args(ctx, args)

        global_options = {"--version", "--help", "-h"}
        if args and args[0] in global_options:
            return super().parse_args(ctx, args)

        if args and args[0].startswith("-"):
            new_args = ["analyze"] + args
            return super().parse_args(ctx, new_args)

        return super().parse_args(ctx, args)


@click.group(cls=AnalyzeAsDefaultGroup, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="GitFlow Analytics")
@click.help_option("-h", "--help")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """GitFlow Analytics - Developer productivity insights from Git history.

    \b
    A comprehensive tool for analyzing Git repositories to generate developer
    productivity metrics, DORA metrics, and team insights without requiring
    external project management tools.

    \b
    QUICK START:
      1. Create a configuration file named config.yaml (see config-sample.yaml)
      2. Run analysis: gitflow-analytics --weeks 4
      3. View reports in the output directory

    \b
    COMMON WORKFLOWS:
      Analyze last 4 weeks:     gitflow-analytics --weeks 4
      Use custom config:        gitflow-analytics -c myconfig.yaml --weeks 4
      Clear cache and analyze:  gitflow-analytics --clear-cache
      Validate configuration:   gitflow-analytics --validate-only

    \b
    COMMANDS:
      analyze    Run complete pipeline: collect -> classify -> report (default)
      collect    Stage 1: Fetch raw commits into the weekly cache
      classify   Stage 2: Classify cached commits with batch LLM
      report     Stage 3: Generate reports from classified data
      install    Interactive installation wizard
      run        Interactive launcher with preferences
      aliases    Generate developer identity aliases using LLM
      identities Manage developer identity resolution
      train      Train ML models for commit classification
      fetch      Fetch external data (GitHub PRs, PM tickets)
      help       Show detailed help and documentation

    \b
    EXAMPLES:
      # Interactive installation
      gitflow-analytics install

      # Interactive launcher
      gitflow-analytics run

      # Generate developer aliases
      gitflow-analytics aliases --apply

      # Run analysis (uses config.yaml by default)
      gitflow-analytics --weeks 4

    \b
    For detailed command help: gitflow-analytics COMMAND --help
    For documentation: https://github.com/bobmatnyc/gitflow-analytics
    """
    if ctx.invoked_subcommand is None:
        if sys.stdin.isatty() and sys.stdout.isatty():
            from gitflow_analytics.cli_wizards.menu import show_main_menu

            show_main_menu()
        else:
            click.echo(ctx.get_help())
        ctx.exit(0)


# ---------------------------------------------------------------------------
# Register commands from sub-modules
# ---------------------------------------------------------------------------

from .cli_analysis import analyze_subcommand, register_analysis_commands  # noqa: E402
from .cli_setup import register_setup_commands  # noqa: E402
from .cli_training import register_training_commands  # noqa: E402

register_analysis_commands(cli)
register_setup_commands(cli)
register_training_commands(cli)

# Identity / alias commands from dedicated module
from .cli_identity import (  # noqa: E402
    alias_rename,
    aliases_command,
    create_alias_interactive,
    identities,
    list_developers,
    merge_identity,
    register_identity_commands,
)

register_identity_commands(cli)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
