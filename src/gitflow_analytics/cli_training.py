"""Training commands for GitFlow Analytics CLI."""

import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import click

from .cli_utils import setup_logging
from .config import ConfigLoader
from .utils.date_utils import get_week_start


def register_training_commands(cli: click.Group) -> None:
    """Register training-related commands onto the CLI group."""
    cli.add_command(train)
    cli.add_command(training_statistics, name="train-stats")


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--weeks", "-w", type=int, default=12, help="Number of weeks to analyze (default: 12)"
)
@click.option(
    "--session-name", type=str, default=None, help="Optional name for the training session"
)
@click.option(
    "--min-examples",
    type=int,
    default=50,
    help="Minimum number of training examples required (default: 50)",
)
@click.option(
    "--validation-split",
    type=float,
    default=0.2,
    help="Fraction of data to use for validation (default: 0.2)",
)
@click.option(
    "--model-type",
    type=click.Choice(["random_forest", "svm", "naive_bayes"]),
    default="random_forest",
    help="Type of model to train (default: random_forest)",
)
@click.option(
    "--incremental", is_flag=True, help="Add to existing training data instead of starting fresh"
)
@click.option(
    "--save-training-data", is_flag=True, help="Save extracted training data as CSV for inspection"
)
@click.option("--clear-cache", is_flag=True, help="Clear cache before training")
@click.option(
    "--log",
    type=click.Choice(["none", "INFO", "DEBUG"], case_sensitive=False),
    default="INFO",
    help="Enable logging with specified level (default: INFO)",
)
def train(
    config: Path,
    weeks: int,
    session_name: Optional[str],
    min_examples: int,
    validation_split: float,
    model_type: str,
    incremental: bool,
    save_training_data: bool,
    clear_cache: bool,
    log: str,
) -> None:
    """Train custom ML models for improved commit classification.

    \b
    This command trains machine learning models on your repository's
    commit history to improve classification accuracy. The models learn:
    - Project-specific commit message patterns
    - Team coding conventions and terminology
    - Domain-specific keywords and concepts
    - File path patterns for different change types

    \b
    EXAMPLES:
      # Train on last 12 weeks of commits
      gitflow-analytics train -c config.yaml --weeks 12

      # Train with custom session name
      gitflow-analytics train -c config.yaml --session-name "q4-training"

      # Save training data for inspection
      gitflow-analytics train -c config.yaml --save-training-data

      # Incremental training on new data
      gitflow-analytics train -c config.yaml --incremental

    \b
    MODEL TYPES:
      - random_forest: Best general performance (default)
      - svm: Good for clear category boundaries
      - naive_bayes: Fast, works well with small datasets

    \b
    TRAINING PROCESS:
      1. Extracts commits with ticket references
      2. Fetches ticket types from PM platforms
      3. Maps ticket types to commit categories
      4. Trains model with cross-validation
      5. Saves model with performance metrics

    \b
    REQUIREMENTS:
      - PM platform integration configured
      - Minimum 50 commits with ticket references
      - scikit-learn and pandas dependencies
      - ~100MB disk space for model storage
    """
    from .core.cache import GitAnalysisCache
    from .integrations.orchestrator import IntegrationOrchestrator

    logger = setup_logging(log, __name__)

    try:
        click.echo("GitFlow Analytics - Commit Classification Training")
        click.echo("=" * 60)

        click.echo(f"Loading configuration from {config}...")
        cfg = ConfigLoader.load(config)

        # Validate PM integration is enabled
        if not cfg.pm_integration or not cfg.pm_integration.enabled:
            click.echo("Error: PM integration must be enabled for training")
            click.echo("   Add PM platform configuration to your config file:")
            click.echo("   pm_integration:")
            click.echo("     enabled: true")
            click.echo("     platforms:")
            click.echo("       jira:")
            click.echo("         enabled: true")
            click.echo("         config: {...}")
            sys.exit(1)

        active_platforms = [
            name for name, platform in cfg.pm_integration.platforms.items() if platform.enabled
        ]

        if not active_platforms:
            click.echo("Error: No PM platforms are enabled")
            click.echo(
                f"   Configure at least one platform: {list(cfg.pm_integration.platforms.keys())}"
            )
            sys.exit(1)

        click.echo(f"PM integration enabled with platforms: {', '.join(active_platforms)}")

        cache_dir = cfg.cache.directory
        if clear_cache:
            click.echo("Clearing cache...")
            import shutil

            if cache_dir.exists():
                shutil.rmtree(cache_dir)

        cache = GitAnalysisCache(cache_dir, ttl_hours=cfg.cache.ttl_hours)

        click.echo("Initializing integrations...")
        orchestrator = IntegrationOrchestrator(cfg, cache)

        if not orchestrator.pm_orchestrator or not orchestrator.pm_orchestrator.is_enabled():
            click.echo("Error: PM framework orchestrator failed to initialize")
            click.echo("   Check your PM platform configurations and credentials")
            sys.exit(1)

        click.echo(
            f"PM framework initialized with "
            f"{len(orchestrator.pm_orchestrator.get_active_platforms())} platforms"
        )

        repositories_to_analyze = cfg.repositories
        if cfg.github.organization and not repositories_to_analyze:
            click.echo(f"Discovering repositories from organization: {cfg.github.organization}")
            try:
                config_dir = Path(config).parent if config else Path.cwd()
                repos_dir = config_dir / "repos"

                def discovery_progress(repo_name: str, count: int) -> None:
                    click.echo(f"\r   Checking repositories... {count}", nl=False)

                discovered_repos = cfg.discover_organization_repositories(
                    clone_base_path=repos_dir, progress_callback=discovery_progress
                )
                repositories_to_analyze = discovered_repos

                click.echo("\r" + " " * 60 + "\r", nl=False)
                click.echo(f"Found {len(discovered_repos)} repositories in organization")
            except Exception as e:
                click.echo(f"Failed to discover repositories: {e}")
                sys.exit(1)

        if not repositories_to_analyze:
            click.echo("Error: No repositories configured for analysis")
            click.echo(
                "   Configure repositories in your config file or use GitHub organization discovery"
            )
            sys.exit(1)

        click.echo(f"Analyzing {len(repositories_to_analyze)} repositories")

        training_config = {
            "min_training_examples": min_examples,
            "validation_split": validation_split,
            "model_type": model_type,
            "save_training_data": save_training_data,
        }

        click.echo("Initializing training pipeline...")
        try:
            from .training.pipeline import CommitClassificationTrainer

            trainer = CommitClassificationTrainer(
                config=cfg,
                cache=cache,
                orchestrator=orchestrator,
                training_config=training_config,
            )
        except ImportError as e:
            click.echo(f"Error: {e}")
            click.echo("\nInstall training dependencies:")
            click.echo("   pip install scikit-learn")
            sys.exit(1)

        click.echo("\nStarting training session...")
        click.echo(f"   Time period: {weeks} weeks")
        click.echo(f"   Repositories: {len(repositories_to_analyze)}")
        click.echo(f"   Model type: {model_type}")
        click.echo(f"   Min examples: {min_examples}")
        click.echo(f"   Validation split: {validation_split:.1%}")

        start_time = time.time()

        try:
            current_time = datetime.now(timezone.utc)
            current_week_start = get_week_start(current_time)
            last_complete_week_start = current_week_start - timedelta(weeks=1)
            since = last_complete_week_start - timedelta(weeks=weeks - 1)

            results = trainer.train(
                repositories=repositories_to_analyze, since=since, session_name=session_name
            )

            training_time = time.time() - start_time

            click.echo("\nTraining completed successfully!")
            click.echo("=" * 50)
            click.echo(f"Session ID: {results['session_id']}")
            click.echo(f"Training examples: {results['training_examples']}")
            click.echo(f"Model accuracy: {results['accuracy']:.1%}")
            click.echo(f"Training time: {training_time:.1f} seconds")
            click.echo(f"Model saved to: {trainer.classifier.model_path}")

            if "results" in results and "class_metrics" in results["results"]:
                click.echo("\nPer-category performance:")
                for category, metrics in results["results"]["class_metrics"].items():
                    if isinstance(metrics, dict) and "precision" in metrics:
                        precision = metrics["precision"]
                        recall = metrics["recall"]
                        f1 = metrics["f1-score"]
                        support = metrics["support"]
                        click.echo(
                            f"   {category:12} - P: {precision:.3f}, R: {recall:.3f},"
                            f" F1: {f1:.3f} (n={support})"
                        )

            if orchestrator.pm_orchestrator:
                platforms = orchestrator.pm_orchestrator.get_active_platforms()
                if platforms:
                    click.echo(f"\nPM platforms used: {', '.join(platforms)}")

            if save_training_data:
                try:
                    training_data_path = trainer._export_training_data(results["session_id"])
                    click.echo(f"Training data saved to: {training_data_path}")
                except Exception as e:
                    click.echo(f"Warning: Failed to save training data: {e}")

            click.echo("\nNext steps:")
            click.echo("   1. Review the training metrics above")
            click.echo("   2. Test the model with 'gitflow-analytics analyze --enable-ml'")
            click.echo("   3. Monitor model performance and retrain as needed")
            click.echo("   4. Use 'gitflow-analytics train-stats' to view training history")

        except ValueError as e:
            click.echo(f"\nTraining failed: {e}")
            if "Insufficient training data" in str(e):
                click.echo("\nSuggestions to get more training data:")
                click.echo("   - Increase --weeks to analyze more history")
                click.echo("   - Ensure commits reference ticket IDs (e.g., PROJ-123)")
                click.echo("   - Check PM platform connectivity and permissions")
                click.echo("   - Lower --min-examples threshold (not recommended)")
            sys.exit(1)

        except Exception as e:
            click.echo(f"\nTraining failed with error: {e}")
            if log.upper() == "DEBUG":
                traceback.print_exc()
            sys.exit(1)

    except Exception as e:
        click.echo(f"\nConfiguration or setup error: {e}")
        if log.upper() == "DEBUG":
            traceback.print_exc()
        sys.exit(1)


@click.command(name="train-stats")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def training_statistics(config: Path) -> None:
    """Display ML model training statistics and performance history.

    \b
    Shows comprehensive training metrics:
    - Total training sessions and success rate
    - Model accuracy and validation scores
    - Training data statistics
    - Best performing model details
    - Recent training session results

    \b
    EXAMPLES:
      # View training statistics
      gitflow-analytics train-stats -c config.yaml

    \b
    Use this to:
    - Monitor model performance over time
    - Identify when retraining is needed
    - Compare different model versions
    """
    try:
        from .core.cache import GitAnalysisCache
        from .training.pipeline import CommitClassificationTrainer

        cfg = ConfigLoader.load(config)
        cache = GitAnalysisCache(cfg.cache.directory)

        trainer = CommitClassificationTrainer(
            config=cfg,
            cache=cache,
            orchestrator=None,
            training_config={},
        )

        stats = trainer.get_training_statistics()

        click.echo("Training Statistics")
        click.echo("=" * 40)
        click.echo(f"Total sessions: {stats['total_sessions']}")
        click.echo(f"Completed sessions: {stats['completed_sessions']}")
        click.echo(f"Failed sessions: {stats['failed_sessions']}")
        click.echo(f"Total models: {stats['total_models']}")
        click.echo(f"Active models: {stats['active_models']}")
        click.echo(f"Training examples: {stats['total_training_examples']}")

        if stats["latest_session"]:
            latest = stats["latest_session"]
            click.echo("\nLatest Session:")
            click.echo(f"   ID: {latest['session_id']}")
            click.echo(f"   Status: {latest['status']}")
            if latest["accuracy"]:
                click.echo(f"   Accuracy: {latest['accuracy']:.1%}")
            if latest["training_time_minutes"]:
                click.echo(f"   Training time: {latest['training_time_minutes']:.1f} minutes")

        if stats["best_model"]:
            best = stats["best_model"]
            click.echo("\nBest Model:")
            click.echo(f"   ID: {best['model_id']}")
            click.echo(f"   Version: {best['version']}")
            click.echo(f"   Accuracy: {best['accuracy']:.1%}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
