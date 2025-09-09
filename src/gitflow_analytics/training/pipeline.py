"""Training pipeline for commit classification using PM platform data."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from ..classification.classifier import CommitClassifier
from ..config import Config
from ..core.analyzer import GitAnalyzer
from ..core.cache import GitAnalysisCache

logger = logging.getLogger(__name__)

# Database models for training data
TrainingBase = declarative_base()


class TrainingSession(TrainingBase):
    """Store training session metadata."""

    __tablename__ = "training_sessions"

    id = Column(String, primary_key=True)
    name = Column(String)
    created_at = Column(DateTime)
    model_type = Column(String)
    training_examples = Column(Integer)
    validation_split = Column(Float)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    model_path = Column(String)
    config_hash = Column(String)


class TrainingData(TrainingBase):
    """Store individual training examples."""

    __tablename__ = "training_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("training_sessions.id"))
    commit_hash = Column(String, index=True)
    repository = Column(String)
    message = Column(Text)
    author = Column(String)
    timestamp = Column(DateTime)
    files_changed = Column(Integer)
    insertions = Column(Integer)
    deletions = Column(Integer)
    ticket_id = Column(String)
    ticket_type = Column(String)
    ticket_platform = Column(String)
    label = Column(String)
    confidence = Column(Float)
    created_at = Column(DateTime)


class CommitClassificationTrainer:
    """Train commit classification models using PM platform data."""

    # Mapping from PM ticket types to classification categories
    TICKET_TYPE_MAPPING = {
        # Bug types
        "bug": "bug_fix",
        "defect": "bug_fix",
        "issue": "bug_fix",
        "incident": "bug_fix",
        "problem": "bug_fix",
        # Feature types
        "feature": "feature",
        "story": "feature",
        "user story": "feature",
        "new feature": "feature",
        "enhancement": "feature",
        "epic": "feature",
        "historia": "feature",  # EWTN custom type (Spanish for Story)
        # Task/maintenance types
        "task": "maintenance",
        "chore": "maintenance",
        "subtask": "maintenance",
        "sub-task": "maintenance",
        # Documentation types
        "documentation": "documentation",
        "docs": "documentation",
        # Improvement/refactoring types
        "improvement": "refactor",
        "refactoring": "refactor",
        "technical debt": "refactor",
        "optimization": "refactor",
        # Test types
        "test": "test",
        "testing": "test",
        "qa": "test",
        # Other types
        "security": "security",
        "hotfix": "hotfix",
        "research": "other",
        "spike": "other",
    }

    def __init__(
        self,
        config: Config,
        cache: GitAnalysisCache,
        orchestrator: Any,
        training_config: Optional[dict[str, Any]] = None,
    ):
        """Initialize the training pipeline.

        Args:
            config: GitFlow Analytics configuration
            cache: Cache instance
            orchestrator: Integration orchestrator with PM platforms
            training_config: Training-specific configuration
        """
        self.config = config
        self.cache = cache
        self.orchestrator = orchestrator
        self.training_config = training_config or {}

        # Initialize database for training data
        self.db_path = cache.cache_dir / "training_data.db"
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        TrainingBase.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # Initialize classifier
        self.classifier = CommitClassifier(
            config=(
                config.analysis.commit_classification.__dict__
                if hasattr(config.analysis, "commit_classification")
                else {}
            ),
            cache_dir=cache.cache_dir,
        )

        logger.info(f"Initialized training pipeline with cache at {cache.cache_dir}")

    def train(
        self, repositories: list[Any], since: datetime, session_name: Optional[str] = None
    ) -> dict[str, Any]:
        """Train a classification model using PM platform data.

        Args:
            repositories: List of repository configurations
            since: Start date for commit extraction
            session_name: Optional name for this training session

        Returns:
            Training results dictionary
        """
        session_id = self._create_training_session(session_name)

        try:
            # Step 1: Extract commits with ticket references
            logger.info("Extracting commits with ticket references...")
            labeled_commits = self._extract_labeled_commits(repositories, since)

            if len(labeled_commits) < self.training_config.get("min_training_examples", 50):
                raise ValueError(
                    f"Insufficient training data: {len(labeled_commits)} examples found, "
                    f"minimum {self.training_config.get('min_training_examples', 50)} required"
                )

            # Step 2: Store training data
            logger.info(f"Storing {len(labeled_commits)} training examples...")
            self._store_training_data(session_id, labeled_commits)

            # Step 3: Train the model
            logger.info("Training classification model...")
            training_data = [(commit["commit_data"], commit["label"]) for commit in labeled_commits]
            results = self.classifier.train_model(
                training_data, validation_split=self.training_config.get("validation_split", 0.2)
            )

            # Step 4: Update session with results
            self._update_training_session(session_id, results, len(labeled_commits))

            # Step 5: Save training data CSV if requested
            if self.training_config.get("save_training_data", False):
                self._export_training_data(session_id)

            return {
                "session_id": session_id,
                "training_examples": len(labeled_commits),
                "accuracy": results.get("accuracy", 0.0),
                "results": results,
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._mark_session_failed(session_id, str(e))
            raise

    def _extract_labeled_commits(
        self, repositories: list[Any], since: datetime
    ) -> list[dict[str, Any]]:
        """Extract commits with PM platform labels.

        Args:
            repositories: List of repository configurations
            since: Start date for commit extraction

        Returns:
            List of labeled commit dictionaries
        """
        labeled_commits = []
        analyzer = GitAnalyzer(
            self.cache,
            batch_size=getattr(self.config.analysis, "batch_size", 1000),
            allowed_ticket_platforms=getattr(
                self.config.analysis, "allowed_ticket_platforms", None
            ),
            story_point_patterns=getattr(self.config.analysis, "story_point_patterns", None),
        )

        for repo_config in repositories:
            if not repo_config.path.exists():
                logger.warning(f"Repository path does not exist: {repo_config.path}")
                continue

            logger.info(f"Analyzing repository: {repo_config.path}")

            # Extract commits
            try:
                commits = analyzer.analyze_repository(
                    repo_config.path, since=since, branch=repo_config.branch
                )

                # Filter commits with ticket references
                for commit in commits:
                    ticket_refs = commit.get("ticket_references", [])
                    if not ticket_refs:
                        continue

                    # Get ticket data from PM platforms
                    ticket_data = self._fetch_ticket_data(ticket_refs)
                    if not ticket_data:
                        continue

                    # Determine label from ticket type
                    label = self._determine_label(ticket_data)
                    if label:
                        # Normalize commit data to ensure files_changed is a list
                        normalized_commit = self._normalize_commit_data(commit)
                        labeled_commits.append(
                            {
                                "commit_data": normalized_commit,
                                "ticket_data": ticket_data,
                                "label": label,
                                "repository": repo_config.name,
                            }
                        )

            except Exception as e:
                logger.error(f"Failed to analyze repository {repo_config.path}: {e}")
                continue

        return labeled_commits

    def _normalize_commit_data(self, commit: dict[str, Any]) -> dict[str, Any]:
        """Normalize commit data to ensure consistency.

        Args:
            commit: Original commit data

        Returns:
            Normalized commit data with files_changed as a list
        """
        normalized = commit.copy()

        # Ensure files_changed is a list
        files_changed = commit.get("files_changed", [])
        if isinstance(files_changed, int):
            # If it's an integer count, we can't reconstruct the file list
            # Store the count separately and use empty list for files
            normalized["files_changed_count"] = files_changed
            normalized["files_changed"] = []
        elif isinstance(files_changed, list):
            # If it's already a list, keep it and also store the count
            normalized["files_changed"] = files_changed
            normalized["files_changed_count"] = len(files_changed)
        else:
            # Fallback for unexpected types
            normalized["files_changed"] = []
            normalized["files_changed_count"] = 0

        return normalized

    def _fetch_ticket_data(self, ticket_refs: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Fetch ticket data from PM platforms.

        Args:
            ticket_refs: List of ticket references

        Returns:
            List of ticket data dictionaries
        """
        if not self.orchestrator.pm_orchestrator:
            return []

        # Get list of configured platforms
        configured_platforms = self.orchestrator.pm_orchestrator.get_active_platforms()
        ticket_data = []

        for ref in ticket_refs:
            platform = ref.get("platform", "")
            ticket_id = ref.get("id", "")

            if not platform or not ticket_id:
                continue

            # Skip platforms that aren't configured
            if platform not in configured_platforms:
                logger.debug(f"Skipping ticket {ticket_id} from unconfigured platform {platform}")
                continue

            try:
                # Fetch ticket from PM platform
                tickets = self.orchestrator.pm_orchestrator.get_issues_by_keys(
                    platform, [ticket_id]
                )

                if tickets and ticket_id in tickets:
                    ticket = tickets[ticket_id]
                    ticket_data.append(
                        {
                            "id": ticket_id,
                            "platform": platform,
                            "type": ticket.issue_type.value if ticket.issue_type else "unknown",
                            "title": ticket.title,
                            "status": ticket.status.value if ticket.status else "unknown",
                        }
                    )

            except Exception as e:
                logger.warning(f"Failed to fetch ticket {ticket_id} from {platform}: {e}")
                continue

        return ticket_data

    def _determine_label(self, ticket_data: list[dict[str, Any]]) -> Optional[str]:
        """Determine classification label from ticket data.

        Args:
            ticket_data: List of ticket data dictionaries

        Returns:
            Classification label or None
        """
        if not ticket_data:
            return None

        # Count ticket types
        type_counts = {}
        for ticket in ticket_data:
            ticket_type = ticket.get("type", "").lower()
            mapped_type = self.TICKET_TYPE_MAPPING.get(ticket_type, None)

            if mapped_type:
                type_counts[mapped_type] = type_counts.get(mapped_type, 0) + 1

        if not type_counts:
            return None

        # Return most common type
        return max(type_counts.items(), key=lambda x: x[1])[0]

    def _create_training_session(self, name: Optional[str] = None) -> str:
        """Create a new training session.

        Args:
            name: Optional session name

        Returns:
            Session ID
        """
        import uuid

        session_id = str(uuid.uuid4())
        session = TrainingSession(
            id=session_id,
            name=name or f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now(timezone.utc),
            model_type=self.training_config.get("model_type", "random_forest"),
            validation_split=self.training_config.get("validation_split", 0.2),
        )

        with self.Session() as db_session:
            db_session.add(session)
            db_session.commit()

        return session_id

    def _store_training_data(self, session_id: str, labeled_commits: list[dict[str, Any]]) -> None:
        """Store training data in database.

        Args:
            session_id: Training session ID
            labeled_commits: List of labeled commit data
        """
        with self.Session() as db_session:
            for item in labeled_commits:
                commit = item["commit_data"]
                ticket_data = item["ticket_data"]

                # Use first ticket for primary data
                primary_ticket = ticket_data[0] if ticket_data else {}

                # Handle files_changed being either int or list
                files_changed_value = commit.get("files_changed", 0)
                if isinstance(files_changed_value, int):
                    files_changed_count = files_changed_value
                elif isinstance(files_changed_value, list):
                    files_changed_count = len(files_changed_value)
                else:
                    files_changed_count = 0

                training_example = TrainingData(
                    session_id=session_id,
                    commit_hash=commit.get("hash", ""),
                    repository=item["repository"],
                    message=commit.get("message", ""),
                    author=commit.get("author_name", ""),
                    timestamp=commit.get("timestamp"),
                    files_changed=files_changed_count,
                    insertions=commit.get("insertions", 0),
                    deletions=commit.get("deletions", 0),
                    ticket_id=primary_ticket.get("id", ""),
                    ticket_type=primary_ticket.get("type", ""),
                    ticket_platform=primary_ticket.get("platform", ""),
                    label=item["label"],
                    confidence=1.0,  # High confidence for PM-based labels
                    created_at=datetime.now(timezone.utc),
                )

                db_session.add(training_example)

            db_session.commit()

    def _update_training_session(
        self, session_id: str, results: dict[str, Any], num_examples: int
    ) -> None:
        """Update training session with results.

        Args:
            session_id: Training session ID
            results: Training results
            num_examples: Number of training examples
        """
        with self.Session() as db_session:
            session = db_session.query(TrainingSession).filter_by(id=session_id).first()
            if session:
                session.training_examples = num_examples
                session.accuracy = results.get("accuracy", 0.0)
                session.precision = results.get("precision", 0.0)
                session.recall = results.get("recall", 0.0)
                session.f1_score = results.get("f1_score", 0.0)
                session.model_path = str(self.classifier.model_path)
                db_session.commit()

    def _mark_session_failed(self, session_id: str, error: str) -> None:
        """Mark a training session as failed.

        Args:
            session_id: Training session ID
            error: Error message
        """
        with self.Session() as db_session:
            session = db_session.query(TrainingSession).filter_by(id=session_id).first()
            if session:
                session.accuracy = -1.0  # Indicates failure
                db_session.commit()

    def _export_training_data(self, session_id: str) -> Path:
        """Export training data to CSV.

        Args:
            session_id: Training session ID

        Returns:
            Path to exported CSV file
        """
        output_path = self.cache.cache_dir / f"training_data_{session_id[:8]}.csv"

        with self.Session() as db_session:
            data = db_session.query(TrainingData).filter_by(session_id=session_id).all()

            rows = []
            for item in data:
                rows.append(
                    {
                        "commit_hash": item.commit_hash,
                        "repository": item.repository,
                        "message": item.message,
                        "author": item.author,
                        "timestamp": item.timestamp,
                        "files_changed": item.files_changed,
                        "insertions": item.insertions,
                        "deletions": item.deletions,
                        "ticket_id": item.ticket_id,
                        "ticket_type": item.ticket_type,
                        "ticket_platform": item.ticket_platform,
                        "label": item.label,
                        "confidence": item.confidence,
                    }
                )

            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)

            logger.info(f"Exported training data to {output_path}")

        return output_path

    def get_training_history(self) -> list[dict[str, Any]]:
        """Get history of training sessions.

        Returns:
            List of training session summaries
        """
        with self.Session() as db_session:
            sessions = (
                db_session.query(TrainingSession).order_by(TrainingSession.created_at.desc()).all()
            )

            history = []
            for session in sessions:
                history.append(
                    {
                        "id": session.id,
                        "name": session.name,
                        "created_at": session.created_at,
                        "model_type": session.model_type,
                        "training_examples": session.training_examples,
                        "accuracy": session.accuracy,
                        "precision": session.precision,
                        "recall": session.recall,
                        "f1_score": session.f1_score,
                    }
                )

            return history
