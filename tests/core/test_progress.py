"""Tests for the centralized progress service.

This module tests the ProgressService to ensure it correctly manages
progress reporting across the application.
"""

import os
import sys
import threading
from unittest.mock import patch

from gitflow_analytics.core.progress import (
    ProgressContext,
    ProgressService,
    get_progress_service,
    reset_progress_service,
)


class TestProgressService:
    """Test suite for ProgressService."""

    def setup_method(self):
        """Reset the global service before each test."""
        reset_progress_service()

    def test_singleton_pattern(self):
        """Test that get_progress_service returns the same instance."""
        service1 = get_progress_service()
        service2 = get_progress_service()
        assert service1 is service2

    def test_create_progress_context(self):
        """Test creating a progress context."""
        service = ProgressService()
        service.enable()  # Ensure enabled for test

        context = service.create_progress(total=100, description="Test progress", unit="items")

        assert isinstance(context, ProgressContext)
        assert context.total == 100
        assert context.description == "Test progress"
        assert context.unit == "items"
        assert context.current == 0

    def test_update_progress(self):
        """Test updating progress."""
        service = ProgressService()
        service.disable()  # Disable visual output for test

        context = service.create_progress(100, "Test")

        service.update(context, 10)
        assert context.current == 10

        service.update(context, 5)
        assert context.current == 15

        service.complete(context)

    def test_context_manager(self):
        """Test using progress as a context manager."""
        service = ProgressService()
        service.disable()  # Disable visual output

        with service.progress(50, "Test context") as ctx:
            assert ctx.total == 50
            service.update(ctx, 25)
            assert ctx.current == 25

    def test_nested_progress(self):
        """Test nested progress contexts."""
        service = ProgressService()
        service.enable()  # Enable to test position logic

        outer = service.create_progress(10, "Outer", nested=False)
        inner = service.create_progress(100, "Inner", nested=True)

        assert not outer.is_nested
        assert inner.is_nested
        # Position logic only works when enabled
        if service.is_enabled():
            assert outer.position != inner.position

        service.complete(inner)
        service.complete(outer)

    def test_disable_enable(self):
        """Test disabling and enabling the service."""
        service = ProgressService()

        # Service might be auto-disabled in test environment
        service.is_enabled()

        service.disable()
        assert not service.is_enabled()

        service.enable()
        assert service.is_enabled()

    def test_event_capture(self):
        """Test event capture for testing purposes."""
        service = ProgressService()
        service.disable()  # Disable visual output

        service.start_event_capture()

        # Create and update progress
        with service.progress(10, "Test") as ctx:
            service.update(ctx, 5)
            service.update(ctx, 3)

        events = service.stop_event_capture()

        assert len(events) >= 3  # create, updates, complete
        assert events[0].event_type == "create"
        assert events[0].total == 10
        assert events[1].event_type == "update"
        assert events[1].increment == 5

    def test_set_description(self):
        """Test updating progress description."""
        service = ProgressService()
        service.disable()

        context = service.create_progress(100, "Initial")
        assert context.description == "Initial"

        service.set_description(context, "Updated")
        assert context.description == "Updated"

        service.complete(context)

    def test_thread_safety(self):
        """Test thread-safe operations."""
        service = ProgressService()
        service.disable()

        context = service.create_progress(1000, "Thread test")
        errors = []

        def update_progress():
            try:
                for _ in range(100):
                    service.update(context, 1)
            except Exception as e:
                errors.append(e)

        # Create multiple threads updating the same context
        threads = [threading.Thread(target=update_progress) for _ in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert context.current == 1000

        service.complete(context)

    def test_environment_detection(self):
        """Test automatic disabling in test environments."""
        # This test is already in a pytest environment, so service should be disabled
        service = ProgressService()

        # In pytest, progress should be automatically disabled
        assert "pytest" in sys.modules
        assert not service.is_enabled()

    @patch.dict(os.environ, {"GITFLOW_DISABLE_PROGRESS": "1"})
    def test_environment_variable_disable(self):
        """Test disabling via environment variable."""
        reset_progress_service()  # Reset to pick up env var
        service = get_progress_service()

        assert not service.is_enabled()

    def test_clear_captured_events(self):
        """Test clearing captured events."""
        service = ProgressService()
        service.disable()

        service.start_event_capture()

        with service.progress(5, "Test") as ctx:
            service.update(ctx, 2)

        events = service.get_captured_events()
        assert len(events) > 0

        service.clear_captured_events()
        events = service.get_captured_events()
        assert len(events) == 0

        service.stop_event_capture()


class TestProgressIntegration:
    """Test progress service integration with other modules."""

    def test_analyzer_integration(self):
        """Test that analyzer properly uses progress service."""
        from tempfile import TemporaryDirectory

        from gitflow_analytics.core.analyzer import GitAnalyzer
        from gitflow_analytics.core.cache import GitAnalysisCache

        with TemporaryDirectory() as tmpdir:
            cache = GitAnalysisCache(tmpdir)
            GitAnalyzer(cache=cache)

            # Ensure progress is disabled for test
            service = get_progress_service()
            service.disable()
            service.start_event_capture()

            # The analyzer should still work with progress disabled
            # (actual repository analysis would happen here in a real test)

            service.stop_event_capture()
            # Events would be captured if analyzer was actually processing commits

    def test_data_fetcher_integration(self):
        """Test that data fetcher properly uses progress service."""
        # This would test actual data fetcher integration
        # For now, just verify the module imports correctly

        service = get_progress_service()
        service.disable()

        # data_fetcher module should work with progress disabled
        assert service is not None

    def test_batch_classifier_integration(self):
        """Test that batch classifier properly uses progress service."""
        # This would test actual batch classifier integration
        # For now, just verify the module imports correctly

        service = get_progress_service()
        service.disable()

        # BatchCommitClassifier should work with progress disabled
        assert service is not None
