"""Distributed tracing."""

import logging

logger = logging.getLogger(__name__)


class TracingService:
    """Distributed tracing service."""

    def __init__(self):
        """Initialize tracing service."""
        self.tracer = None
        self._setup_tracer()

    def _setup_tracer(self) -> None:
        """Setup tracer."""
        try:
            from opentelemetry import trace

            self.tracer = trace.get_tracer(__name__)
            logger.info("Tracing service initialized")
        except ImportError:
            logger.warning("OpenTelemetry not installed, tracing disabled")

    def start_span(self, name: str):
        """Start a new span."""
        if self.tracer:
            return self.tracer.start_as_current_span(name)
        return None
