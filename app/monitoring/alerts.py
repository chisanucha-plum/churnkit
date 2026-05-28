"""Alert management."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AlertManager:
    """Manage system alerts."""

    def __init__(self):
        """Initialize alert manager."""
        self.alerts = []

    def create_alert(
        self, alert_type: str, severity: str, message: str, metadata: Dict[str, Any] = None
    ) -> None:
        """Create an alert."""
        alert = {
            "type": alert_type,
            "severity": severity,
            "message": message,
            "metadata": metadata or {},
        }

        self.alerts.append(alert)
        logger.warning(f"Alert created: {alert_type} - {message}")

    def get_alerts(self, severity: str = None) -> list:
        """Get alerts."""
        if severity:
            return [a for a in self.alerts if a["severity"] == severity]
        return self.alerts

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts.clear()
        logger.info("Alerts cleared")
