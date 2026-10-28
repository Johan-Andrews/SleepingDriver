"""Run logging to CSV time-series and JSON metadata files."""

from __future__ import annotations
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LogStep:
    time_s: float
    ego_speed_kmh: float
    ego_accel_mps2: float
    ego_brake_intensity: float
    follower_speed_kmh: float
    gap_m: float
    ttc_s: float
    state: str
    drowsiness_score: float
    risk_level: str


class RunLogger:
    """Logs simulation data to CSV and metadata to JSON."""

    def __init__(self, log_dir: str = "data/runs", run_id: Optional[str] = None) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.csv_path = self.log_dir / f"run_{self.run_id}.csv"
        self.json_path = self.log_dir / f"run_{self.run_id}.json"

        self._records: List[LogStep] = []
        self._metadata: Dict[str, Any] = {}

    def log_step(self, step: LogStep) -> None:
        """Buffer a single time-series step."""
        self._records.append(step)

    def set_metadata(self, key: str, value: Any) -> None:
        """Store metadata key-value."""
        self._metadata[key] = value

    def save(self) -> Tuple[str, str]:
        """Write out buffered time-series to CSV and metadata to JSON."""
        # 1. Write CSV
        fieldnames = [
            "time_s",
            "ego_speed_kmh",
            "ego_accel_mps2",
            "ego_brake_intensity",
            "follower_speed_kmh",
            "gap_m",
            "ttc_s",
            "state",
            "drowsiness_score",
            "risk_level",
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self._records:
                writer.writerow(asdict(record))

        # 2. Write JSON
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2)

        logger.info("Saved run logs to '%s' and '%s'.", self.csv_path, self.json_path)
        return str(self.csv_path), str(self.json_path)
