"""
Structured result from a completed tuning run.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass
from typing import Any

import keras_tuner as kt
import tensorflow as tf


@dataclass
class TuningResult:
    """
    Everything produced by a tuning run.

    Attributes
    ----------
    model : tf.keras.Model
        Best model (compiled, untrained with final weights — call ``.fit()``
        for the production training loop).
    best_hps : dict[str, Any]
        Hyperparameter values from the best trial.
    elapsed_seconds : float
        Wall-clock time for the search.
    master_csv_path : str
        Path to the master CSV log.
    config_snapshot : dict[str, Any]
        The TunerConfig that produced this result (serialized).
    tuner : kt.engine.tuner.Tuner
        The underlying KerasTuner instance (for advanced inspection,
        e.g. ``tuner.results_summary()``).
    """

    model: tf.keras.Model
    best_hps: dict[str, Any]
    elapsed_seconds: float
    master_csv_path: str
    config_snapshot: dict[str, Any]
    tuner: kt.engine.tuner.Tuner

    @property
    def elapsed_formatted(self) -> str:
        """Human-readable elapsed time (HH:MM:SS)."""
        return str(datetime.timedelta(seconds=int(self.elapsed_seconds)))

    def summary(self) -> str:
        """One-line-per-item summary for logging or notebooks."""
        lines = [
            f"Search completed in {self.elapsed_formatted}",
            f"Master log: {self.master_csv_path}",
            f"Strategy: {self.config_snapshot.get('strategy', '?')}",
            "Best hyperparameters:",
        ]
        for k, v in self.best_hps.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def save_summary(self, path: str) -> None:
        """Write summary + config snapshot to a JSON file."""
        payload = {
            "best_hps": self.best_hps,
            "elapsed_seconds": self.elapsed_seconds,
            "elapsed_formatted": self.elapsed_formatted,
            "master_csv_path": self.master_csv_path,
            "config": self.config_snapshot,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
