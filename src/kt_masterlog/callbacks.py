"""
Per-epoch master CSV logger.

Writes one row per epoch per trial, combining trial identity,
hyperparameter values, and all training/validation metrics into
a single flat CSV. This is the core primitive — everything else
in the package exists to wire this into KerasTuner cleanly.
"""

from __future__ import annotations

import csv
import logging
import os
import threading
from typing import Any

import tensorflow as tf

logger = logging.getLogger(__name__)

_file_locks: dict[str, threading.Lock] = {}
_fieldname_cache: dict[str, list[str]] = {}


def _get_lock(path: str) -> threading.Lock:
    """Thread-safe file lock per CSV path (handles parallel trial execution)."""
    if path not in _file_locks:
        _file_locks[path] = threading.Lock()
    return _file_locks[path]


class MasterEpochLogger(tf.keras.callbacks.Callback):
    """
    Keras callback that appends one CSV row per epoch.

    Each row contains:
        trial_id | epoch | hp_1 | hp_2 | ... | loss | val_loss | metric_1 | ...

    Parameters
    ----------
    csv_path : str
        Path to the master CSV file. Created on first write.
    trial_id : str
        Identifier for the current trial (injected by the tuner).
    hps : dict[str, Any]
        Hyperparameter values for this trial.
    extra_fields : dict[str, Any], optional
        Additional static fields to include in every row
        (e.g. dataset name, experiment tag).
    """

    def __init__(
        self,
        csv_path: str,
        trial_id: str,
        hps: dict[str, Any],
        extra_fields: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.trial_id = trial_id
        self.hps = hps
        self.extra_fields = extra_fields or {}

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}

        row: dict[str, Any] = {"trial_id": self.trial_id, "epoch": epoch + 1}
        row.update(self.extra_fields)
        row.update(self.hps)
        row.update({k: _safe_scalar(v) for k, v in logs.items()})

        lock = _get_lock(self.csv_path)
        with lock:
            file_exists = os.path.exists(self.csv_path)

            # Lock fieldnames to first write for stable column ordering.
            # New columns from later trials are silently dropped (logged below).
            if self.csv_path not in _fieldname_cache:
                _fieldname_cache[self.csv_path] = list(row.keys())

            fieldnames = _fieldname_cache[self.csv_path]
            extra_keys = set(row.keys()) - set(fieldnames)
            if extra_keys:
                logger.debug(
                    "Trial %s has columns not in header (ignored): %s",
                    self.trial_id,
                    extra_keys,
                )

            with open(self.csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=fieldnames, extrasaction="ignore"
                )
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)


def _safe_scalar(value: Any) -> Any:
    """Convert tensors / numpy scalars to plain Python types for CSV writing."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "item"):
        return value.item()
    return value
