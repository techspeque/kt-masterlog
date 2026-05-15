"""Unit tests for MasterEpochLogger.

Includes a concurrency stress test that verifies the per-path file lock
keeps writes from multiple threads atomic and complete.
"""

import csv
import os
import threading
from collections import Counter

import numpy as np
import pytest

from kt_masterlog.callbacks import MasterEpochLogger


class TestMasterEpochLogger:
    def test_creates_csv_with_header(self, tmp_path):
        csv_path = str(tmp_path / "log.csv")
        logger = MasterEpochLogger(
            csv_path=csv_path,
            trial_id="trial_0",
            hps={"lr": 0.001, "units": 64},
        )

        logger.on_epoch_end(0, logs={"loss": 0.5, "val_loss": 0.6})

        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        assert rows[0]["trial_id"] == "trial_0"
        assert rows[0]["epoch"] == "1"
        assert rows[0]["lr"] == "0.001"
        assert rows[0]["units"] == "64"
        assert float(rows[0]["loss"]) == pytest.approx(0.5)

    def test_appends_multiple_epochs(self, tmp_path):
        csv_path = str(tmp_path / "log.csv")
        logger = MasterEpochLogger(
            csv_path=csv_path,
            trial_id="trial_1",
            hps={"lr": 0.01},
        )

        logger.on_epoch_end(0, logs={"loss": 1.0})
        logger.on_epoch_end(1, logs={"loss": 0.8})
        logger.on_epoch_end(2, logs={"loss": 0.6})

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 3
        assert [r["epoch"] for r in rows] == ["1", "2", "3"]

    def test_extra_fields_included(self, tmp_path):
        csv_path = str(tmp_path / "log.csv")
        logger = MasterEpochLogger(
            csv_path=csv_path,
            trial_id="trial_0",
            hps={"lr": 0.001},
            extra_fields={"dataset": "utkface", "git_sha": "abc123"},
        )

        logger.on_epoch_end(0, logs={"loss": 0.5})

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["dataset"] == "utkface"
        assert rows[0]["git_sha"] == "abc123"

    def test_handles_numpy_scalars(self, tmp_path):
        csv_path = str(tmp_path / "log.csv")
        logger = MasterEpochLogger(
            csv_path=csv_path,
            trial_id="trial_0",
            hps={},
        )

        logger.on_epoch_end(0, logs={"loss": np.float32(0.12345)})

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        assert float(rows[0]["loss"]) == pytest.approx(0.12345, rel=1e-4)


class TestConcurrentWrites:
    """KerasTuner can run parallel trials. The per-path file lock must
    keep concurrent writes atomic — no interleaved/partial rows, no lost
    rows, and every row must have all declared columns."""

    def test_no_lost_or_corrupted_rows_under_thread_pressure(self, tmp_path):
        csv_path = str(tmp_path / "concurrent.csv")
        n_threads = 16
        rows_per_thread = 50

        barrier = threading.Barrier(n_threads)

        def writer(trial_id: str):
            logger = MasterEpochLogger(
                csv_path=csv_path,
                trial_id=trial_id,
                hps={"lr": 0.001, "units": 64},
                extra_fields={"tag": "stress"},
            )
            barrier.wait()  # release all threads simultaneously
            for i in range(rows_per_thread):
                logger.on_epoch_end(i, logs={"loss": 0.5, "val_loss": 0.6})

        threads = [
            threading.Thread(target=writer, args=(f"trial_{i:03d}",))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == n_threads * rows_per_thread

        for r in rows:
            assert r["trial_id"].startswith("trial_")
            assert r["lr"] == "0.001"
            assert r["units"] == "64"
            assert r["tag"] == "stress"
            assert r["loss"] == "0.5"
            assert r["val_loss"] == "0.6"
            assert r["epoch"] in {str(i + 1) for i in range(rows_per_thread)}

        per_trial = Counter(r["trial_id"] for r in rows)
        assert len(per_trial) == n_threads
        for trial_id, count in per_trial.items():
            assert count == rows_per_thread, (
                f"{trial_id} wrote {count} rows, expected {rows_per_thread}"
            )
