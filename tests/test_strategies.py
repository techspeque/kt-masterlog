"""End-to-end strategy tests.

One test per built-in strategy. Random is the "deep" verification of
the full CSV column structure and TuningResult fields; bayesian and
hyperband are slim smoke tests that exercise their distinct code paths
in optimize() (strategy_kwargs forwarding, max_trials gating, etc.).
"""

from __future__ import annotations

import csv
import json
import os

from kt_masterlog import TunerConfig, optimize


class TestRandomSearch:
    def test_produces_master_csv_with_all_expected_columns(
        self, tmp_path, tiny_data, builder
    ):
        x_train, y_train, x_val, y_val = tiny_data

        config = TunerConfig(
            project_name="smoke_random",
            output_dir=str(tmp_path),
            strategy="random",
            max_trials=2,
            search_epochs=2,
            early_stop_patience=10,
            extra_fields={"dataset": "synthetic", "tag": "smoke"},
        )

        result = optimize(
            builder_fn=builder,
            train_data=x_train,
            val_data=(x_val, y_val),
            config=config,
            search_kwargs={"y": y_train, "verbose": 0},
        )

        assert os.path.exists(result.master_csv_path)
        with open(result.master_csv_path) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) >= 2
        trial_ids = {r["trial_id"] for r in rows}
        assert len(trial_ids) == 2

        first = rows[0]
        for required in ("trial_id", "epoch", "units", "lr", "loss", "val_loss"):
            assert required in first, f"missing column: {required}"
        assert first["dataset"] == "synthetic"
        assert first["tag"] == "smoke"

        assert isinstance(result.best_hps, dict)
        assert set(result.best_hps.keys()) == {"units", "lr"}
        assert result.best_hps["units"] in (4, 8)
        assert result.best_hps["lr"] in (1e-2, 1e-3)
        assert result.elapsed_seconds > 0
        assert isinstance(result.elapsed_formatted, str)
        assert result.config_snapshot["strategy"] == "random"

        config_path = os.path.join(tmp_path, "smoke_random_config.json")
        assert os.path.exists(config_path)
        with open(config_path) as f:
            saved = json.load(f)
        assert saved["project_name"] == "smoke_random"
        assert saved["strategy"] == "random"


class TestBayesianSearch:
    def test_runs_end_to_end_with_strategy_kwargs(
        self, tmp_path, tiny_data, builder
    ):
        """Bayesian uses max_trials + strategy_kwargs (num_initial_points)."""
        x_train, y_train, x_val, y_val = tiny_data

        config = TunerConfig(
            project_name="smoke_bayesian",
            output_dir=str(tmp_path),
            strategy="bayesian",
            max_trials=2,
            strategy_kwargs={"num_initial_points": 2},
            search_epochs=2,
            early_stop_patience=10,
        )
        result = optimize(
            builder_fn=builder,
            train_data=x_train,
            val_data=(x_val, y_val),
            config=config,
            search_kwargs={"y": y_train, "verbose": 0},
        )

        assert os.path.exists(result.master_csv_path)
        with open(result.master_csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len({r["trial_id"] for r in rows}) == 2
        assert result.config_snapshot["strategy"] == "bayesian"
        assert type(result.tuner).__name__ == "LoggingBayesianOptimization"


class TestHyperbandSearch:
    def test_runs_end_to_end_with_max_epochs_and_factor(
        self, tmp_path, tiny_data, builder
    ):
        """Hyperband ignores max_trials and reads max_epochs/factor from
        strategy_kwargs."""
        x_train, y_train, x_val, y_val = tiny_data

        config = TunerConfig(
            project_name="smoke_hyperband",
            output_dir=str(tmp_path),
            strategy="hyperband",
            strategy_kwargs={
                "max_epochs": 2,
                "factor": 3,
                "hyperband_iterations": 1,
            },
            search_epochs=2,
            early_stop_patience=10,
        )
        result = optimize(
            builder_fn=builder,
            train_data=x_train,
            val_data=(x_val, y_val),
            config=config,
            search_kwargs={"y": y_train, "verbose": 0},
        )

        assert os.path.exists(result.master_csv_path)
        with open(result.master_csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        assert result.config_snapshot["strategy"] == "hyperband"
        assert type(result.tuner).__name__ == "LoggingHyperband"
