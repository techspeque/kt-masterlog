"""Unit tests for TunerConfig."""

import tensorflow as tf

from kt_masterlog.config import TunerConfig


class TestTunerConfig:
    def test_defaults(self):
        cfg = TunerConfig(project_name="test")
        assert cfg.strategy == "bayesian"
        assert cfg.max_trials == 30
        assert cfg.objective_metric == "val_loss"
        assert cfg.objective_direction == "min"
        assert cfg.early_stop_monitor == "val_loss"  # derived from objective_metric

    def test_early_stop_monitor_defaults_to_objective(self):
        cfg = TunerConfig(project_name="test", objective_metric="val_accuracy")
        assert cfg.early_stop_monitor == "val_accuracy"

    def test_early_stop_monitor_explicit_override(self):
        cfg = TunerConfig(
            project_name="test",
            objective_metric="val_accuracy",
            early_stop_monitor="val_loss",
        )
        assert cfg.early_stop_monitor == "val_loss"

    def test_master_csv_filename_uses_project_name(self):
        cfg = TunerConfig(project_name="my_sweep")
        assert cfg.master_csv_filename == "my_sweep_master_log.csv"

    def test_master_csv_filename_uses_prefix(self):
        cfg = TunerConfig(project_name="my_sweep", master_csv_prefix="exp_42")
        assert cfg.master_csv_filename == "exp_42_master_log.csv"

    def test_json_roundtrip(self, tmp_path):
        original = TunerConfig(
            project_name="roundtrip_test",
            strategy="hyperband",
            max_trials=10,
            strategy_kwargs={"max_epochs": 40, "factor": 3},
            extra_fields={"dataset": "cifar10"},
        )
        path = str(tmp_path / "config.json")
        original.to_json(path)

        loaded = TunerConfig.from_json(path)
        assert loaded.project_name == "roundtrip_test"
        assert loaded.strategy == "hyperband"
        assert loaded.max_trials == 10
        assert loaded.strategy_kwargs == {"max_epochs": 40, "factor": 3}
        assert loaded.extra_fields == {"dataset": "cifar10"}

    def test_to_dict_excludes_callbacks(self):
        callback = tf.keras.callbacks.Callback()
        cfg = TunerConfig(project_name="test", extra_callbacks=[callback])
        d = cfg.to_dict()
        assert "extra_callbacks" not in d
