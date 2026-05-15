"""Integration tests for the optimize() orchestrator.

Covers behavior that's strategy-agnostic: error paths, the result API,
optional callbacks (extra_callbacks, reduce_lr), and dataset/step
passthrough. Strategy-specific tests live in test_strategies.py.
"""

from __future__ import annotations

import csv
import json
import os

import pytest
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau

import kt_masterlog.core as core_mod
from kt_masterlog import STRATEGY_REGISTRY, TunerConfig, optimize


class TestRegistryAndErrors:
    def test_strategy_registry_contains_builtins(self):
        assert "bayesian" in STRATEGY_REGISTRY
        assert "random" in STRATEGY_REGISTRY
        assert "hyperband" in STRATEGY_REGISTRY

    def test_unknown_strategy_raises(self, tmp_path, tiny_data, builder):
        x_train, y_train, x_val, y_val = tiny_data
        config = TunerConfig(
            project_name="bad_strategy",
            output_dir=str(tmp_path),
            strategy="nonexistent_strategy",
        )
        with pytest.raises(ValueError, match="Unknown strategy"):
            optimize(
                builder_fn=builder,
                train_data=x_train,
                val_data=(x_val, y_val),
                config=config,
                search_kwargs={"y": y_train, "verbose": 0},
            )


class TestResultAPI:
    def test_save_summary_roundtrip(self, tmp_path, tiny_data, builder):
        x_train, y_train, x_val, y_val = tiny_data
        config = TunerConfig(
            project_name="smoke_summary",
            output_dir=str(tmp_path),
            strategy="random",
            max_trials=2,
            search_epochs=1,
            early_stop_patience=10,
        )
        result = optimize(
            builder_fn=builder,
            train_data=x_train,
            val_data=(x_val, y_val),
            config=config,
            search_kwargs={"y": y_train, "verbose": 0},
        )

        summary_path = str(tmp_path / "summary.json")
        result.save_summary(summary_path)
        with open(summary_path) as f:
            payload = json.load(f)

        assert payload["best_hps"] == result.best_hps
        assert payload["master_csv_path"] == result.master_csv_path
        assert payload["config"]["strategy"] == "random"


class TestExtraCallbacks:
    def test_user_callback_is_invoked_during_search(
        self, tmp_path, tiny_data, builder
    ):
        x_train, y_train, x_val, y_val = tiny_data

        epoch_count = [0]

        class _Counter(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                epoch_count[0] += 1

        config = TunerConfig(
            project_name="extra_cb",
            output_dir=str(tmp_path),
            strategy="random",
            max_trials=2,
            search_epochs=2,
            early_stop_patience=10,
            extra_callbacks=[_Counter()],
        )
        optimize(
            builder_fn=builder,
            train_data=x_train,
            val_data=(x_val, y_val),
            config=config,
            search_kwargs={"y": y_train, "verbose": 0},
        )

        # 2 trials * 2 epochs each = up to 4 invocations; require at least one
        # to prove the user callback was wired in.
        assert epoch_count[0] >= 1, "extra_callbacks were never invoked"


class TestReduceLR:
    def test_reduce_lr_true_instantiates_plateau_callback(
        self, tmp_path, tiny_data, builder, monkeypatch
    ):
        """When reduce_lr=True, ReduceLROnPlateau is constructed exactly once
        per optimize() call with the configured parameters."""
        x_train, y_train, x_val, y_val = tiny_data
        instantiated = []

        class _SpyReduce(ReduceLROnPlateau):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                instantiated.append(kwargs)

        monkeypatch.setattr(core_mod, "ReduceLROnPlateau", _SpyReduce)

        config = TunerConfig(
            project_name="reduce_lr_on",
            output_dir=str(tmp_path),
            strategy="random",
            max_trials=1,
            search_epochs=1,
            early_stop_patience=10,
            reduce_lr=True,
            reduce_lr_patience=4,
            reduce_lr_factor=0.25,
            reduce_lr_min=1e-7,
        )
        optimize(
            builder_fn=builder,
            train_data=x_train,
            val_data=(x_val, y_val),
            config=config,
            search_kwargs={"y": y_train, "verbose": 0},
        )

        assert len(instantiated) == 1
        kwargs = instantiated[0]
        assert kwargs["factor"] == 0.25
        assert kwargs["patience"] == 4
        assert kwargs["min_lr"] == 1e-7

    def test_reduce_lr_false_does_not_instantiate(
        self, tmp_path, tiny_data, builder, monkeypatch
    ):
        x_train, y_train, x_val, y_val = tiny_data
        instantiated = []

        class _SpyReduce(ReduceLROnPlateau):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                instantiated.append(self)

        monkeypatch.setattr(core_mod, "ReduceLROnPlateau", _SpyReduce)

        config = TunerConfig(
            project_name="reduce_lr_off",
            output_dir=str(tmp_path),
            strategy="random",
            max_trials=1,
            search_epochs=1,
            early_stop_patience=10,
            reduce_lr=False,
        )
        optimize(
            builder_fn=builder,
            train_data=x_train,
            val_data=(x_val, y_val),
            config=config,
            search_kwargs={"y": y_train, "verbose": 0},
        )
        assert instantiated == []


class TestStepsPerEpochPassthrough:
    def test_steps_kwargs_reach_tuner_search(self, tmp_path, tiny_data, builder):
        """An infinite tf.data.Dataset.repeat() would hang fit() without
        steps_per_epoch — so a successful run with finite epochs proves
        the kwarg made it through to tuner.search()."""
        x_train, y_train, x_val, y_val = tiny_data

        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .batch(16)
            .repeat()
        )
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(16).repeat()

        config = TunerConfig(
            project_name="steps_test",
            output_dir=str(tmp_path),
            strategy="random",
            max_trials=1,
            search_epochs=2,
            early_stop_patience=10,
        )
        result = optimize(
            builder_fn=builder,
            train_data=train_ds,
            val_data=val_ds,
            config=config,
            steps_per_epoch=2,
            validation_steps=2,
            search_kwargs={"verbose": 0},
        )

        assert os.path.exists(result.master_csv_path)
        with open(result.master_csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2  # 1 trial × 2 epochs
