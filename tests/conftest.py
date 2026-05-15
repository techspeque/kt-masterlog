"""Shared fixtures for integration tests.

Provides a tiny synthetic classification dataset and a minimal builder
so every integration test runs in a couple of seconds on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture
def tiny_data():
    """64-sample / 4-feature binary classification dataset (deterministic)."""
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((64, 4)).astype("float32")
    y_train = rng.integers(0, 2, size=64).astype("int32")
    x_val = rng.standard_normal((32, 4)).astype("float32")
    y_val = rng.integers(0, 2, size=32).astype("int32")
    return x_train, y_train, x_val, y_val


@pytest.fixture
def builder():
    """Return a minimal KerasTuner builder function."""

    def _build(hp):
        units = hp.Choice("units", [4, 8])
        lr = hp.Choice("lr", [1e-2, 1e-3])

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(4,)),
                tf.keras.layers.Dense(units, activation="relu"),
                tf.keras.layers.Dense(2, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    return _build
