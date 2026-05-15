"""
Main orchestrator: the ``optimize()`` function.

This is the primary public API. It takes a model builder and a config,
runs the search, and returns a structured result.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from kt_masterlog.config import TunerConfig
from kt_masterlog.registry import register_run, update_run_status
from kt_masterlog.result import TuningResult
from kt_masterlog.tuners import STRATEGY_REGISTRY, make_logging_tuner

logger = logging.getLogger(__name__)


def optimize(
    builder_fn: Callable,
    train_data: Any,
    val_data: Any,
    config: TunerConfig,
    steps_per_epoch: int | None = None,
    validation_steps: int | None = None,
    search_kwargs: dict[str, Any] | None = None,
) -> TuningResult:
    """
    Run a full hyperparameter search and return the best model with metadata.

    Parameters
    ----------
    builder_fn : callable(hp) -> tf.keras.Model
        KerasTuner-compatible model builder. Receives an ``hp`` object
        and returns a compiled ``tf.keras.Model``.
    train_data : tf.data.Dataset, tuple, or generator
        Training data passed to ``tuner.search(x=...)``.
    val_data : tuple or tf.data.Dataset
        Validation data passed to ``tuner.search(validation_data=...)``.
    config : TunerConfig
        All tuning configuration.
    steps_per_epoch : int, optional
        Required when ``train_data`` is an infinite dataset/generator.
    validation_steps : int, optional
        Required when ``val_data`` is an infinite dataset/generator.
    search_kwargs : dict, optional
        Additional keyword arguments forwarded to ``tuner.search()``.
        These override any conflicting auto-generated kwargs.

    Returns
    -------
    TuningResult
        Contains the best model, hyperparameters, timing, and
        path to the master CSV.

    Raises
    ------
    ValueError
        If ``config.strategy`` is not found in ``STRATEGY_REGISTRY``.

    Examples
    --------
    >>> from kt_masterlog import optimize, TunerConfig
    >>>
    >>> def build_model(hp):
    ...     lr = hp.Choice("lr", [1e-3, 1e-4])
    ...     model = tf.keras.Sequential([...])
    ...     model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    ...     return model
    >>>
    >>> result = optimize(
    ...     builder_fn=build_model,
    ...     train_data=train_ds,
    ...     val_data=val_ds,
    ...     config=TunerConfig(project_name="my_sweep", max_trials=20),
    ... )
    >>> result.model.save("best_model.keras")
    """
    os.makedirs(config.output_dir, exist_ok=True)
    master_csv_path = os.path.join(config.output_dir, config.master_csv_filename)

    if config.overwrite_master_csv and os.path.exists(master_csv_path):
        os.remove(master_csv_path)
        logger.info("Removed existing master CSV: %s", master_csv_path)

    # --- Resolve tuner class ---
    strategy_key = config.strategy.lower()
    if strategy_key not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{config.strategy}'. "
            f"Registered: {list(STRATEGY_REGISTRY.keys())}. "
            f"Add custom strategies via STRATEGY_REGISTRY[key] = YourTunerClass."
        )

    base_class = STRATEGY_REGISTRY[strategy_key]
    LoggingTuner = make_logging_tuner(base_class)

    # --- Build tuner kwargs ---
    tuner_kwargs: dict[str, Any] = {
        "hypermodel": builder_fn,
        "objective": kt.Objective(config.objective_metric, config.objective_direction),
        "directory": config.output_dir,
        "project_name": config.project_name,
        "master_csv_path": master_csv_path,
        "master_extra_fields": config.extra_fields,
    }

    # max_trials applies to bayesian and random but not hyperband
    if strategy_key in ("bayesian", "random"):
        tuner_kwargs["max_trials"] = config.max_trials

    # Merge any strategy-specific kwargs (num_initial_points, factor, etc.)
    tuner_kwargs.update(config.strategy_kwargs)

    tuner = LoggingTuner(**tuner_kwargs)
    logger.info(
        "Initialized %s tuner for '%s' (%d max trials)",
        strategy_key,
        config.project_name,
        config.max_trials,
    )

    # --- Build callback stack ---
    callbacks = list(config.extra_callbacks)

    callbacks.append(
        EarlyStopping(
            monitor=config.early_stop_monitor,
            patience=config.early_stop_patience,
            restore_best_weights=config.restore_best_weights,
        )
    )

    if config.reduce_lr:
        callbacks.append(
            ReduceLROnPlateau(
                monitor=config.early_stop_monitor,
                factor=config.reduce_lr_factor,
                patience=config.reduce_lr_patience,
                min_lr=config.reduce_lr_min,
            )
        )

    # --- Assemble search kwargs ---
    final_search_kwargs: dict[str, Any] = {
        "x": train_data,
        "validation_data": val_data,
        "epochs": config.search_epochs,
        "callbacks": callbacks,
    }
    if steps_per_epoch is not None:
        final_search_kwargs["steps_per_epoch"] = steps_per_epoch
    if validation_steps is not None:
        final_search_kwargs["validation_steps"] = validation_steps

    # User overrides win
    if search_kwargs:
        final_search_kwargs.update(search_kwargs)

    # --- Register run (best-effort; never blocks the search) ---
    run_id = (
        register_run(config.project_name, master_csv_path)
        if config.register_run
        else None
    )

    # --- Run search ---
    logger.info("Starting search: %s", config.project_name)
    start = time.time()
    try:
        tuner.search(**final_search_kwargs)
    except BaseException:
        # Includes KeyboardInterrupt — the run is no longer "running",
        # but we don't distinguish interrupt from failure here. Readers
        # see status="failed" and can also notice the dead PID via
        # registry stale-check logic.
        update_run_status(run_id, "failed")
        raise
    elapsed = time.time() - start
    update_run_status(run_id, "completed")
    logger.info("Search complete in %.1fs", elapsed)

    # --- Extract best model ---
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)

    # --- Save config snapshot alongside results ---
    config_snapshot = config.to_dict()
    config_path = os.path.join(config.output_dir, f"{config.project_name}_config.json")
    config.to_json(config_path)
    logger.info("Config saved to %s", config_path)

    result = TuningResult(
        model=best_model,
        best_hps=best_hps.values,
        elapsed_seconds=elapsed,
        master_csv_path=master_csv_path,
        config_snapshot=config_snapshot,
        tuner=tuner,
    )

    logger.info("\n%s", result.summary())
    return result
