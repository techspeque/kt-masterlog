"""
Strategy-agnostic tuner factory.

Dynamically subclasses any KerasTuner tuner class to inject
MasterEpochLogger into every trial's callback list, without
requiring a separate subclass per strategy.
"""

from __future__ import annotations

import logging
from typing import Any

import keras_tuner as kt

from kt_masterlog.callbacks import MasterEpochLogger

logger = logging.getLogger(__name__)

# Built-in strategy registry. Users can extend this before calling optimize().
STRATEGY_REGISTRY: dict[str, type[kt.engine.tuner.Tuner]] = {
    "bayesian": kt.BayesianOptimization,
    "hyperband": kt.Hyperband,
    "random": kt.RandomSearch,
}

# Cache to avoid creating duplicate classes
_tuner_class_cache: dict[str, type] = {}


def make_logging_tuner(base_class: type[kt.engine.tuner.Tuner]) -> type:
    """
    Wrap any KerasTuner strategy class with master-log injection.

    The returned class accepts an additional ``master_csv_path`` kwarg.
    When set, every trial automatically gets a ``MasterEpochLogger``
    appended to its callbacks.

    Parameters
    ----------
    base_class : type
        A KerasTuner tuner class (e.g. ``kt.BayesianOptimization``).

    Returns
    -------
    type
        A subclass with logging injection in ``run_trial``.

    Examples
    --------
    >>> LoggingBayesian = make_logging_tuner(kt.BayesianOptimization)
    >>> tuner = LoggingBayesian(
    ...     hypermodel=build_fn,
    ...     objective="val_loss",
    ...     max_trials=20,
    ...     master_csv_path="./tuning_log.csv",
    ... )
    """
    class_name = base_class.__name__
    if class_name in _tuner_class_cache:
        return _tuner_class_cache[class_name]

    class _LoggingTuner(base_class):  # type: ignore[misc]
        def __init__(
            self,
            *args: Any,
            master_csv_path: str | None = None,
            master_extra_fields: dict[str, Any] | None = None,
            **kwargs: Any,
        ):
            super().__init__(*args, **kwargs)
            self.master_csv_path = master_csv_path
            self.master_extra_fields = master_extra_fields or {}

        def run_trial(self, trial: Any, *args: Any, **kwargs: Any) -> Any:
            if self.master_csv_path:
                hps = trial.hyperparameters.values
                epoch_logger = MasterEpochLogger(
                    csv_path=self.master_csv_path,
                    trial_id=trial.trial_id,
                    hps=hps,
                    extra_fields=self.master_extra_fields,
                )
                callbacks = list(kwargs.get("callbacks", []))
                callbacks.append(epoch_logger)
                kwargs["callbacks"] = callbacks
                logger.debug(
                    "Injected MasterEpochLogger for trial %s", trial.trial_id
                )

            return super().run_trial(trial, *args, **kwargs)

    _LoggingTuner.__name__ = f"Logging{class_name}"
    _LoggingTuner.__qualname__ = f"Logging{class_name}"
    _tuner_class_cache[class_name] = _LoggingTuner
    return _LoggingTuner
