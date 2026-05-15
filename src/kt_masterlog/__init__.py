"""
kt-masterlog: Master-log orchestration layer for KerasTuner.

One CSV per tuning run. Strategy-agnostic configuration. Structured results.
"""

from kt_masterlog.callbacks import MasterEpochLogger
from kt_masterlog.tuners import make_logging_tuner, STRATEGY_REGISTRY
from kt_masterlog.config import TunerConfig
from kt_masterlog.result import TuningResult
from kt_masterlog.core import optimize

__all__ = [
    "MasterEpochLogger",
    "make_logging_tuner",
    "optimize",
    "TunerConfig",
    "TuningResult",
    "STRATEGY_REGISTRY",
]

__version__ = "0.1.0"
