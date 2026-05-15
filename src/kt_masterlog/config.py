"""
Configuration dataclass for tuning runs.

Designed to be serializable (dataclasses.asdict → JSON), diffable
across experiments, and version-controllable alongside model builders.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class TunerConfig:
    """
    Complete specification of a tuning run.

    Parameters
    ----------
    project_name : str
        KerasTuner project directory name. Also used as the default
        master CSV prefix if ``master_csv_prefix`` is not set.
    output_dir : str
        Root directory for tuner state and the master CSV.
    strategy : str
        Tuning strategy key. Must exist in ``STRATEGY_REGISTRY``.
        Built-in: ``"bayesian"``, ``"random"``, ``"hyperband"``.
    max_trials : int
        Maximum number of trials (bayesian, random).
    objective_metric : str
        Metric name to optimize (e.g. ``"val_loss"``, ``"val_accuracy"``).
    objective_direction : str
        ``"min"`` or ``"max"``.
    search_epochs : int
        Max epochs per trial during the search phase.
    early_stop_patience : int
        Epochs without improvement before stopping a trial.
    early_stop_monitor : str
        Metric to monitor for early stopping. Defaults to ``objective_metric``.
    restore_best_weights : bool
        Whether early stopping restores weights from the best epoch.
    reduce_lr : bool
        Whether to add a ReduceLROnPlateau callback during search.
    master_csv_prefix : str or None
        Filename prefix for the master CSV. Defaults to ``project_name``.
    overwrite_master_csv : bool
        If True, delete an existing master CSV before starting.
    extra_fields : dict
        Static key-value pairs added to every CSV row (e.g. dataset name,
        git SHA, experiment tag).
    extra_callbacks : list
        Additional Keras callbacks injected into every trial.
    strategy_kwargs : dict
        Extra keyword arguments forwarded to the KerasTuner constructor
        (e.g. ``num_initial_points``, ``factor``, ``hyperband_iterations``).

    Examples
    --------
    >>> config = TunerConfig(
    ...     project_name="resnet_sweep",
    ...     strategy="hyperband",
    ...     strategy_kwargs={"max_epochs": 60, "factor": 3},
    ...     objective_metric="val_accuracy",
    ...     objective_direction="max",
    ... )
    >>> config.to_json("config.json")
    """

    project_name: str
    output_dir: str = "./tuner_runs"

    # Strategy
    strategy: str = "bayesian"
    max_trials: int = 30
    strategy_kwargs: dict[str, Any] = field(default_factory=dict)

    # Objective
    objective_metric: str = "val_loss"
    objective_direction: Literal["min", "max"] = "min"

    # Search training
    search_epochs: int = 50

    # Early stopping
    early_stop_patience: int = 15
    early_stop_monitor: str | None = None  # defaults to objective_metric
    restore_best_weights: bool = True

    # LR reduction
    reduce_lr: bool = False
    reduce_lr_patience: int = 8
    reduce_lr_factor: float = 0.5
    reduce_lr_min: float = 1e-6

    # Master CSV
    master_csv_prefix: str | None = None
    overwrite_master_csv: bool = True
    extra_fields: dict[str, Any] = field(default_factory=dict)

    # Callbacks
    extra_callbacks: list = field(default_factory=list)

    # Run registry — when True, kt-masterlog writes a per-run JSON manifest
    # to ~/.kt-masterlog/runs/ (or $KT_MASTERLOG_REGISTRY_DIR) so sister
    # tools like kt-masterviz can auto-discover runs. Disable for sandboxed
    # or CI environments where filesystem side effects are undesirable.
    register_run: bool = True

    def __post_init__(self) -> None:
        if self.early_stop_monitor is None:
            self.early_stop_monitor = self.objective_metric

    @property
    def master_csv_filename(self) -> str:
        prefix = self.master_csv_prefix or self.project_name
        return f"{prefix}_master_log.csv"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (excludes non-serializable callbacks)."""
        d = asdict(self)
        d.pop("extra_callbacks", None)
        return d

    def to_json(self, path: str) -> None:
        """Write config to a JSON file for reproducibility."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: str) -> TunerConfig:
        """Load config from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
