# kt-masterlog

Master-log orchestration layer for [KerasTuner](https://keras.io/keras_tuner/). One CSV per tuning run, strategy-agnostic configuration, structured results.

## The problem

KerasTuner scatters trial data across per-trial directories. Comparing hyperparameters against metrics across trials means writing custom aggregation scripts every time. Switching strategies (Bayesian → Hyperband → Random) requires swapping classes and rewriting constructor arguments. There's no structured result object — you print things, then lose them.

## What this does

**`MasterEpochLogger`** — A Keras callback that writes one flat CSV row per epoch per trial: trial ID, all hyperparameter values, and all metrics in a single file. Load it into pandas and you have your entire tuning run in one DataFrame.

**`make_logging_tuner()`** — Dynamically subclasses any KerasTuner strategy to inject the logger automatically. No separate subclass per strategy.

**`TunerConfig`** — A serializable dataclass holding every tuning knob. JSON-roundtrippable for reproducibility. Diffable across experiments.

**`optimize()`** — Single-call orchestrator. Takes a model builder and a config, returns a `TuningResult` with the best model, hyperparameters, timing, and the path to the master CSV.

## Install

With pip:

```bash
pip install kt-masterlog
```

With uv:

```bash
uv add kt-masterlog
```

Or from source:

```bash
git clone https://github.com/techspeque/kt-masterlog.git
cd kt-masterlog

# pip
pip install -e ".[dev]"

# uv
uv sync --group dev
```

## Quickstart

```python
from kt_masterlog import optimize, TunerConfig

def build_model(hp):
    lr = hp.Choice("lr", [1e-3, 3e-4, 1e-4])
    units = hp.Choice("units", [64, 128, 256])

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

result = optimize(
    builder_fn=build_model,
    train_data=x_train,
    val_data=(x_val, y_val),
    config=TunerConfig(
        project_name="mnist_sweep",
        max_trials=20,
    ),
    search_kwargs={"y": y_train},
)

print(result.best_hps)       # {'lr': 0.0003, 'units': 128}
print(result.elapsed_formatted)  # '0:04:12'

# Best model is compiled and ready for production training
result.model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))
```

## Switching strategies

```python
# Bayesian (default)
config = TunerConfig(project_name="sweep", strategy="bayesian", max_trials=30)

# Hyperband
config = TunerConfig(
    project_name="sweep",
    strategy="hyperband",
    strategy_kwargs={"max_epochs": 50, "factor": 3},
)

# Random search
config = TunerConfig(project_name="sweep", strategy="random", max_trials=50)
```

## Custom strategies

```python
from kt_masterlog import STRATEGY_REGISTRY

STRATEGY_REGISTRY["my_tuner"] = MyCustomTunerClass

config = TunerConfig(project_name="sweep", strategy="my_tuner")
```

## Config serialization

```python
config = TunerConfig(project_name="experiment_42", max_trials=30)

# Save
config.to_json("config.json")

# Load
config = TunerConfig.from_json("config.json")
```

## The master CSV

Each row in the master CSV contains:

| trial_id | epoch | hp_1 | hp_2 | ... | loss | val_loss | accuracy | val_accuracy |
|----------|-------|------|------|-----|------|----------|----------|--------------|
| 0001     | 1     | 0.001| 128  | ... | 0.82 | 0.91     | 0.72     | 0.68         |
| 0001     | 2     | 0.001| 128  | ... | 0.65 | 0.74     | 0.78     | 0.75         |
| 0002     | 1     | 0.0003| 64  | ... | 0.79 | 0.85     | 0.74     | 0.71         |

Add experiment-level metadata via `extra_fields`:

```python
config = TunerConfig(
    project_name="sweep",
    extra_fields={"dataset": "utkface", "git_sha": "a1b2c3d"},
)
```

## API reference

### `optimize(builder_fn, train_data, val_data, config, ...)`

Run a hyperparameter search. Returns a `TuningResult`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `builder_fn` | `callable(hp) → Model` | KerasTuner model builder |
| `train_data` | Dataset / tuple | Training data |
| `val_data` | Dataset / tuple | Validation data |
| `config` | `TunerConfig` | Tuning configuration |
| `steps_per_epoch` | `int`, optional | Steps per epoch (for generators/datasets) |
| `validation_steps` | `int`, optional | Validation steps (for generators/datasets) |
| `search_kwargs` | `dict`, optional | Extra kwargs for `tuner.search()` |

### `TunerConfig`

All fields have defaults except `project_name`. See [config.py](src/kt_masterlog/config.py) for the full list.

### `TuningResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `.model` | `tf.keras.Model` | Best model (compiled, untrained) |
| `.best_hps` | `dict` | Best hyperparameter values |
| `.elapsed_seconds` | `float` | Search wall-clock time |
| `.elapsed_formatted` | `str` | Human-readable time |
| `.master_csv_path` | `str` | Path to the master CSV |
| `.tuner` | `kt.Tuner` | Underlying KerasTuner instance |
| `.summary()` | `str` | Printable summary |
| `.save_summary(path)` | — | Write summary + config to JSON |

### `MasterEpochLogger`

Standalone Keras callback. Use directly if you don't need the full orchestrator:

```python
from kt_masterlog import MasterEpochLogger

logger = MasterEpochLogger(
    csv_path="training_log.csv",
    trial_id="run_1",
    hps={"lr": 0.001},
    extra_fields={"tag": "baseline"},
)

model.fit(..., callbacks=[logger])
```

### `make_logging_tuner(base_class)`

Wrap any KerasTuner class with master-log injection:

```python
from kt_masterlog import make_logging_tuner
import keras_tuner as kt

LoggingHyperband = make_logging_tuner(kt.Hyperband)
tuner = LoggingHyperband(..., master_csv_path="log.csv")
```

## Requirements

- Python 3.12 (tested; 3.13+ may work but is not yet verified)
- TensorFlow ≥ 2.12
- KerasTuner ≥ 1.4

## License

MIT
