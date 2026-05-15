"""
Multi-output example: tune a dual-head model (classification + regression).

Demonstrates:
    - Multi-output model builders
    - Strategy switching (bayesian vs hyperband)
    - Config serialization for reproducibility
    - Extending the strategy registry
    - Using extra_fields for experiment tracking
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Dense, Flatten,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from kt_masterlog import optimize, TunerConfig, STRATEGY_REGISTRY


def build_dual_head(hp):
    """Example multi-output model: classification head + regression head."""
    filters = hp.Choice("filters", [32, 64])
    dense_units = hp.Choice("dense_units", [64, 128])
    dropout = hp.Float("dropout", 0.2, 0.5, step=0.1)
    lr = hp.Choice("lr", [1e-3, 3e-4, 1e-4])

    inputs = Input(shape=(128, 128, 3))

    # Shared backbone
    x = Conv2D(filters, (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(filters * 2, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    # Classification branch
    c = Dense(dense_units, activation="relu")(x)
    c = Dropout(dropout)(c)
    class_output = Dense(1, activation="sigmoid", name="class_output")(c)

    # Regression branch
    r = Dense(dense_units, activation="relu")(x)
    r = Dropout(dropout)(r)
    reg_output = Dense(1, activation="linear", name="reg_output")(r)

    model = Model(inputs=inputs, outputs=[class_output, reg_output])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss={"class_output": "binary_crossentropy", "reg_output": "mae"},
        loss_weights={"class_output": 5.0, "reg_output": 0.05},
        metrics={"class_output": ["accuracy"], "reg_output": ["mae"]},
    )
    return model


# --- Config: Bayesian search ---
config_bayesian = TunerConfig(
    project_name="dual_head_bayesian",
    output_dir="./runs",
    strategy="bayesian",
    max_trials=30,
    strategy_kwargs={"num_initial_points": 5},
    search_epochs=50,
    early_stop_patience=15,
    extra_fields={"experiment": "architecture_search", "dataset": "utkface"},
)

# Save the config for reproducibility
config_bayesian.to_json("./runs/dual_head_bayesian_config.json")

# --- Config: Hyperband search (same model, different strategy) ---
config_hyperband = TunerConfig(
    project_name="dual_head_hyperband",
    output_dir="./runs",
    strategy="hyperband",
    strategy_kwargs={"max_epochs": 50, "factor": 3},
    early_stop_patience=10,
    reduce_lr=True,
    reduce_lr_patience=5,
    extra_fields={"experiment": "architecture_search", "dataset": "utkface"},
)


# --- Run (uncomment with real data) ---
# result = optimize(
#     builder_fn=build_dual_head,
#     train_data=train_dataset,
#     val_data=val_data,
#     config=config_bayesian,
#     steps_per_epoch=steps_per_epoch,
# )
#
# print(result.summary())
# result.save_summary("./runs/best_result.json")
# result.model.save("best_dual_head.keras")


# --- Extending the strategy registry ---
# If you have a custom tuner (e.g. from a fork or a research paper):
#
#   import my_custom_tuner
#   STRATEGY_REGISTRY["my_strategy"] = my_custom_tuner.MyTuner
#
#   config = TunerConfig(strategy="my_strategy", ...)
#   result = optimize(builder_fn=build_dual_head, ..., config=config)
