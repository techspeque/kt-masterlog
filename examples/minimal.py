"""
Minimal example: tune a simple classifier on MNIST.

Demonstrates the core workflow:
    1. Define a builder function
    2. Create a TunerConfig
    3. Call optimize()
    4. Use the result
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import mnist

from kt_masterlog import optimize, TunerConfig


# --- Data ---
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0


# --- Model builder (standard KerasTuner signature) ---
def build_model(hp):
    units = hp.Choice("units", [64, 128, 256])
    dropout = hp.Float("dropout", 0.1, 0.5, step=0.1)
    lr = hp.Choice("lr", [1e-3, 3e-4, 1e-4])

    model = tf.keras.Sequential(
        [
            Flatten(input_shape=(28, 28)),
            Dense(units, activation="relu"),
            Dropout(dropout),
            Dense(units // 2, activation="relu"),
            Dropout(dropout),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --- Tune ---
result = optimize(
    builder_fn=build_model,
    train_data=x_train,
    val_data=(x_val, y_val),
    config=TunerConfig(
        project_name="mnist_demo",
        output_dir="./runs",
        strategy="bayesian",
        max_trials=10,
        search_epochs=5,
        early_stop_patience=3,
        extra_fields={"dataset": "mnist"},
    ),
    search_kwargs={"y": y_train},
)

# --- Use ---
print(result.summary())
print(f"\nMaster CSV at: {result.master_csv_path}")
print(f"Best hyperparameters: {result.best_hps}")

# The result.model is compiled and ready for a production training run
result.model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
result.model.save("best_mnist_model.keras")
