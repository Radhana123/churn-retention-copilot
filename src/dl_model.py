"""
Neural network baseline for the churn model comparison.

model_trainer.py compares Logistic Regression, Random Forest and XGBoost. All
three either assume a linear decision boundary (logistic, on scaled features)
or build one from axis-aligned splits (the trees). A small feedforward network
is added here as a fourth point of comparison - not because it's expected to
win on ~7,000 rows of tabular data (gradient-boosted trees usually do), but
because *whether* it wins, and by how much, is itself the finding. If a
3-layer network with dropout can't beat XGBoost here, that's evidence the
churn signal is close to linearly/tree-separable already - worth stating
plainly rather than skipping the comparison because "tabular + small data
favours trees" is usually true anyway.

Run after model_trainer.py's preprocessing (uses the same train/test split
and the same StandardScaler-scaled features Logistic Regression trains on -
a network needs standardized inputs for the same reason logistic regression
does: gradient descent on unscaled columns converges unevenly across them).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.metrics import precision_score, recall_score, roc_auc_score

tf.random.set_seed(42)


def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_neural_net(X_train_scaled, X_test_scaled, y_train, y_test, class_weight=None):
    model = build_model(X_train_scaled.shape[1])

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=0,
    )

    probs = model.predict(X_test_scaled, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "precision": round(precision_score(y_test, preds), 3),
        "recall": round(recall_score(y_test, preds), 3),
        "roc_auc": round(roc_auc_score(y_test, probs), 3),
        "epochs_trained": len(history.history["loss"]),
    }
    return metrics, model


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from data_loader import load_data
    from model_trainer import preprocess_data, train_and_evaluate

    df = load_data()
    X, y, encoders = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Telco churn is imbalanced (~27% positive class). Without weighting, the
    # network can minimize loss by mostly predicting "no churn" and still
    # look decent on accuracy while missing most actual churners.
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    class_weight = {0: 1.0, 1: n_neg / n_pos}

    print("Training neural network...")
    nn_metrics, _ = train_neural_net(X_train_scaled, X_test_scaled, y_train, y_test, class_weight)

    print("\n=== Neural Network ===")
    print(nn_metrics)

    print("\n=== Classical models (for direct comparison) ===")
    results, _, _, _, _, _, _ = train_and_evaluate(X, y)
    for name, metrics in results.items():
        print(name + ": " + str(metrics))