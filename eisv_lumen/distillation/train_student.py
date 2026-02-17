"""Train student classifiers from teacher distillation data.

Trains three RandomForest classifiers that replicate the teacher's
expression generation:
1. Pattern classifier: shape + EISV features → pattern (5 classes)
2. Token-1 classifier: shape + EISV features → primary token (15 classes)
3. Token-2 classifier: shape + EISV features + token_1 → secondary token (16 classes)

Usage:
    python -m eisv_lumen.distillation.train_student \
        --data data/distillation/teacher_outputs.json \
        --output outputs/student
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from eisv_lumen.shapes.shape_classes import TrajectoryShape


# All possible output values
ALL_TOKENS = [
    "~warmth~", "~curiosity~", "~resonance~", "~stillness~", "~boundary~",
    "~reaching~", "~reflection~", "~ripple~", "~deep_listening~", "~emergence~",
    "~questioning~", "~holding~", "~releasing~", "~threshold~", "~return~",
]

ALL_PATTERNS = ["SINGLE", "PAIR", "TRIPLE", "REPETITION", "QUESTION"]

SHAPE_NAMES = sorted([s.value for s in TrajectoryShape])

# Numeric feature columns in order
NUMERIC_FEATURES = [
    "mean_E", "mean_I", "mean_S", "mean_V",
    "dE", "dI", "dS", "dV",
    "d2E", "d2I", "d2S", "d2V",
]


@dataclass
class StudentModels:
    """Container for trained student classifiers."""

    pattern_clf: RandomForestClassifier
    token1_clf: RandomForestClassifier
    token2_clf: RandomForestClassifier
    scaler: StandardScaler
    shape_encoder: LabelEncoder
    token1_encoder: LabelEncoder
    token2_encoder: LabelEncoder
    pattern_encoder: LabelEncoder


def load_distillation_data(path: str) -> List[Dict[str, Any]]:
    """Load teacher outputs JSON."""
    with open(path) as f:
        return json.load(f)


def prepare_features(
    data: List[Dict[str, Any]],
    scaler: Optional[StandardScaler] = None,
    shape_encoder: Optional[LabelEncoder] = None,
    fit: bool = True,
) -> Tuple[np.ndarray, StandardScaler, LabelEncoder]:
    """Convert raw data dicts to feature matrix.

    Returns (X, scaler, shape_encoder) where X has shape (n, 21):
    - 12 numeric features (scaled)
    - 9 one-hot shape features
    """
    # Numeric features
    numeric = np.array([
        [d[f] for f in NUMERIC_FEATURES] for d in data
    ], dtype=np.float64)

    # Shape encoding (one-hot)
    shapes = [d["shape"] for d in data]

    if shape_encoder is None:
        shape_encoder = LabelEncoder()
    if fit:
        shape_encoder.fit(SHAPE_NAMES)

    shape_indices = shape_encoder.transform(shapes)
    n_shapes = len(SHAPE_NAMES)
    shape_onehot = np.zeros((len(data), n_shapes))
    for i, idx in enumerate(shape_indices):
        shape_onehot[i, idx] = 1.0

    # Scale numeric features
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        numeric_scaled = scaler.fit_transform(numeric)
    else:
        numeric_scaled = scaler.transform(numeric)

    X = np.hstack([numeric_scaled, shape_onehot])
    return X, scaler, shape_encoder


def train_student_models(
    data: List[Dict[str, Any]],
    n_estimators: int = 200,
    max_depth: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[StudentModels, Dict[str, Any]]:
    """Train the three student classifiers.

    Returns (StudentModels, metrics_dict).
    """
    # Prepare features
    X, scaler, shape_encoder = prepare_features(data)

    # Prepare targets
    patterns = [d["pattern"] for d in data]
    token1s = [d["token_1"] for d in data]
    token2s = [d["token_2"] for d in data]

    pattern_encoder = LabelEncoder()
    pattern_encoder.fit(ALL_PATTERNS)
    y_pattern = pattern_encoder.transform(patterns)

    token1_encoder = LabelEncoder()
    token1_encoder.fit(ALL_TOKENS)
    y_token1 = token1_encoder.transform(token1s)

    token2_classes = ALL_TOKENS + ["none"]
    token2_encoder = LabelEncoder()
    token2_encoder.fit(token2_classes)
    y_token2 = token2_encoder.transform(token2s)

    # For token-2, add token_1 as extra feature
    token1_indices = y_token1.reshape(-1, 1).astype(np.float64)
    X_token2 = np.hstack([X, token1_indices])

    # Train/test split
    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=0.2, random_state=seed, stratify=y_pattern,
    )

    X_train, X_test = X[train_idx], X[test_idx]
    X_t2_train, X_t2_test = X_token2[train_idx], X_token2[test_idx]

    metrics: Dict[str, Any] = {"n_train": len(train_idx), "n_test": len(test_idx)}

    # 1. Pattern classifier
    if verbose:
        print("Training pattern classifier ...")
    pattern_clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed, n_jobs=-1, class_weight="balanced",
    )
    pattern_clf.fit(X_train, y_pattern[train_idx])
    pattern_acc = pattern_clf.score(X_test, y_pattern[test_idx])
    metrics["pattern_accuracy"] = float(pattern_acc)
    if verbose:
        print(f"  Pattern accuracy: {pattern_acc:.3f}")

    # 2. Token-1 classifier
    if verbose:
        print("Training token-1 classifier ...")
    token1_clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed, n_jobs=-1, class_weight="balanced",
    )
    token1_clf.fit(X_train, y_token1[train_idx])
    token1_acc = token1_clf.score(X_test, y_token1[test_idx])
    metrics["token1_accuracy"] = float(token1_acc)
    if verbose:
        print(f"  Token-1 accuracy: {token1_acc:.3f}")

    # 3. Token-2 classifier
    if verbose:
        print("Training token-2 classifier ...")
    token2_clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed, n_jobs=-1, class_weight="balanced",
    )
    token2_clf.fit(X_t2_train, y_token2[train_idx])
    token2_acc = token2_clf.score(X_t2_test, y_token2[test_idx])
    metrics["token2_accuracy"] = float(token2_acc)
    if verbose:
        print(f"  Token-2 accuracy: {token2_acc:.3f}")

    # Cross-validation for robustness estimate
    if verbose:
        print("\nCross-validation (5-fold) ...")
    cv_pattern = cross_val_score(pattern_clf, X, y_pattern, cv=5, n_jobs=-1)
    cv_token1 = cross_val_score(token1_clf, X, y_token1, cv=5, n_jobs=-1)
    metrics["cv_pattern_mean"] = float(cv_pattern.mean())
    metrics["cv_pattern_std"] = float(cv_pattern.std())
    metrics["cv_token1_mean"] = float(cv_token1.mean())
    metrics["cv_token1_std"] = float(cv_token1.std())

    if verbose:
        print(f"  Pattern CV: {cv_pattern.mean():.3f} ± {cv_pattern.std():.3f}")
        print(f"  Token-1 CV: {cv_token1.mean():.3f} ± {cv_token1.std():.3f}")

    models = StudentModels(
        pattern_clf=pattern_clf,
        token1_clf=token1_clf,
        token2_clf=token2_clf,
        scaler=scaler,
        shape_encoder=shape_encoder,
        token1_encoder=token1_encoder,
        token2_encoder=token2_encoder,
        pattern_encoder=pattern_encoder,
    )

    return models, metrics


def save_student_models(models: StudentModels, output_dir: str) -> None:
    """Save trained models to disk."""
    os.makedirs(output_dir, exist_ok=True)

    pickle.dump(models.pattern_clf, open(os.path.join(output_dir, "pattern_clf.pkl"), "wb"))
    pickle.dump(models.token1_clf, open(os.path.join(output_dir, "token1_clf.pkl"), "wb"))
    pickle.dump(models.token2_clf, open(os.path.join(output_dir, "token2_clf.pkl"), "wb"))
    pickle.dump(models.scaler, open(os.path.join(output_dir, "scaler.pkl"), "wb"))
    pickle.dump(models.shape_encoder, open(os.path.join(output_dir, "shape_encoder.pkl"), "wb"))
    pickle.dump(models.token1_encoder, open(os.path.join(output_dir, "token1_encoder.pkl"), "wb"))
    pickle.dump(models.token2_encoder, open(os.path.join(output_dir, "token2_encoder.pkl"), "wb"))
    pickle.dump(models.pattern_encoder, open(os.path.join(output_dir, "pattern_encoder.pkl"), "wb"))

    # Report sizes
    total = 0
    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        size = os.path.getsize(fpath)
        total += size
        print(f"  {fname}: {size / 1024:.1f} KB")
    print(f"  Total: {total / 1024:.1f} KB")


def load_student_models(model_dir: str) -> StudentModels:
    """Load trained models from disk."""
    def _load(name):
        with open(os.path.join(model_dir, name), "rb") as f:
            return pickle.load(f)

    return StudentModels(
        pattern_clf=_load("pattern_clf.pkl"),
        token1_clf=_load("token1_clf.pkl"),
        token2_clf=_load("token2_clf.pkl"),
        scaler=_load("scaler.pkl"),
        shape_encoder=_load("shape_encoder.pkl"),
        token1_encoder=_load("token1_encoder.pkl"),
        token2_encoder=_load("token2_encoder.pkl"),
        pattern_encoder=_load("pattern_encoder.pkl"),
    )


def predict(
    models: StudentModels,
    shape: str,
    features: Dict[str, float],
) -> Dict[str, Any]:
    """Run student inference on a single input.

    Parameters
    ----------
    models:
        Loaded StudentModels.
    shape:
        Trajectory shape name.
    features:
        Dict with keys: mean_E, mean_I, mean_S, mean_V, dE, dI, dS, dV,
        d2E, d2I, d2S, d2V.

    Returns
    -------
    Dict with keys: pattern, token_1, token_2, token_3, eisv_tokens.
    """
    # Build feature vector
    data_row = {"shape": shape}
    data_row.update(features)

    X, _, _ = prepare_features([data_row], models.scaler, models.shape_encoder, fit=False)

    # Predict pattern
    pattern_idx = models.pattern_clf.predict(X)[0]
    pattern = models.pattern_encoder.inverse_transform([pattern_idx])[0]

    # Predict token-1
    token1_idx = models.token1_clf.predict(X)[0]
    token_1 = models.token1_encoder.inverse_transform([token1_idx])[0]

    # Predict token-2 (add token_1 as feature)
    X_t2 = np.hstack([X, np.array([[token1_idx]], dtype=np.float64)])
    token2_idx = models.token2_clf.predict(X_t2)[0]
    token_2 = models.token2_encoder.inverse_transform([token2_idx])[0]

    # Build token list based on pattern
    if pattern == "SINGLE":
        eisv_tokens = [token_1]
    elif pattern == "REPETITION":
        eisv_tokens = [token_1, token_1]
    elif pattern == "PAIR" or pattern == "QUESTION":
        eisv_tokens = [token_1, token_2] if token_2 != "none" else [token_1]
    elif pattern == "TRIPLE":
        # Token-3: use random weighted choice (too sparse for classifier)
        eisv_tokens = [token_1, token_2] if token_2 != "none" else [token_1]
        # Token-3 would be added by caller if needed
    else:
        eisv_tokens = [token_1]

    return {
        "pattern": pattern,
        "token_1": token_1,
        "token_2": token_2,
        "eisv_tokens": eisv_tokens,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train student classifiers")
    parser.add_argument(
        "--data", required=True,
        help="Path to teacher_outputs.json",
    )
    parser.add_argument(
        "--output", default="outputs/student",
        help="Output directory for models",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data
    print(f"Loading distillation data from {args.data} ...")
    data = load_distillation_data(args.data)
    print(f"  {len(data)} examples loaded")

    # Train
    models, metrics = train_student_models(
        data,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        seed=args.seed,
    )

    # Save
    print(f"\nSaving models to {args.output} ...")
    save_student_models(models, args.output)

    # Save metrics
    metrics_path = os.path.join(args.output, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print("\n" + "=" * 60)
    print("Student Training Results")
    print("=" * 60)
    print(f"  Pattern accuracy:  {metrics['pattern_accuracy']:.3f}")
    print(f"  Token-1 accuracy:  {metrics['token1_accuracy']:.3f}")
    print(f"  Token-2 accuracy:  {metrics['token2_accuracy']:.3f}")
    print(f"  Pattern CV:        {metrics['cv_pattern_mean']:.3f} ± {metrics['cv_pattern_std']:.3f}")
    print(f"  Token-1 CV:        {metrics['cv_token1_mean']:.3f} ± {metrics['cv_token1_std']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
