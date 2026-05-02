"""
03_train_svm.py
---------------
Train a LinearSVC on HOG features extracted from the training set.
Evaluate honestly on the SEPARATE test set — not a split of training data.

What changed from the previous version:
  - Loads hog_features.npz (train) and hog_features_test.npz (test) separately
  - No more train_test_split() on training data — that was giving fake-good val scores
  - Saves scaler alongside model — you need both at inference time
  - Prints per-class breakdown clearly so we can see Bus performance honestly
  - Saves confusion matrix with percentages (not just counts) — easier to read

Run from repo root:
    python src/03_train_svm.py

Requires:
    outputs/features/hog_features.npz       <- from 02_hog_extraction.py (train)
    outputs/features/hog_features_test.npz  <- from 02_hog_extraction.py (test)
"""

import os
import time
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)

# ── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_FEATURES = os.path.join(BASE_DIR, "outputs", "features", "hog_features.npz")
TEST_FEATURES  = os.path.join(BASE_DIR, "outputs", "features", "hog_features_test.npz")

MODEL_OUT_DIR  = os.path.join(BASE_DIR, "models")
METRICS_OUT    = os.path.join(BASE_DIR, "outputs", "metrics")

# Label index → name mapping must match what 02_hog_extraction.py assigned
# 02 iterates CLASS_MAP in order: f1=Car(0), f4=Bus(1), f5=Truck(2)
LABEL_NAMES = ["Car", "Bus", "Truck"]
# ─────────────────────────────────────────────────────────────────────────────


def load_features(path, split_name):
    print(f"\n  Loading {split_name} features from:\n    {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] {path} not found.\n"
            f"Run 02_hog_extraction.py with the {'test' if 'test' in path else 'train'} "
            f"dataset root first."
        )
    data = np.load(path, allow_pickle=True)
    X, y = data['X'], data['y']
    print(f"  Shape: X={X.shape}, y={y.shape}")

    # Per-class count so we know what we're working with
    for idx, name in enumerate(LABEL_NAMES):
        count = (y == idx).sum()
        print(f"    {name:8s}: {count} samples")

    return X, y


def plot_confusion_matrix(cm, label_names, title, out_path):
    """
    Plot confusion matrix twice side by side:
    left = raw counts, right = row-normalised percentages.
    Percentages show recall per class — much more informative than counts alone
    when classes are imbalanced.
    """
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_pct],
        ['d', '.1f'],
        ['Raw Counts', 'Row-Normalised (Recall %)']
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=label_names, yticklabels=label_names,
            linewidths=0.5, ax=ax,
            annot_kws={"size": 11}
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\n  [Saved] Confusion matrix → {out_path}")


def main():
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    os.makedirs(METRICS_OUT, exist_ok=True)

    # ── 1. LOAD DATA ─────────────────────────────────────────────────────────
    print("\n── LOADING DATA ─────────────────────────────────────────────────")
    X_train, y_train = load_features(TRAIN_FEATURES, "TRAIN")
    X_test,  y_test  = load_features(TEST_FEATURES,  "TEST")

    # ── 2. SCALE ─────────────────────────────────────────────────────────────
    # StandardScaler: zero-mean, unit-variance per feature.
    # Fit ONLY on train, apply same transform to test.
    # Critical: if you fit on test data too, you're leaking information.
    print("\n── SCALING ──────────────────────────────────────────────────────")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)       # transform only — no fit
    print("  StandardScaler fitted on train, applied to test. ✓")

    # ── 3. TRAIN ─────────────────────────────────────────────────────────────
    print("\n── TRAINING ─────────────────────────────────────────────────────")
    print("  Classifier : LinearSVC")
    print("  C          : 1.0  (regularisation — higher = tighter fit)")
    print("  class_weight: balanced  (compensates for 33:1 car:bus imbalance)")
    print("  max_iter   : 5000")
    print()

    clf = LinearSVC(
        C=1.0,
        class_weight='balanced',   # weights inversely proportional to class freq
        max_iter=5000,
        random_state=42,
        dual='auto'                # sklearn ≥1.5 prefers auto
    )

    t0 = time.time()
    clf.fit(X_train_sc, y_train)
    duration = time.time() - t0
    print(f"  Training complete in {duration:.1f}s ✓")

    # ── 4. EVALUATE ON REAL TEST SET ─────────────────────────────────────────
    print("\n── EVALUATION (real test set) ───────────────────────────────────")
    y_pred = clf.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Overall Accuracy : {acc*100:.2f}%")
    print()
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES,
                                digits=3))

    # ── 5. CONFUSION MATRIX ──────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, LABEL_NAMES,
        title="HOG + LinearSVC",
        out_path=os.path.join(METRICS_OUT, "confusion_matrix_v2.png")
    )

    # Print a plain-text version too so you can read it in terminal
    print("\n  Confusion matrix (rows=true, cols=predicted):")
    header = f"{'':10s}" + "".join(f"{n:>10s}" for n in LABEL_NAMES)
    print(f"  {header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>10d}" for v in row)
        print(f"  {LABEL_NAMES[i]:10s}{row_str}")

    # ── 6. SAVE MODEL + SCALER ───────────────────────────────────────────────
    print("\n── SAVING ───────────────────────────────────────────────────────")
    model_path  = os.path.join(MODEL_OUT_DIR, "hog_svm_model.joblib")
    scaler_path = os.path.join(MODEL_OUT_DIR, "scaler.joblib")
    labels_path = os.path.join(MODEL_OUT_DIR, "label_names.npy")

    joblib.dump(clf,    model_path)
    joblib.dump(scaler, scaler_path)
    np.save(labels_path, np.array(LABEL_NAMES))

    print(f"  Model  → {model_path}")
    print(f"  Scaler → {scaler_path}")
    print(f"  Labels → {labels_path}")

    print("\n── DONE ─────────────────────────────────────────────────────────")
    print("  Next: run 07_cross_modal_verifier.py to build the fraud layer.\n")


if __name__ == "__main__":
    main()