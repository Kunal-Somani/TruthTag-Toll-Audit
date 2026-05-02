"""
02_hog_extraction.py
--------------------
Extract HOG descriptors from BOTH train and test dataset folders.
Saves two separate .npz files so 03_train_svm.py has clean inputs.

What changed from previous version:
  - Runs on both TRAIN and TEST roots in one go
  - Saves hog_features.npz (train) AND hog_features_test.npz (test)
  - Prints feature vector length once so you know what you're working with
  - Shows per-class count in each split so you catch imbalance early

Run from repo root:
    python src/02_hog_extraction.py
"""

import os
import cv2
import numpy as np

# ── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_ROOT = os.path.join(BASE_DIR, "dataset", "cropped_data")
TEST_ROOT  = os.path.join(BASE_DIR, "dataset", "cropped_test_data")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "features")

CLASS_MAP = {
    "f1": 0,   # Car
    "f4": 1,   # Bus
    "f5": 2,   # Truck
}
LABEL_NAMES = ["Car", "Bus", "Truck"]

# HOG params — must stay identical between extraction and inference
# 64x128 is the industry standard for vehicle HOG
WIN_SIZE     = (64, 128)
BLOCK_SIZE   = (16, 16)
BLOCK_STRIDE = (8, 8)
CELL_SIZE    = (8, 8)
NBINS        = 9
# ─────────────────────────────────────────────────────────────────────────────


def build_hog():
    return cv2.HOGDescriptor(
        WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS
    )


def extract_features_from_folder(dataset_root, class_map, hog):
    """
    Walk each class subfolder, compute HOG for every image.
    Returns X (n_samples, n_features) and y (n_samples,) as numpy arrays.
    """
    X, y = [], []

    for folder, label_idx in class_map.items():
        folder_path = os.path.join(dataset_root, folder)

        if not os.path.exists(folder_path):
            print(f"  [MISSING] {folder_path} — skipping")
            continue

        files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        label_name = LABEL_NAMES[label_idx]
        print(f"  {label_name:8s} ({folder}) — {len(files)} images", flush=True)

        skipped = 0
        for i, fname in enumerate(files, 1):
            path = os.path.join(folder_path, fname)
            img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                skipped += 1
                continue

            # Resize to HOG window size if not already correct
            if (img.shape[1], img.shape[0]) != WIN_SIZE:
                img = cv2.resize(img, WIN_SIZE, interpolation=cv2.INTER_AREA)

            desc = hog.compute(img)
            if desc is None:
                skipped += 1
                continue

            X.append(desc.flatten())
            y.append(label_idx)

            if i % 1000 == 0:
                print(f"    {i}/{len(files)} processed...", flush=True)

        if skipped:
            print(f"    [!] {skipped} images skipped (unreadable)")

    return np.vstack(X), np.array(y, dtype=np.int32)


def save_features(X, y, out_path):
    np.savez_compressed(out_path, X=X, y=y,
                        label_names=np.array(LABEL_NAMES))
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n  Saved → {out_path}  ({size_mb:.1f} MB)")
    print(f"  X shape : {X.shape}  (samples × features)")
    print(f"  y shape : {y.shape}")
    for idx, name in enumerate(LABEL_NAMES):
        print(f"    {name:8s}: {(y == idx).sum()} samples")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    hog = build_hog()

    # Print HOG descriptor length once so there are no surprises later
    dummy = np.zeros((WIN_SIZE[1], WIN_SIZE[0]), dtype=np.uint8)
    feat_len = hog.compute(dummy).flatten().shape[0]
    print(f"\nHOG descriptor length: {feat_len} features per image")
    print(f"Window: {WIN_SIZE} | Block: {BLOCK_SIZE} | "
          f"Stride: {BLOCK_STRIDE} | Cell: {CELL_SIZE} | Bins: {NBINS}\n")

    # ── TRAIN ────────────────────────────────────────────────────────────────
    print("── EXTRACTING TRAIN SET ─────────────────────────────────────────")
    X_train, y_train = extract_features_from_folder(TRAIN_ROOT, CLASS_MAP, hog)
    save_features(X_train, y_train,
                  os.path.join(OUTPUT_DIR, "hog_features.npz"))

    # ── TEST ─────────────────────────────────────────────────────────────────
    print("\n── EXTRACTING TEST SET ──────────────────────────────────────────")
    X_test, y_test = extract_features_from_folder(TEST_ROOT, CLASS_MAP, hog)
    save_features(X_test, y_test,
                  os.path.join(OUTPUT_DIR, "hog_features_test.npz"))

    print("\n── DONE ─────────────────────────────────────────────────────────")
    print("  Next: run 03_train_svm.py\n")


if __name__ == "__main__":
    main()