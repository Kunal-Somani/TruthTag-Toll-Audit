"""
04_test_on_cropped_test.py
Run the trained model on dataset/cropped_test_data and report metrics.
Usage:
    python src/04_test_on_cropped_test.py
"""
import os
import cv2
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# CONFIG - must match training mapping/order used in 02_hog_extraction.py
DATASET_TEST_ROOT = r"C:\Users\ujjwa\OneDrive\Desktop\tiet-ucs532p-bteam\dataset\cropped_test_data"
CLASS_MAP = {
    "f1": "Car",
    "f4": "Bus",
    "f5": "Truck",
}
MODEL_PATH = r"C:\Users\ujjwa\OneDrive\Desktop\tiet-ucs532p-bteam\models\hog_svm_model.joblib"
SCALER_PATH = r"C:\Users\ujjwa\OneDrive\Desktop\tiet-ucs532p-bteam\models\scaler.joblib"
OUT_DIR = r"C:\Users\ujjwa\OneDrive\Desktop\tiet-ucs532p-bteam\outputs\metrics"
os.makedirs(OUT_DIR, exist_ok=True)
FEATURES_TEST_NPZ = r"C:\Users\ujjwa\OneDrive\Desktop\tiet-ucs532p-bteam\outputs\features\hog_features_test.npz"

# HOG params
win_size = (64, 128)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9
hog = cv2.HOGDescriptor(_winSize=win_size,
                        _blockSize=block_size,
                        _blockStride=block_stride,
                        _cellSize=cell_size,
                        _nbins=nbins)


def extract_hog(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if (img.shape[1], img.shape[0]) != win_size:
        img = cv2.resize(img, win_size, interpolation=cv2.INTER_AREA)
    d = hog.compute(img)
    if d is None:
        return None
    return d.flatten()


def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Model or scaler not found. Run src/03_train_svm.py first.")
        return

    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Prefer the precomputed test features for reproducible, fast evaluation.
    if os.path.exists(FEATURES_TEST_NPZ):
        print(f"\nLoading test features from {FEATURES_TEST_NPZ} and running batch prediction...")
        data = np.load(FEATURES_TEST_NPZ, allow_pickle=True)
        X_test, y_test = data['X'], data['y']
        # label names should match training order
        label_names = [str(n) for n in data['label_names']]
        Xs = scaler.transform(X_test)
        y_pred = clf.predict(Xs).astype(int)
        y_true = y_test.astype(int)
        # print counts
        for idx, name in enumerate(label_names):
            print(f"  {name:8s}: {(y_true == idx).sum()} samples")
    else:
        # Fallback: compute HOG per-image and predict (slower, but useful if .npz missing)
        y_true = []
        y_pred = []
        label_names = list(CLASS_MAP.values())
        label_to_idx = {name: idx for idx, name in enumerate(label_names)}

        print("\nScanning test folders and predicting (image-by-image)...")
        for folder, label in CLASS_MAP.items():
            folder_path = os.path.join(DATASET_TEST_ROOT, folder)
            if not os.path.exists(folder_path):
                print(f"  [MISSING] {folder_path}")
                continue
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]
            print(f"  {label} ({folder}): {len(files)} images")
            for fname in tqdm(files, desc=f"Predicting {label}"):
                path = os.path.join(folder_path, fname)
                desc = extract_hog(path)
                if desc is None:
                    continue
                Xs = scaler.transform(desc.reshape(1, -1))
                pred = clf.predict(Xs)[0]
                y_true.append(label_to_idx[label])
                y_pred.append(int(pred))

    if len(y_true) == 0:
        print("No test samples found or processed.")
        return

    print("\n--- Test Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=label_names))

    # Confusion matrix: raw counts and row-normalised percentages side-by-side
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    with np.errstate(all='ignore'):
        cm_perc = (cm.astype(float) / cm.sum(axis=1, keepdims=True))
    cm_perc = np.nan_to_num(cm_perc)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names, ax=axes[0])
    axes[0].set_title('Counts')
    axes[0].set_ylabel('True')
    axes[0].set_xlabel('Pred')

    sns.heatmap(cm_perc, annot=True, fmt='.2f', cmap='Blues', xticklabels=label_names, yticklabels=label_names, ax=axes[1])
    axes[1].set_title('Row-normalised (%)')
    axes[1].set_ylabel('True')
    axes[1].set_xlabel('Pred')

    out_path = os.path.join(OUT_DIR, 'test_confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved test confusion matrix: {out_path}")

if __name__ == '__main__':
    main()
