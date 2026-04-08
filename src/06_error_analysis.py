"""
06_error_analysis.py
Find misclassified images on the test set and save a grid of misclassified examples.
This script re-extracts HOG from the test folders (so we can keep filenames/order),
uses the saved `models/hog_svm_model.joblib` + `models/scaler.joblib` and outputs:
  - `outputs/metrics/misclassified_grid.png`
  - `outputs/metrics/misclassified_list.csv` (filename,true,pred)

Run:
    python src/06_error_analysis.py
"""

import os
import csv
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Config (match other scripts)
TEST_ROOT = r"C:\Users\ujjwa\OneDrive\Desktop\tiet-ucs532p-bteam\dataset\cropped_test_data"
MODEL_PATH = r"C:\Users\ujjwa\OneDrive\Desktop\tiet-ucs532p-bteam\models\hog_svm_model.joblib"
SCALER_PATH = r"C:\Users\ujjwa\OneDrive\Desktop\tiet-ucs532p-bteam\models\scaler.joblib"
OUT_DIR = r"C:\Users\ujjwa\OneDrive\Desktop\tiet-ucs532p-bteam\outputs\metrics"
CLASS_MAP = {"f1": 0, "f4": 1, "f5": 2}
LABEL_NAMES = ["Car", "Bus", "Truck"]

WIN_SIZE = (64, 128)
BLOCK_SIZE = (16, 16)
BLOCK_STRIDE = (8, 8)
CELL_SIZE = (8, 8)
NBINS = 9


def build_hog():
    return cv2.HOGDescriptor(WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS)


def extract_test_features_and_files():
    hog = build_hog()
    X = []
    y = []
    files = []
    for folder, label_idx in CLASS_MAP.items():
        folder_path = os.path.join(TEST_ROOT, folder)
        if not os.path.exists(folder_path):
            continue
        imgs = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for fname in imgs:
            path = os.path.join(folder_path, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if (img.shape[1], img.shape[0]) != WIN_SIZE:
                img = cv2.resize(img, WIN_SIZE, interpolation=cv2.INTER_AREA)
            desc = hog.compute(img)
            if desc is None:
                continue
            X.append(desc.flatten())
            y.append(label_idx)
            files.append(path)
    if not X:
        return None, None, None
    return np.vstack(X), np.array(y, dtype=np.int32), files


def save_misclassified_grid(mis_list, max_images=36):
    if not mis_list:
        print('No misclassified images')
        return
    n = min(len(mis_list), max_images)
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axs = axs.flatten()
    for i in range(n):
        path, true, pred = mis_list[i]
        img = cv2.imread(path)
        if img is None:
            axs[i].axis('off')
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].imshow(img)
        axs[i].set_title(f"T:{LABEL_NAMES[true]} P:{LABEL_NAMES[pred]}", fontsize=8)
        axs[i].axis('off')
    for j in range(n, len(axs)):
        axs[j].axis('off')
    os.makedirs(OUT_DIR, exist_ok=True)
    out_img = os.path.join(OUT_DIR, 'misclassified_grid.png')
    plt.tight_layout()
    plt.savefig(out_img)
    print(f"Saved misclassified grid to {out_img}")


def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print('Model or scaler missing; run src/03_train_svm.py first')
        return
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X, y, files = extract_test_features_and_files()
    if X is None:
        print('No test features found')
        return

    Xs = scaler.transform(X)
    preds = clf.predict(Xs).astype(int)

    # Save CSV list of misclassifications
    mis = []
    for path, t, p in zip(files, y, preds):
        if int(t) != int(p):
            mis.append((path, int(t), int(p)))

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_out = os.path.join(OUT_DIR, 'misclassified_list.csv')
    with open(csv_out, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['filepath', 'true', 'pred'])
        for r in mis:
            writer.writerow([r[0], LABEL_NAMES[r[1]], LABEL_NAMES[r[2]]])
    print(f"Saved misclassified list to {csv_out} ({len(mis)} entries)")

    save_misclassified_grid(mis)

    # Print classification report for quick reference
    print('\nClassification report:')
    print(classification_report(y, preds, target_names=LABEL_NAMES))


if __name__ == '__main__':
    main()
