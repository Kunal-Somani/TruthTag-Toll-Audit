"""
05_visualize_hog.py
Render HOG visualizations for one example of each class side-by-side.
Falls back to saving grayscale images if `skimage` visualization isn't available.

Run:
    python src/05_visualize_hog.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths and HOG params (match other scripts)
TEST_ROOT = r"C:\Users\ujjwa\OneDrive\Desktop\tiet-ucs532p-bteam\dataset\cropped_test_data"
OUT_PATH = r"C:\Users\ujjwa\OneDrive\Desktop\tiet-ucs532p-bteam\outputs\metrics\hog_visualization.png"
CLASS_MAP = {"f1": "Car", "f4": "Bus", "f5": "Truck"}

WIN_SIZE = (64, 128)
BLOCK_SIZE = (16, 16)
BLOCK_STRIDE = (8, 8)
CELL_SIZE = (8, 8)
NBINS = 9


def build_hog():
    return cv2.HOGDescriptor(WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS)


def try_skimage_visualize(img):
    try:
        from skimage.feature import hog as sk_hog
        hog_vec, hog_img = sk_hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, feature_vector=True)
        return hog_img
    except Exception:
        return None


def main():
    hog = build_hog()
    examples = []

    for folder, name in CLASS_MAP.items():
        folder_path = os.path.join(TEST_ROOT, folder)
        if not os.path.exists(folder_path):
            print(f"Missing {folder_path}, skipping {name}")
            examples.append((name, None))
            continue
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            examples.append((name, None))
            continue
        path = os.path.join(folder_path, files[0])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            examples.append((name, None))
            continue
        if (img.shape[1], img.shape[0]) != WIN_SIZE:
            img = cv2.resize(img, WIN_SIZE, interpolation=cv2.INTER_AREA)

        # Try skimage-style HOG visualization first
        viz = try_skimage_visualize(img)
        if viz is None:
            # Fallback: use the image (rescaled) as a placeholder
            viz = img
        examples.append((name, viz))

    # Plot side-by-side
    cols = len(examples)
    fig, axs = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axs = [axs]
    for ax, (name, viz) in zip(axs, examples):
        if viz is None:
            ax.text(0.5, 0.5, 'No image', ha='center', va='center')
            ax.set_title(name)
            ax.axis('off')
            continue
        ax.imshow(viz, cmap='gray')
        ax.set_title(name)
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH)
    print(f"Saved HOG visualization to {OUT_PATH}")


if __name__ == '__main__':
    main()
