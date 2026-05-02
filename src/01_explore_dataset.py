"""
01_explore_dataset.py
---------------------
STEP 1: Before we build anything, we look at what we have.
- Count images per class
- Check for corrupt/unreadable images
- Visualize samples from each class
- Check if all images are truly 64x128 or if there are surprises
- Look at brightness/contrast distribution across classes

Run from inside the repo:
    python src/01_explore_dataset.py
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless - no display needed
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────────────────────────
# Change this to wherever your dataset lives relative to where you run the script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset", "cropped_data")

CLASS_MAP = {
    "f1": "Car",
    "f4": "Bus",
    "f5": "Truck",
}

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "exploration")
# ─────────────────────────────────────────────────────────────────────────────


def count_and_validate(dataset_root, class_map):
    """
    Walk each class folder.
    Count total images, find corrupt ones, check sizes.
    """
    print("\n── DATASET SCAN ─────────────────────────────────────────────")

    summary = {}

    for folder, label in class_map.items():
        folder_path = os.path.join(dataset_root, folder)

        if not os.path.exists(folder_path):
            print(f"  [MISSING] {folder_path}")
            continue

        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        total       = len(files)
        corrupt     = 0
        sizes       = []
        brightnesses = []

        for fname in files:
            img = cv2.imread(os.path.join(folder_path, fname))
            if img is None:
                corrupt += 1
                continue
            h, w = img.shape[:2]
            sizes.append((w, h))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightnesses.append(gray.mean())

        unique_sizes = set(sizes)
        print(f"\n  {label} ({folder})/")
        print(f"    Total images   : {total}")
        print(f"    Corrupt/unread : {corrupt}")
        print(f"    Unique sizes   : {unique_sizes}")
        print(f"    Avg brightness : {np.mean(brightnesses):.1f}  "
              f"(std: {np.std(brightnesses):.1f})")

        summary[label] = {
            "folder": folder,
            "total": total,
            "corrupt": corrupt,
            "sizes": unique_sizes,
            "brightnesses": brightnesses,
        }

    return summary


def plot_class_distribution(summary, output_dir):
    """Bar chart of how many images per class."""
    labels = list(summary.keys())
    counts = [summary[l]["total"] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts, color=["#4C72B0", "#DD8452", "#55A868"],
                  edgecolor="black", linewidth=0.6)
    ax.set_title("Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Image Count")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                str(count), ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\n  [Saved] {path}")


def plot_brightness_distribution(summary, output_dir):
    """Overlapping brightness histograms per class."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"Car": "#4C72B0", "Bus": "#DD8452", "Truck": "#55A868"}

    for label, info in summary.items():
        ax.hist(info["brightnesses"], bins=50, alpha=0.55,
                label=label, color=colors.get(label, "gray"))

    ax.set_title("Brightness Distribution by Class", fontsize=13,
                 fontweight="bold")
    ax.set_xlabel("Mean Pixel Brightness (0–255)")
    ax.set_ylabel("Number of Images")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "brightness_distribution.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  [Saved] {path}")


def plot_sample_grid(dataset_root, class_map, output_dir, n_samples=5):
    """
    Show n_samples images per class in a grid.
    This is the most important thing — actually LOOK at the data.
    """
    n_classes = len(class_map)
    fig, axes = plt.subplots(n_classes, n_samples,
                             figsize=(n_samples * 2, n_classes * 2.5))
    fig.suptitle("Sample Images per Class (Top-down Toll Camera)",
                 fontsize=13, fontweight="bold", y=1.01)

    for row, (folder, label) in enumerate(class_map.items()):
        folder_path = os.path.join(dataset_root, folder)
        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Pick n evenly spaced samples so we get variety, not just the first n
        indices = np.linspace(0, len(files) - 1, n_samples, dtype=int)
        samples = [files[i] for i in indices]

        for col, fname in enumerate(samples):
            img = cv2.imread(os.path.join(folder_path, fname))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[row][col].imshow(img_rgb)
            axes[row][col].axis("off")
            if col == 0:
                axes[row][col].set_ylabel(label, fontsize=11,
                                          fontweight="bold", rotation=90,
                                          labelpad=40, va="center")

    plt.tight_layout()
    path = os.path.join(output_dir, "sample_grid.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Count, validate, check sizes
    summary = count_and_validate(DATASET_ROOT, CLASS_MAP)

    if not summary:
        print("\n[ERROR] No classes found. Check your DATASET_ROOT path.")
        return

    # 2. Class imbalance chart
    print("\n── GENERATING PLOTS ─────────────────────────────────────────")
    plot_class_distribution(summary, OUTPUT_DIR)

    # 3. Brightness distribution
    plot_brightness_distribution(summary, OUTPUT_DIR)

    # 4. Sample grid — actually look at the data
    plot_sample_grid(DATASET_ROOT, CLASS_MAP, OUTPUT_DIR, n_samples=5)

    # 5. Print the imbalance ratio so we know what we're dealing with
    print("\n── CLASS IMBALANCE ANALYSIS ─────────────────────────────────")
    totals = {l: summary[l]["total"] for l in summary}
    max_class = max(totals, key=totals.get)
    for label, count in totals.items():
        ratio = totals[max_class] / count
        print(f"  {label:10s}: {count:5d} images  "
              f"(ratio vs largest: 1 : {ratio:.1f})")

    print("\n── DONE ─────────────────────────────────────────────────────")
    print(f"  Check your outputs in: {OUTPUT_DIR}")
    print("  Next step: 02_hog_extraction.py\n")


if __name__ == "__main__":
    main()