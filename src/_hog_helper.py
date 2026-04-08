"""
_hog_helper.py
Helper: compute HOG descriptors for a folder mapping.
Used by `03_train_svm.py` when evaluating a separate test folder.
"""
import os
import cv2
import numpy as np

# HOG params - must match 02_hog_extraction.py
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


def compute_hog_for_folder(root_folder, class_map):
    """Return X, y where class_map maps folder->label_index (e.g. {'f1':0})"""
    features = []
    labels = []
    for folder, label_idx in class_map.items():
        folder_path = os.path.join(root_folder, folder)
        if not os.path.exists(folder_path):
            continue
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for fname in files:
            path = os.path.join(folder_path, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if (img.shape[1], img.shape[0]) != win_size:
                img = cv2.resize(img, win_size, interpolation=cv2.INTER_AREA)
            desc = hog.compute(img)
            if desc is None:
                continue
            features.append(desc.flatten())
            labels.append(label_idx)
    if features:
        return np.vstack(features), np.array(labels, dtype=np.int32)
    else:
        return np.empty((0,)), np.empty((0,))
