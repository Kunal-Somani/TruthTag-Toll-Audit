"""
07_cross_modal_verifier.py
Core verifier: given an image and an RFID-declared class, return a structured verdict and fraud score.
Simple heuristic confidence via LinearSVC decision_function + softmax.

Usage:
    python src/07_cross_modal_verifier.py <image_path> <rfid_claim>
"""

import os
import sys
import math
import cv2
import numpy as np
import joblib

MODEL_PATH = r"C:\Users\ujjwa\OneDrive\Desktop\tiet-ucs532p-bteam\models\hog_svm_model.joblib"
SCALER_PATH = r"C:\Users\ujjwa\OneDrive\Desktop\tiet-ucs532p-bteam\models\scaler.joblib"

# HOG params (must match training)
WIN_SIZE = (64, 128)
BLOCK_SIZE = (16, 16)
BLOCK_STRIDE = (8, 8)
CELL_SIZE = (8, 8)
NBINS = 9

CLASS_ORDER = ["Car", "Bus", "Truck"]


def build_hog():
    return cv2.HOGDescriptor(WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def verify(image_path, rfid_claim, threshold=0.5):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError('Model or scaler missing; run training first')
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError('Unable to read image')
    if (img.shape[1], img.shape[0]) != WIN_SIZE:
        img = cv2.resize(img, WIN_SIZE, interpolation=cv2.INTER_AREA)

    hog = build_hog()
    desc = hog.compute(img)
    if desc is None:
        raise ValueError('HOG failed')
    Xs = scaler.transform(desc.reshape(1, -1))

    # decision_function -> convert to soft probabilities
    try:
        scores = clf.decision_function(Xs)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        probs = softmax(scores[0])
    except Exception:
        # fallback: use predict only
        pred = int(clf.predict(Xs)[0])
        probs = np.zeros(len(CLASS_ORDER), dtype=float)
        probs[pred] = 1.0

    pred_idx = int(np.argmax(probs))
    pred_name = CLASS_ORDER[pred_idx]

    # Map RFID claim to index if possible
    try:
        claimed_idx = CLASS_ORDER.index(rfid_claim)
    except ValueError:
        claimed_idx = None

    match = (claimed_idx is not None) and (claimed_idx == pred_idx)
    fraud_score = 0.0
    if claimed_idx is None:
        fraud_score = 1.0  # unknown claim -> suspicious
    else:
        # higher score = more suspicious (1 - probability of claimed class)
        fraud_score = 1.0 - float(probs[claimed_idx])

    return {
        'image': image_path,
        'predicted': pred_name,
        'predicted_idx': pred_idx,
        'probabilities': probs.tolist(),
        'rfid_claim': rfid_claim,
        'match': bool(match),
        'fraud_score': float(fraud_score)
    }


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python src/07_cross_modal_verifier.py <image_path> <rfid_claim>')
        sys.exit(1)
    image_path = sys.argv[1]
    rfid_claim = sys.argv[2]
    out = verify(image_path, rfid_claim)
    import json
    print(json.dumps(out, indent=2))
