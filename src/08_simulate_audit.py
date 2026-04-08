"""
08_simulate_audit.py
Run the verifier across a batch of test images, simulate RFID claims (with an injected fraud rate),
and produce a simple CSV audit report + summary statistics.

Run:
    python src/08_simulate_audit.py
"""

import os
import csv
import random
from tqdm import tqdm
# placeholder import removed (was breaking run-time execution)
    
# We'll import the verifier function dynamically to avoid import issues in some environments
from importlib import import_module

VERIFIER_MODULE = 'src.07_cross_modal_verifier'
VERIFIER_FUNC = 'verify'

TEST_ROOT = r"C:\Users\ujjwa\OneDrive\Desktop\tiet-ucs532p-bteam\dataset\cropped_test_data"
OUT_CSV = r"C:\Users\ujjwa\OneDrive\Desktop\tiet-ucs532p-bteam\outputs\metrics\audit_report.csv"
CLASS_FOLDERS = {'f1': 'Car', 'f4': 'Bus', 'f5': 'Truck'}
INJECT_FRAUD_RATE = 0.05  # fraction of samples where RFID claim is intentionally wrong


def gather_test_images(limit_per_class=None):
    items = []
    for folder, name in CLASS_FOLDERS.items():
        folder_path = os.path.join(TEST_ROOT, folder)
        if not os.path.exists(folder_path):
            continue
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if limit_per_class:
            files = files[:limit_per_class]
        for f in files:
            items.append((f, name))
    return items


def simulate_claim(true_label, inject_rate=INJECT_FRAUD_RATE):
    # With probability inject_rate, return a wrong claim (random other class)
    labels = list(CLASS_FOLDERS.values())
    if random.random() < inject_rate:
        choices = [l for l in labels if l != true_label]
        return random.choice(choices)
    return true_label


def main():
    mod = import_module(VERIFIER_MODULE)
    verify = getattr(mod, VERIFIER_FUNC)

    items = gather_test_images()
    if not items:
        print('No test images found')
        return

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['image', 'true', 'rfid_claim', 'predicted', 'fraud_score', 'match'])
        for path, true in tqdm(items, desc='Auditing'):
            claim = simulate_claim(true)
            try:
                res = verify(path, claim)
            except Exception as e:
                print('Verifier error for', path, e)
                continue
            writer.writerow([res['image'], true, claim, res['predicted'], res['fraud_score'], res['match']])

    print(f'Audit saved to {OUT_CSV}')


if __name__ == '__main__':
    main()
