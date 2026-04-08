import os
import csv
import runpy

# Sample audit runner: uses gather_test_images(limit_per_class=...) to limit work
# load scripts as modules via runpy to avoid package import problems
amod = runpy.run_path('src/08_simulate_audit.py')
gather_test_images = amod.get('gather_test_images')
simulate_claim = amod.get('simulate_claim')

vmod = runpy.run_path('src/07_cross_modal_verifier.py')
verify = vmod.get('verify')

items = gather_test_images(limit_per_class=20)
OUT = os.path.join('outputs', 'metrics', 'audit_report_sample.csv')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(OUT, 'w', newline='', encoding='utf-8') as fh:
    writer = csv.writer(fh)
    writer.writerow(['image', 'true', 'rfid_claim', 'predicted', 'fraud_score', 'match'])
    count = 0
    for path, true in items:
        claim = simulate_claim(true)
        try:
            res = verify(path, claim)
        except Exception as e:
            print('Verifier error for', path, e)
            continue
        writer.writerow([res['image'], true, claim, res['predicted'], res['fraud_score'], res['match']])
        count += 1

print(f'Wrote {count} rows to {OUT}')
