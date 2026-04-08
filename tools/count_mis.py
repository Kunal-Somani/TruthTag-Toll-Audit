import csv
from collections import Counter
p = r'C:\Users\ujjwa\OneDrive\Desktop\tiet-ucs532p-bteam\outputs\metrics\misclassified_list.csv'
with open(p, encoding='utf-8') as f:
    next(f)
    r = csv.reader(f)
    c = Counter()
    for row in r:
        c[(row[1], row[2])] += 1
for (t,p),v in c.items():
    print(f'{t} -> {p}: {v}')
print('TOTAL', sum(c.values()))
