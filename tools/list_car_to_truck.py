import csv, os
p = r'C:\Users\ujjwa\OneDrive\Desktop\tiet-ucs532p-bteam\outputs\metrics\misclassified_list.csv'
with open(p, encoding='utf-8') as f:
    next(f)
    r = csv.reader(f)
    files = [row[0] for row in r if row[1]=='Car' and row[2]=='Truck']
for fn in files:
    print(os.path.basename(fn))
print('TOTAL', len(files))
