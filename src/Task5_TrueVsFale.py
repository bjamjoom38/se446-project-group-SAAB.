import csv
from collections import defaultdict

results = defaultdict(int)

with open('Chicago_Crimes.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    header = next(reader)

    # Find the index of the "Arrest" which is index 6
    arrest_index = header.index('Arrest')
    print("Arrest column index:", arrest_index)

    # MAP PHASE: process each row and emit key-value pairs
    for row in reader:
        # Extract arrest status from the row
        arrest_status = row[arrest_index].strip().lower() 

        if arrest_status == 'true':
            results['True'] += 1   # counts arrests
        elif arrest_status == 'false':
            results['False'] += 1  # counts non-arrests

# Output final results
print("Arrest Status | Count")
print("----------------------")

for key in results:
    print(f"{key} | {results[key]}")
    print("=======================")