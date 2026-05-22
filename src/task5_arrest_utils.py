import csv
from collections import Counter


def count_arrest_status(csv_path="Chicago_Crimes.csv"):
    counts = Counter()
    skipped_rows = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        arrest_index = header.index("Arrest")

        for row in reader:
            if len(row) <= arrest_index:
                skipped_rows += 1
                continue

            arrest_status = row[arrest_index].strip().lower()
            if arrest_status == "true":
                counts["True"] += 1
            elif arrest_status == "false":
                counts["False"] += 1
            else:
                skipped_rows += 1

    return counts, skipped_rows, arrest_index


def arrest_percentage(counts):
    total_crimes = counts["True"] + counts["False"]
    if total_crimes == 0:
        return 0, 0, 0.0

    arrests = counts["True"]
    percentage = (arrests / total_crimes) * 100
    return total_crimes, arrests, percentage
