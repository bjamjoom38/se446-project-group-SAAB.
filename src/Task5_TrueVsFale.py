from task5_arrest_utils import count_arrest_status

results, skipped_rows, arrest_index = count_arrest_status("Chicago_Crimes.csv")
print("Arrest column index:", arrest_index)

# Output final results
print("Arrest Status | Count")
print("----------------------")

for key in ("True", "False"):
    print(f"{key} | {results[key]}")
    print("=======================")

if skipped_rows:
    print(f"Skipped malformed rows: {skipped_rows}")
