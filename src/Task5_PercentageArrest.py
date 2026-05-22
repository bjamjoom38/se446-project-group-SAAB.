from task5_arrest_utils import arrest_percentage, count_arrest_status


counts, skipped_rows, _ = count_arrest_status("Chicago_Crimes.csv")
total_crimes, arrests, percentage = arrest_percentage(counts)

# Print result
print(f"Total Crimes: {total_crimes}")
print(f"Crimes with Arrest: {arrests}")
print(f"Percentage of Crimes Resulting in Arrest: {percentage:.2f}%")

if skipped_rows:
    print(f"Skipped malformed rows: {skipped_rows}")
