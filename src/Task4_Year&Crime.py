from collections import Counter

years = []

with open('Chicago_Crimes.csv', 'r') as file:
    next(file)  
    
    for line in file:
        parts = line.split(',')   # split CSV line into columns
        
        date_str = parts[2]
        
        year = date_str.split('/')[2].split(' ')[0]
        
        years.append(year)

# Count crimes per year
crime_counts = Counter(years)

# Print
print("Year | Number of Crimes")
print("------------------------")

for year in sorted(crime_counts):
    print(f"{year} | {crime_counts[year]}")