import pandas as pd

# ================== SETTINGS ==================
input_file = "data/data2008-2024.csv"   
output_file = "data/sampled_data.csv"
n = 10                         # Number of rows per year
# =============================================

# Load the data
df = pd.read_csv(input_file, on_bad_lines='skip')

print(f"Original rows: {len(df)}")

# Make sure we have a clean 'year' column
df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year

# Remove any rows where year could not be parsed
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

# Sample n rows per year
sampled = []

for year in sorted(df['year'].unique()):
    group = df[df['year'] == year]
    
    if len(group) <= n:
        print(f"Year {year}: only {len(group)} rows → taking all")
        sampled.append(group)
    else:
        print(f"Year {year}: sampling {n} rows out of {len(group)}")
        sampled.append(group.sample(n=n, random_state=42))

# Combine all samples
result = pd.concat(sampled, ignore_index=True)

# Sort by year
result = result.sort_values(by=['year']).reset_index(drop=True)

print(f"\nFinal sampled rows: {len(result)}")
print("\nRows per year:")
print(result['year'].value_counts().sort_index())

# Save to new file
result.to_csv(output_file, index=False)
print(f"\nSaved to: {output_file}")