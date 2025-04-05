import pandas as pd

# Load the CSV file
df = pd.read_csv("ecg_reports_output.csv")

# Filter for rows where 'filename_lr' starts with "records100/00000/" 
filtered_df = df[df['filename_lr'].str.startswith("records100/00000/")]

# Save the filtered data to a new CSV file
filtered_df.to_csv("ecg_reports_outpu100_filtered.csv", index=False)

print("Filtered rows saved to 'ecg_reports_outpu100_filtered.csv'")
