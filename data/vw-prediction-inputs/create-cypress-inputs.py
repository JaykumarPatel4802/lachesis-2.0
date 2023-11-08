import os
import pandas as pd

# 1. Get a list of all CSV files in the current directory
csv_files = [f for f in os.listdir() if f.endswith(".csv")]

for csv_file in csv_files:
    # 2. Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # 3. Find the maximum duration value in the duration column
    max_duration = df['duration'].max()

    # 4. Create a new column 'SLO' with 1.2 * max duration value
    df['SLO'] = 1.2 * max_duration

    # 5. Create a 'batch_size' column with SLO/duration values
    df['batch_size'] = df['SLO'] / df['duration']

    # Prepend 'cypress-' to the existing file name
    output_file = 'cypress-' + csv_file

    # Save the DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

    print(f"Processed {csv_file} and saved as {output_file}")