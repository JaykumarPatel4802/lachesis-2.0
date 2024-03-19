import pandas as pd
import os

def remove_outliers(csv_file):

    # Define a function to get the median value from a group
    def get_median(group):
        return group['energy'].median()

    # Load your CSV dataset into a DataFrame
    df = pd.read_csv(csv_file)

    # Assuming your dataset has columns 'core_count', 'frequency', 'input', and 'energy_reading'

    # Filter out rows where energy is -1
    df = df[df['energy'] != -1]

    # Group the DataFrame by 'core_count', 'frequency', and 'input'
    grouped = df.groupby(['cpu_limit', 'frequency', 'inputs'])

    # Apply the function to get the median energy reading for each group
    median_energy = grouped.apply(get_median)

    # Merge the median energy readings back to the original DataFrame
    df = df.merge(median_energy.rename('median_energy'), on=['cpu_limit', 'frequency', 'inputs'])

    # Filter the DataFrame to keep only the rows with the median energy readings
    filtered_df = df[df['energy'] == df['median_energy']]

    # Save the filtered DataFrame to a new CSV file
    filtered_file_name = './filtered_data/filtered_' + csv_file.split('/')[-1]
    filtered_df.to_csv(filtered_file_name, index=False)

def main():
    #go through all the .db files in sqlite directory and convert them to .csv files
    directory = './data/'
    # CSVs = ["floatmatmult", "imageprocess", "videoprocess"]
    CSVs = ["linpack"]
    for csv in CSVs:
        remove_outliers(directory + csv + '.csv')
    
    
    
if __name__ == '__main__':
    main()