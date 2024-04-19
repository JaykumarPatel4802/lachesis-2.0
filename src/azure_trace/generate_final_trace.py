import pandas as pd
import matplotlib.pyplot as plt
import random

# Load the data from a text file
file_path = 'AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt'  # Update this to the path of your text file
data = pd.read_csv(file_path, header=0)  # Assuming the first line contains headers

# Calculate the start timestamp
data['start_timestamp'] = data['end_timestamp'] - data['duration']

threshold = 21600
hour_6_data = data[data['start_timestamp'] < threshold]
hour_6_data.to_csv("hour_6_data.csv", index=False)

# Save to a new CSV file
# output_file_path = 'final_trace.csv'  # Define the output file name
# data.to_csv(output_file_path, index=False)

app_to_funcs = hour_6_data.groupby('app')['func'].agg(set).to_dict()

# Count occurrences of each app
app_counts = hour_6_data['app'].value_counts().to_dict()

app_func_qty_data = []

# Example of printing the dictionary
for app, funcs in app_to_funcs.items():
    app_func_qty_data.append((app, len(list(funcs)), app_counts[app]))
    # print(f"App: {app}, Funcs: {len(list(funcs))}, Count: {app_counts[app]}")

sorted_app_func_qty_data = sorted(app_func_qty_data, reverse=True, key=lambda x: x[2])
for i in sorted_app_func_qty_data:
    print(f"App: {i[0]}, Funcs: {i[1]}, Count: {i[2]}")


print("CSV file has been created with start timestamps.")

# # Count unique func values
func_counts = hour_6_data['func'].value_counts()
print(len(func_counts))

app_counts = hour_6_data['app'].value_counts()
print(len(app_counts))

"""
App: a594f92f84072b4cd031fe5283d1781a6e98f430696dec0a8e3b02eadb5fc0b8, Funcs: 1, Count: 12684
App: 06da275043bac5526d5c2252a4daa222bb062165977f111b693ed8d335917291, Funcs: 3, Count: 7481
App: 734272c01926d19690e5ec308bab64ef97950b75b1c7582283e0783fce1751d8, Funcs: 8, Count: 770
App: 7fa05b607ae861b85ec53cea12d3efaed8be0f9a92f5d6e8067244161d491e96, Funcs: 1, Count: 573
App: 85479ef37b5dc75dd5aeca3bab499129b97a134dac5d740d2c68941de9d63031, Funcs: 8, Count: 492
"""

filtered_hour_6_data = hour_6_data[ 
    (hour_6_data['app'] == "a594f92f84072b4cd031fe5283d1781a6e98f430696dec0a8e3b02eadb5fc0b8") | 
    (hour_6_data['app'] == "06da275043bac5526d5c2252a4daa222bb062165977f111b693ed8d335917291") | 
    (hour_6_data['app'] == "734272c01926d19690e5ec308bab64ef97950b75b1c7582283e0783fce1751d8") | 
    (hour_6_data['app'] == "7fa05b607ae861b85ec53cea12d3efaed8be0f9a92f5d6e8067244161d491e96") | 
    (hour_6_data['app'] == "85479ef37b5dc75dd5aeca3bab499129b97a134dac5d740d2c68941de9d63031")
]

filtered_hour_6_data.to_csv("filtered_hour_6_data.csv", index=False)

# Function to sample between 450 and 700 rows randomly
def random_sample(group):
    n = random.randint(450, 700)  # Random number between 450 and 700
    return group.sample(min(n, len(group)))  # Sample min(n, len(group)) rows to handle groups smaller than 450


sampled_6_hour_df = filtered_hour_6_data.groupby('app').apply(random_sample).reset_index(drop=True)

# Assuming 'start_timestamp' is the column you want to sort by
sorted_sampled_filtered_hour_6_data = sampled_6_hour_df.sort_values(by='start_timestamp')
sorted_sampled_filtered_hour_6_data = sorted_sampled_filtered_hour_6_data.sort_values(by='app')


def subsample_group(group, min_interval):
    # Ensure the group is sorted by timestamp; this is crucial for the diff() to work correctly.
    group = group.sort_values(by='start_timestamp')
    # Calculate the time difference between consecutive timestamps
    time_diffs = group['start_timestamp'].diff()
    # Keep the first row and then those where the difference with the previous timestamp is at least min_interval seconds
    return group[(time_diffs >= min_interval) | time_diffs.isna()]


X = 5  # Example: 30 seconds
sorted_sampled_filtered_hour_6_data = sorted_sampled_filtered_hour_6_data.groupby('app', group_keys=False).apply(subsample_group, min_interval=X)


sorted_sampled_filtered_hour_6_data.to_csv("final_6_hour_trace.csv", index=False)