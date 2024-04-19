import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns

# Load the data from a text file
file_path = 'final_6_hour_trace.csv'  # Update this to the path of your text file
data = pd.read_csv(file_path, header=0)  # Assuming the first line contains headers


times = []
for index, row in data.iterrows():
    times.append(row['start_timestamp'])

times.sort()
time_differences = []
min_time = max(times)
for i in range(1, len(times)):
    min_time = min(min_time, times[i] - times[i-1])
    time_differences.append(times[i] - times[i-1])
print(min_time)

# # Set the aesthetic style of the plots
# sns.set_style("whitegrid")

# # Create a line plot
# plt.figure(figsize=(20, 20))  # You can adjust the size to fit your needs
# lineplot = sns.lineplot(data=data, x='start_timestamp', y='app', hue='app', marker='o')

# # Setting the title
# plt.title('Line Graph of Apps over Time')

# # Rotate date labels if necessary
# plt.xticks(rotation=45)

# # Show the plot
# plt.savefig("line_plot")
