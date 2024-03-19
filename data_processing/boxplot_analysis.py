import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Import CSV file based on flag')

# Add the --floatmatmul flag
parser.add_argument('--floatmatmult', action='store_true', help='Import floatmatmult CSV file')
# Add the --videoprocess flag
parser.add_argument('--videoprocess', action='store_true', help='Import videoprocess CSV file')
# Add the --imageprocess flag
parser.add_argument('--imageprocess', action='store_true', help='Import imageprocess CSV file')
# Add the --linpack flag
parser.add_argument('--linpack', action='store_true', help='Import linpack CSV file')
# Add the --encrypt flag
parser.add_argument('--encrypt', action='store_true', help='Import encrypt CSV file')

# Parse the command-line arguments
args = parser.parse_args()

def get_function_input():
    if args.floatmatmult:
        return ""
    elif args.videoprocess:
        return ""
    elif args.imageprocess:
        return ""
    elif args.linpack:
        return ""
    elif args.encrypt:
        return ""
    else:
        parser.print_help()
    return None

# Function to import CSV file based on flag
def import_csv():
    if args.floatmatmult:
        return pd.read_csv('floatmatmult.csv')
    elif args.videoprocess:
        return pd.read_csv('videoprocess.csv')
    elif args.imageprocess:
        return pd.read_csv('imageprocess.csv')
    elif args.linpack:
        return pd.read_csv('linpack.csv')
    elif args.encrypt:
        return pd.read_csv('encrypt.csv')
    else:
        parser.print_help()

    return None

# Main function
def main():
    df = import_csv()

    if df:
        parse(df)

def parse(df):
    return 


def compile_summary_data(df):
    function_input_column = ""
    function_input_value = [get_function_input()]
    filtered_df = df[df[function_input_column] == function_input_value]
    # now we only have rows for a specific input to the function
    # save a subset of the columns
    columns_to_keep = ['energy', 'frequency', 'CPU']
    subset_df = filtered_df[columns_to_keep]
    file_path = ""
    subset_df.to_csv(file_path, index=False)
    
    
def plot_energy_box_plot():
    #go through all .csv files in ./data directory and create box plots for each function
    directory = './filtered_data/'
    #get all the .csv files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    function_energy = {}
    for csv_file in csv_files:
        
        function_name = csv_file.split('.')[0]
        df = pd.read_csv(directory + csv_file)
        print(df.columns)
        #get set of unique inputs from the 'input' column
        unique_inputs = df['inputs'].unique()
        #choose on of the inputs and get the subset of the dataframe
        # input_value = unique_inputs[0]
        input_value = None
        if function_name == 'filtered_linpack':
            input_value = unique_inputs[0]
        else:
            input_value = unique_inputs[len(unique_inputs) // 4]
        print(input_value)
        filtered_df = df[df['inputs'] == input_value]
        #get all the energy values
        energy_values = filtered_df['energy']
        #remove rows where energy is -1 or 0
        energy_values = energy_values[energy_values > 0]
        function_energy[function_name] = energy_values
    
    label_map = {
        'filtered_imageprocess': 'Image\nProcessing',
        'filtered_videoprocess': 'Video\nProcessing',
        'filtered_floatmatmult': 'Matrix\nMultiplication',
        'filtered_encrypt': 'Encryption',
        'filtered_linpack': 'Linpack'
    }
    labels = [label_map[key] for key in function_energy]
    max_energy = max([max(function_energy[key]) for key in function_energy])
    max_energy_rounded = np.ceil(max_energy / 25) * 25
    #create a box plot for each function and put them in the same figure 
    fig, ax = plt.subplots()
    ax.boxplot(function_energy.values())
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(np.arange(0, max_energy_rounded + 1, 25))
    ax.set_ylabel('Energy (Joules)', fontsize=14)
    ax.set_xlabel('Function', fontsize=14)
    plt.title('Energy Consumption for Each Function\n(Fixed Input and Variable CPU Count and Frequency)', fontsize=16)
    # plt.show()
    fig.set_figheight(7)
    fig.set_figwidth(9)

    plt.yticks(fontsize=12)

    # Save the plot to a file
    plt.savefig('plots/energy_boxplot.png')  # You can change the file format if you want, e.g., 'boxplot_functions.pdf'

    # Optionally, you can close the figure to free up memory
    plt.close(fig)
    
def plot_duration_box_plot():
    #go through all .csv files in ./data directory and create box plots for each function
    directory = './filtered_data/'
    #get all the .csv files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    function_duration = {}
    for csv_file in csv_files:
        
        function_name = csv_file.split('.')[0]
        df = pd.read_csv(directory + csv_file)
        print(df.columns)
        #get set of unique inputs from the 'input' column
        unique_inputs = df['inputs'].unique()
        #choose on of the inputs and get the subset of the dataframe
        # input_value = unique_inputs[0]
        input_value = None
        if function_name == 'filtered_linpack':
            input_value = unique_inputs[0]
        else:
            input_value = unique_inputs[len(unique_inputs) // 4]
        print(input_value)
        filtered_df = df[df['inputs'] == input_value]
        #get all the energy values
        duration_values = filtered_df['duration']
        #remove rows where energy is -1 or 0
        duration_values = duration_values[duration_values > 0]
        function_duration[function_name] = duration_values
    
    label_map = {
        'filtered_imageprocess': 'Image\nProcessing',
        'filtered_videoprocess': 'Video\nProcessing',
        'filtered_floatmatmult': 'Matrix\nMultiplication',
        'filtered_encrypt': 'Encryption',
        'filtered_linpack': 'Linpack'
    }
    labels = [label_map[key] for key in function_duration]
    max_duration = max([max(function_duration[key]) for key in function_duration])
    max_duration_rounded = np.ceil(max_duration / 25) * 25
    #create a box plot for each function and put them in the same figure 
    fig, ax = plt.subplots()
    ax.boxplot(function_duration.values())
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(np.arange(0, max_duration_rounded + 1, 1000))
    ax.set_ylabel('Duration (Milliseconds)', fontsize=14)
    ax.set_xlabel('Function', fontsize=14)
    plt.title('Duration for Each Function\n(Fixed Input and Variable CPU Count and Frequency)', fontsize=16)
    #change figsize
    fig.set_figheight(7)
    fig.set_figwidth(9)
    # plt.show()

    plt.yticks(fontsize=12)

    # Save the plot to a file
    plt.savefig('plots/duration_boxplot.png')  # You can change the file format if you want, e.g., 'boxplot_functions.pdf'

    # Optionally, you can close the figure to free up memory
    plt.close(fig)
    
        
# Entry point of the script
if __name__ == "__main__":
    # main()
    plot_energy_box_plot()
    plot_duration_box_plot()

"""
Story:
- Problem with og energat
    -creates a new process to track energy for each container that is spun up
    -this is not scalable, can lead to high overhead, resource contention
- Our solution
    -have one process that runs in background, continously loops through any active containers and tracks energy
    -more scaleable if there are hundreds of containers running at once
- Why the delta is ok
    -in order to alleviate the original issue of high overhead, we have a lower energy sampling frequency since we are tracking many containers in a consecutive manner
    -have to use interpolation techniques to estimate the overall energy consumption, leading to a small delta between the actual and estimated energy
    -However, the trends in energy consumption are still captured, and relative energy values are still accurate enough to identify patterns and more energy efficient configurations
    
- If we keep everything else constant, even input, how does change in cpu allocation impact energy consumption
- Summary graph, 

x axis function (pick one input per function)
y axis energy
box plot for each function
- will show how energy can vary from 
This is in isolation, so image many invocations running at the same time, noise can play a factor

STORY TELLING:
-first plot: box plot that shows how energy consumption varies for each function for a fixed input and variable cpu count and frequency
    -why this is important: even with the same input, the number of cpus allocated and the frequency at which they run can impact energy consumption for a given function

Can so energy for one funciton for one input for one cpu value for many frequencies

Walk through the graphs, what the X, Y axis are, etc.

"""