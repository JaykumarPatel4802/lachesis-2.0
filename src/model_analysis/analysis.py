import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

data = [(0.1, 4), (0.2, 2), (0.3, 3), (0.4, 5)]

filtered_data = {
    'floatmatmult': pd.read_csv('../../data_processing/filtered_data/filtered_floatmatmult.csv'),
    'linpack': pd.read_csv('../../data_processing/filtered_data/filtered_linpack.csv'),
    'imageprocess': pd.read_csv('../../data_processing/filtered_data/filtered_imageprocess.csv'),
    'videoprocess': pd.read_csv('../../data_processing/filtered_data/filtered_videoprocess.csv'),
    'encrypt': pd.read_csv('../../data_processing/filtered_data/filtered_encrypt.csv')
}

# for x, y in data:
for i in range(1):
    # print(f"Looking at: {x}")
    # import data
    # df = pd.read_csv(f'../final_results_on_measurement_data_{x}_run{y}_copy.csv')
    df = pd.read_csv('../final_results_on_measurement_data_matern_0.1.csv')
    # img_dir = x
    img_dir = 'matern_0.1'

    # filter df on column "function_type" 
    df_floatmatmult = df[df['function_type'] == 'floatmatmult']  
    df_linpack = df[df['function_type'] == 'linpack']
    df_imageprocess = df[df['function_type'] == 'imageprocess']
    df_videoprocess = df[df['function_type'] == 'videoprocess']
    df_encrypt = df[df['function_type'] == 'encrypt']

    function_dfs = {
        'floatmatmult': df_floatmatmult,
        'linpack': df_linpack,
        'imageprocess': df_imageprocess,
        'videoprocess': df_videoprocess,
        'encrypt': df_encrypt
    }

    function_slos = {
        'floatmatmult': 76132.2,
        'linpack': 45008.67293666026,
        'imageprocess': 5491.8,
        'videoprocess': 20592.6,
        'encrypt': 41125.799999999996
    }

    # get unique set of "function_input" column values for each df
    unique_inputs = {}
    for key in function_dfs:
        unique_inputs[key] = list(function_dfs[key]['function_input'].unique())

    


    sampled_inputs = {}
    for key in unique_inputs:
        sampled_inputs[key] = np.random.choice(unique_inputs[key], 5, replace=False)

    min_known_energy = {}
    # get min energy usage for function input
    for key in sampled_inputs:
        min_known_energy_per_type = {}
        for input_value in sampled_inputs[key]:
            formatted_input = input_value.replace('\'', '"')  # Replace " with "" for matching DataFrame format
            if key == "encrypt" or key == "linpack":
                formatted_list = json.loads(formatted_input)
                # Convert each element in the list to float and format it
                formatted_list = [f"{float(item):.1f}" for item in formatted_list]
                # Convert the list back to a string that looks like a list
                formatted_input = json.dumps(formatted_list)
            # get all rows with the "inputs" column equal to the input value
            df = filtered_data[key]
            subset = df[df['inputs'] == formatted_input]
            min_known_energy_per_type[input_value] = subset['energy'].min()
        min_known_energy[key] = min_known_energy_per_type
    # print(f'Min known energy: {min_known_energy}')

    # plot the "energy" and "duration" data for each function type
    # each function type will have its own plot, and each plot will have a line for each input value (on the same plot)
    # make the points evenly spaced on the x-axis

    for func_type, df in function_dfs.items():

        plt.figure(figsize=(24, 5))

        # Plot for Energy
        plt.subplot(1, 4, 1)  # 1 row, 2 columns, 1st subplot
        for input_value in sampled_inputs[func_type]:
            truncated_input = (input_value[:20] + '...') if len(input_value) > 20 else input_value
            subset = df[df['function_input'] == input_value]  # Adjust for actual matching
            plt.plot(subset.index, subset['energy'], label=f'Input {truncated_input}')  # Using DataFrame index
        plt.title(f'Energy Plot for {func_type}')
        plt.xlabel('Index (Row Number)')
        plt.ylabel('Energy (J)')
        plt.legend()

        # Plot for Duration
        plt.subplot(1, 4, 2)  # 1 row, 2 columns, 2nd subplot
        for input_value in sampled_inputs[func_type]:
            truncated_input = (input_value[:20] + '...') if len(input_value) > 20 else input_value
            subset = df[df['function_input'] == input_value]  # Adjust for actual matching
            plt.plot(subset.index, subset['duration'], label=f'Input {truncated_input}')  # Using DataFrame index
        plt.axhline(y=function_slos[func_type], color='r', linestyle='--', label='SLO')  # SLO line
        plt.title(f'Duration Plot for {func_type}')
        plt.xlabel('Index (Row Number)')
        plt.ylabel('Duration (ms)')
        plt.legend()

        # Plot for Predicted Energy v Actual Energy
        plt.subplot(1, 4, 3)  # 1 row, 2 columns, 2nd subplot
        for input_value in sampled_inputs[func_type]:
            truncated_input = (input_value[:20] + '...') if len(input_value) > 20 else input_value
            subset = df[df['function_input'] == input_value]  # Adjust for actual matching
            plt.plot(subset.index, subset['energy'] - subset['predicted_energy'], label=f'Input {truncated_input}')  # Using DataFrame index
        plt.title(f'Energy Difference Plot for {func_type}')
        plt.xlabel('Index (Row Number)')
        plt.ylabel('Energy Difference (J)')
        plt.legend()

        # Plot the actual energy - min known energy for each input value
        plt.subplot(1, 4, 4)
        for input_value in sampled_inputs[func_type]:
            truncated_input = (input_value[:20] + '...') if len(input_value) > 20 else input_value
            subset = df[df['function_input'] == input_value]
            known_energy = min_known_energy[func_type][input_value]
            plt.plot(subset.index, subset['energy'] - known_energy, label=f'Input {truncated_input}')
        plt.title(f'Energy Difference from Known Min Energy for {func_type}')
        plt.xlabel('Index (Row Number)')
        plt.ylabel('Energy Difference (J)')
        plt.legend()

        # if x == 0.2:
        # print the ratio of latency where it is less than the SLO
        for input_value in sampled_inputs[func_type]:
            subset = df[df['function_input'] == input_value]
            ratio = subset[subset['duration'] < function_slos[func_type]].shape[0] / subset.shape[0]
            print(f'Ratio of {func_type} with input {input_value} that is less than SLO: {ratio}')


        plt.tight_layout()
        plt.show()

        plt.savefig(f'plots/{img_dir}/{func_type}_plots.png')  # Save the figure to a file
