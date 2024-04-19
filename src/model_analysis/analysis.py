import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import data
df = pd.read_csv('../final_results_on_measurement_data_copy.csv')

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

# plot the "energy" and "duration" data for each function type
# each function type will have its own plot, and each plot will have a line for each input value (on the same plot)
# make the points evenly spaced on the x-axis

for func_type, df in function_dfs.items():

    plt.figure(figsize=(12, 5))

    # Plot for Energy
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    for input_value in sampled_inputs[func_type]:
        truncated_input = (input_value[:15] + '...') if len(input_value) > 15 else input_value
        subset = df[df['function_input'] == input_value]  # Adjust for actual matching
        plt.plot(subset.index, subset['energy'], label=f'Input {truncated_input}')  # Using DataFrame index
    plt.title(f'Energy Plot for {func_type}')
    plt.xlabel('Index (Row Number)')
    plt.ylabel('Energy')
    plt.legend()

    # Plot for Duration
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    for input_value in sampled_inputs[func_type]:
        truncated_input = (input_value[:15] + '...') if len(input_value) > 15 else input_value
        subset = df[df['function_input'] == input_value]  # Adjust for actual matching
        plt.plot(subset.index, subset['duration'], label=f'Input {truncated_input}')  # Using DataFrame index
    plt.axhline(y=function_slos[func_type], color='r', linestyle='--', label='SLO')  # SLO line
    plt.title(f'Duration Plot for {func_type}')
    plt.xlabel('Index (Row Number)')
    plt.ylabel('Duration')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.savefig(f'plots/{func_type}_plots.png')  # Save the figure to a file
