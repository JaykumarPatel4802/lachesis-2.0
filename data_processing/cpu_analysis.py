import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

class Function(Enum):
    FLOATMATMULT = 0
    IMAGEPROCESS = 1
    VIDEOPROCESS = 2
    ENCRYPT = 3
    LINPACK = 4


CONST_FREQ = 2400000
CONST_INPUTS = {
    Function.FLOATMATMULT: '["matrix1_2000_0.7.txt", "matrix1_2000_0.7.txt"]',
    Function.IMAGEPROCESS: '["1.8M-opera_2_199207.jpg"]',
    Function.VIDEOPROCESS: '["2.2M.mp4"]',
    Function.ENCRYPT: 100,
    Function.LINPACK: 100,
}
CONST_FILE = {
    Function.FLOATMATMULT: './filtered_data/filtered_floatmatmult.csv',
    Function.IMAGEPROCESS: './filtered_data/filtered_imageprocess.csv',
    Function.VIDEOPROCESS: './filtered_data/filtered_videoprocess.csv',
    Function.ENCRYPT: './filtered_data/filtered_encrypt.csv',
    Function.LINPACK: './filtered_data/filtered_linpack.csv',
}
    
"""
Each csv file contains data for function input, cpu count, frequency, and energy
This function will plot the energy over the number of cpu cores for each function
The frequency will be fixed at CONST_FREQ and the input will be fixed at CONST_INPUTS
"""
def analyze(function, input):
    # Read the CSV file
    df = pd.read_csv(CONST_FILE[function])
    # Filter the data based on the input
    df = df[df['inputs'] == input]
    # Filter the data based on the frequency
    df = df[df['frequency'] == CONST_FREQ]
    # Plot the energy over the number of cpu cores
    # plt.plot(df['cpu_limit'], df['energy'])
    # plt.xlabel('Number of CPU Cores')
    # plt.ylabel('Energy (Joules)')
    # plt.title('Energy vs Number of CPU Cores for ' + function.name)

    # # Save the plot
    # plt.savefig('./plots/cpu_analysis/cpu_analysis_' + function.name + '_' + input + '_' + str(CONST_FREQ) + '.png')    
    # plt.close()

    directory = None

    if function == Function.FLOATMATMULT:
        directory = 'floatmatmult'
    elif function == Function.IMAGEPROCESS:
        directory = 'imageprocess'
    elif function == Function.VIDEOPROCESS:
        directory = 'videoprocess'
    elif function == Function.ENCRYPT:
        directory = 'encrypt'
    elif function == Function.LINPACK:
        directory = 'linpack'

    # expanded
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot energy vs frequency
    axs[0, 0].plot(df['cpu_limit'], df['energy'])
    axs[0, 0].set_xlabel('Number of CPU Cores')
    axs[0, 0].set_ylabel('Energy (Joules)')
    axs[0, 0].set_title('Energy vs Number of CPU Cores')

    # Plot memory vs frequency
    axs[0, 1].plot(df['cpu_limit'], df['max_mem'], color='orange')
    axs[0, 1].set_xlabel('Number of CPU Cores')
    axs[0, 1].set_ylabel('Max Memory')
    axs[0, 1].set_title('Max Memory vs Number of CPU Cores')

    # Plot cpu vs frequency
    axs[1, 0].plot(df['cpu_limit'], df['max_cpu'], color='green')
    axs[1, 0].set_xlabel('Number of CPU Cores')
    axs[1, 0].set_ylabel('Max CPU Usage')
    axs[1, 0].set_title('Max CPU Usage vs Number of CPU Cores')

    # Plot duration vs frequency
    axs[1, 1].plot(df['cpu_limit'], df['duration'], color='red')
    axs[1, 1].set_xlabel('Number of CPU Cores')
    axs[1, 1].set_ylabel('Duration')
    axs[1, 1].set_title('Duration vs Number of CPU Cores')

    # Adjust layout
    plt.tight_layout()

    directory_path = './plots/cpu_analysis/expanded/' + directory + '/freq_2_4/'

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    plt.savefig(directory_path + function.name + '_' + input + '_' + str(CONST_FREQ) + '.png')      
    plt.close()

    # basic
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    # Plot energy vs frequency
    ax1.plot(df['cpu_limit'], df['energy'])
    ax1.set_xlabel('Number of CPU Cores')
    ax1.set_ylabel('Energy (Joules)')
    ax1.set_title('Energy vs Number of CPU Cores')

    # Plot duration vs frequency
    ax2.plot(df['cpu_limit'], df['duration'], color='red')
    ax2.set_xlabel('Number of CPU Cores')
    ax2.set_ylabel('Duration')
    ax2.set_title('Duration vs Number of CPU Cores')

    # Adjust layout
    plt.tight_layout()

    directory_path = './plots/cpu_analysis/basic/' + directory + '/freq_2_4/'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    plt.savefig(directory_path + function.name + '_' + input + '_' + str(CONST_FREQ) + '.png')      
    plt.close()
    return
        
# Entry point of the script
if __name__ == "__main__":
    for function in Function:
        # if function == Function.FLOATMATMULT or function == Function.IMAGEPROCESS or function == Function.VIDEOPROCESS or function == Function.ENCRYPT:
        if function == Function.LINPACK:
            df = pd.read_csv(CONST_FILE[function])
            unique_inputs = df['inputs'].unique()
            for input in unique_inputs:
                analyze(function, input)
