#create a 3D plot that shows the relationship between CPU frequency, number of cores, and energy consumption for one of the
#inputs for every function

# import the necessary packages
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def plot_heatmap():
    #go through all .csv files in ./data directory and create box plots for each function
    directory = './filtered_data/'
    plot_directory = './plots/cpu_freq_heatmap/'
    #get all the .csv files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    function_energy = {}
    for csv_file in csv_files:
        
        function_name = csv_file.split('.')[0]
        df = pd.read_csv(directory + csv_file)
        print(df.columns)
        #get set of unique inputs from the 'input' column
        inputs = df['inputs'].unique()
        for input in inputs:
            #filter the dataframe based on the input
            df_input = df[df['inputs'] == input]
            #create a 3D plot that shows the relationship between CPU frequency, number of cores, and energy consumption for one of the
            #make one plot so that each cpu count is a different color
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            #get the unique cpu counts
            cpu_counts = df_input['cpu_limit'].unique()
            for cpu_count in cpu_counts:
                df_cpu = df_input[df_input['cpu_limit'] == cpu_count]
                ax.scatter(df_cpu['frequency'], df_cpu['cpu_limit'], df_cpu['energy'], label=cpu_count)
                
            ax.set_xlabel('CPU Frequency')
            ax.set_ylabel('CPU Cores')
            ax.set_zlabel('Energy (Joules)')
            ax.set_title('Energy vs CPU Frequency and Cores for ' + function_name + ' with input ' + input)
            ax.legend(title='CPU Cores')
            
            #save the plot
            #check if plot_directory/funtion_name exists, if not create it
            if not os.path.exists(plot_directory + function_name):
                os.makedirs(plot_directory + function_name)
            plt.savefig(plot_directory + function_name + '/' + input + '_cpu.png')
            
            #make another plot where each frequency is a different color
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            #get the unique cpu counts
            frequencies = df_input['frequency'].unique()
            
            for frequency in frequencies:
                df_freq = df_input[df_input['frequency'] == frequency]
                ax.scatter(df_freq['cpu_limit'], df_freq['frequency'], df_freq['energy'], label=frequency)
            ax.set_xlabel('CPU Cores')
            ax.set_ylabel('CPU Frequency')
            ax.set_zlabel('Energy (Joules)')
            ax.set_title('Energy vs CPU Cores and Frequency for ' + function_name + ' with input ' + input)
            ax.legend(title='CPU Frequency')
            
            #save the plot
            plt.savefig(plot_directory + function_name + '/' + input + '_freq.png')
            plt.close()
            
            
            
def plot_contour_map():
    #go through all .csv files in ./data directory and create box plots for each function
    directory = './filtered_data/'
    plot_directory = './plots/cpu_freq_contour/'
    #get all the .csv files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    function_energy = {}
    for csv_file in csv_files:
        
        function_name = csv_file.split('.')[0]
        df = pd.read_csv(directory + csv_file)
        print(df.columns)
        #get set of unique inputs from the 'input' column
        inputs = df['inputs'].unique()
        for input in inputs:
            #filter the dataframe based on the input
            df_input = df[df['inputs'] == input]
            #create a 3D plot that shows the relationship between CPU frequency, number of cores, and energy consumption for one of the
            fig = plt.figure()
            
            #get the unique cpu counts
            X = df_input['frequency']
            Y = df_input['cpu_limit']
            Z = df_input['energy']
            #negate the energy values so that the contour map is inverted
            Z = -Z
            
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(X, Y, Z, cmap='viridis_r')
            # ax.plot_trisurf(X, Y, Z, cmap='autumn_r')
            # ax.plot_trisurf(X, Y, Z, cmap=plt.cm.coolwarm)
            # ax.plot_trisurf(X, Y, Z, cmap=plt.cm.Greys)
            

            
            
            ax.set_xlabel('CPU Frequency')
            ax.set_ylabel('CPU Cores')
            ax.set_zlabel('Energy (Joules)')
            ax.set_title('Energy vs CPU Frequency and Cores for ' + function_name + ' with input ' + input)
            
            #save the plot
            #check if plot_directory/funtion_name exists, if not create it
            if not os.path.exists(plot_directory + function_name):
                os.makedirs(plot_directory + function_name)
            plt.savefig(plot_directory + function_name + '/' + input + '.png')
            
            plt.close()
            
            
            
            
            
           
            
            
            
if __name__ == '__main__':
    # plot_heatmap()
    plot_contour_map()
