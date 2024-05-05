import pandas as pd
import seaborn as sns
import sqlite3
import matplotlib.pyplot as plt
import argparse
import numpy as np

def create_cpu_util_timeline():

    conn = sqlite3.connect('invoker_data.db')
    df_invoker = pd.read_sql_query('SELECT * FROM function_utilization_advanced;', conn)

    df_list = []
    for container in df_invoker['container_id'].unique():
        df_sub = df_invoker[df_invoker['container_id'] == container].copy()
        cpu_usages = np.array(df_sub['cpu_usage_ns'])
        num_cores = np.array(df_sub['num_cores'])
        curr_system_usages = np.array(df_sub['curr_system_usage'])

        # Compute CPU utilization in same form as docker stats
        cpu_utilization_list = np.zeros(len(cpu_usages), dtype=float)
        prev_cpu_usages = np.roll(cpu_usages, 1)
        prev_cpu_usages[0] = 0
        prev_system_usages = np.roll(curr_system_usages, 1)
        prev_system_usages[0] = 0

        cpu_delta = cpu_usages - prev_cpu_usages
        system_delta = curr_system_usages - prev_system_usages
        mask = (system_delta > 0.0) & (cpu_delta > 0.0)
        cpu_utilization_list[mask] = (cpu_delta[mask] / system_delta[mask]) * num_cores[mask] * 100.0
        cpu_utilization_list[0] = 0
        df_sub['cpu_util'] = cpu_utilization_list
        df_list.append(df_sub)

    if len(df_list) != 0:
        df = pd.concat(df_list)

        # Convert 'timestamp' column to datetime data type
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        min_timestamp = df['timestamp'].min()
        df['timestamp'] = (df['timestamp'] - min_timestamp).dt.total_seconds()

        # Create a new DataFrame for aggregated data
        agg_df = pd.DataFrame(columns=['timestamp', 'cpu_util_agg'])

        # Calculate the aggregated values
        current_time = df['timestamp'].iloc[0]
        while current_time <= df['timestamp'].iloc[-1]:
            next_time = current_time + 0.5
            mask = (df['timestamp'] >= current_time) & (df['timestamp'] < next_time)
            cpu_util_sum = df[mask]['cpu_util'].sum() / 100.0
            agg_df = pd.concat([agg_df, pd.DataFrame({'timestamp': [current_time], 'cpu_util_agg': [cpu_util_sum]})])
            current_time = next_time
        
        agg_df = agg_df[agg_df['timestamp'] < 1000]
        sns.set(style="whitegrid")
        plt.figure(figsize=(16, 6))
        ax = sns.lineplot(x='timestamp', y='cpu_util_agg', data=agg_df)
        ax.axhline(y=90, color='orange', linestyle='--', label='90 Cores')
        plt.xlabel('Timestamp')
        plt.ylabel('Aggregated CPU Utilization')
        plt.savefig(f'./{SYSTEM}-invoker-{INVOKER_NAME}-cpu-util-timeline.png', bbox_inches='tight', dpi=300)
        plt.close()

    # Close the SQLite connection
    conn.close()

if __name__=='__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description='Script to create CPU utilization graph at each invoker')
    parser.add_argument('--invoker-name', dest='invoker_name', default='w1', help='Invoker name [w1, w2...]')
    parser.add_argument('--system', dest='system', default='full-lachesis-rps-1', help='system name with rps value')
    args = parser.parse_args()

    INVOKER_NAME = args.invoker_name
    SYSTEM = args.system

    create_cpu_util_timeline()