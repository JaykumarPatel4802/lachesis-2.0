import grpc
import pandas as pd
from sklearn.utils import shuffle
import subprocess
import json
import time
import random
import seaborn as sns
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import requests
import urllib3
import os

from generated import lachesis_pb2_grpc, lachesis_pb2, cypress_pb2_grpc, cypress_pb2

# Used for interacting with OpenWhisk API
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
OPENWHISK_API_URL = 'https://127.0.1.1:443'
AUTHORIZATION_KEY = 'MjNiYzQ2YjEtNzFmNi00ZWQ1LThjNTQtODE2YWE0ZjhjNTAyOjEyM3pPM3haQ0xyTU42djJCS0sxZFhZRnBYbFBrY2NPRnFtMTJDZEFzTWdSVTRWck5aOWx5R1ZDR3VNREdJd1A='
RESULTS_PER_PAGE = 100  # Adjust as needed

feature_dict = {'floatmatmult': shuffle(pd.read_csv('../data/vw-prediction-inputs/floatmatmult-inputs.csv'), random_state=0), 
                'imageprocess': shuffle(pd.read_csv('../data/vw-prediction-inputs/imageprocess-inputs.csv'), random_state=0),
                'videoprocess': shuffle(pd.read_csv('../data/vw-prediction-inputs/videoprocess-inputs.csv'), random_state=0),
                # 'transcribe': shuffle(pd.read_csv('../data/vw-prediction-inputs/audio-inputs.csv'), random_state=0),
                'sentiment': shuffle(pd.read_csv('../data/vw-prediction-inputs/sentiment-inputs.csv'), random_state=0),
                'lrtrain': shuffle(pd.read_csv('../data/vw-prediction-inputs/lrtrain-inputs.csv'), random_state=0), 
                'mobilenet': shuffle(pd.read_csv('../data/vw-prediction-inputs/mobilenet-inputs.csv'), random_state=0),
                'encrypt': shuffle(pd.read_csv('../data/vw-prediction-inputs/encrypt-inputs.csv'), random_state=0),
                'linpack': shuffle(pd.read_csv('../data/vw-prediction-inputs/linpack-inputs.csv'), random_state=0)}

cypress_feature_dict = {'floatmatmult': shuffle(pd.read_csv('../data/vw-prediction-inputs/cypress-floatmatmult-inputs.csv'), random_state=0), 
                        'imageprocess': shuffle(pd.read_csv('../data/vw-prediction-inputs/cypress-imageprocess-inputs.csv'), random_state=0),
                        'videoprocess': shuffle(pd.read_csv('../data/vw-prediction-inputs/cypress-videoprocess-inputs.csv'), random_state=0),
                        # 'transcribe': shuffle(pd.read_csv('../data/vw-prediction-inputs/cypress-audio-inputs.csv'), random_state=0),
                        'sentiment': shuffle(pd.read_csv('../data/vw-prediction-inputs/cypress-sentiment-inputs.csv'), random_state=0),
                        'lrtrain': shuffle(pd.read_csv('../data/vw-prediction-inputs/cypress-lrtrain-inputs.csv'), random_state=0), 
                        'mobilenet': shuffle(pd.read_csv('../data/vw-prediction-inputs/cypress-mobilenet-inputs.csv'), random_state=0),
                        'encrypt': shuffle(pd.read_csv('../data/vw-prediction-inputs/cypress-encrypt-inputs.csv'), random_state=0),
                        'linpack': shuffle(pd.read_csv('../data/vw-prediction-inputs/cypress-linpack-inputs.csv'), random_state=0)}

functions = ['floatmatmult', 'imageprocess', 'videoprocess', 'sentiment', 'lrtrain', 'mobilenet', 'encrypt', 'linpack']

SLO_MULTIPLIER = 0.4

'''
Plotting Functions
'''
def get_activations(limit):
    headers = {
        'Authorization': f'Basic {AUTHORIZATION_KEY}',
        'Content-Type': 'application/json',
    }

    activations = []
    total_fetched = 0

    while total_fetched < limit:
        # Calculate the number of activations to fetch in this iteration
        remaining_to_fetch = limit - total_fetched
        fetch_count = min(remaining_to_fetch, RESULTS_PER_PAGE)

        # Calculate the offset for pagination
        offset = total_fetched

        # Make a GET request to fetch activations with SSL certificate verification disabled
        response = requests.get(
            f'{OPENWHISK_API_URL}/api/v1/namespaces/_/activations',
            headers=headers,
            params={'limit': fetch_count, 'skip': offset},
            verify=False  # Disable SSL certificate verification
        )

        if response.status_code == 200:
            activations.extend(response.json())
            total_fetched += fetch_count
        else:
            print(f'Failed to retrieve activations. Status code: {response.status_code}')
            break

    return activations

def create_activation_df(limit=2000):
    activations = get_activations(limit)
    
    if activations:
        # Initialize lists to store data
        activation_ids = []
        cpu_limits = []
        memory_limits = []
        wait_times = []
        init_times = []
        durations = []
        names = []
        start_times = []
        end_times = []
        status_codes = []

        for activation in activations:
            # Extract data from the activation JSON
            activation_id = activation['activationId']
            annotation = next((ann for ann in activation['annotations'] if ann['key'] == 'limits'), None)
            cpu_limit = annotation['value']['cpu'] if annotation else None
            memory_limit = annotation['value']['memory'] if annotation else None
            wait_time = next((ann['value'] for ann in activation['annotations'] if ann['key'] == 'waitTime'), 0)
            init_time = next((ann['value'] for ann in activation['annotations'] if ann['key'] == 'initTime'), 0)
            duration = activation['duration']
            name = activation['name'].split('_')[0]
            start_time = activation['start']
            end_time = activation['end']
            status_code = activation.get('statusCode', None)

            # Append extracted data to lists
            activation_ids.append(activation_id)
            cpu_limits.append(cpu_limit)
            memory_limits.append(memory_limit)
            wait_times.append(wait_time)
            init_times.append(init_time)
            durations.append(duration)
            names.append(name)
            start_times.append(start_time)
            end_times.append(end_time)
            status_codes.append(status_code)

        # Create a DataFrame from the lists
        data = {
            'activation_id': activation_ids,
            'cpu': cpu_limits,
            'memory': memory_limits,
            'wait_time': wait_times,
            'init_time': init_times,
            'duration': durations,
            'name': names,
            'start_time': start_times,
            'end_time': end_times,
            'status_code': status_codes,
        }

        df = pd.DataFrame(data)
        return df

def create_experiment_data(exp):
    conn = sqlite3.connect('./datastore/lachesis-controller.db')
    df = pd.read_sql_query('SELECT * FROM fxn_exec_data', conn)
    df = df[df['exp_no'] == f'{exp}']
    df['total_duration'] = df['duration'] + df['cold_start_latency']
    df['slo_violation'] = df.apply(lambda row: 1 if row['total_duration'] > row['slo'] else 0, axis=1)
    df['no_cores'] = np.ceil(df['max_cpu']).astype(int)
    df['idle_cores'] = df['cpu_limit'] - df['no_cores']
    df['idle_cores'] = df['idle_cores'].apply(lambda x: max(0, x))
    df['slack'] = df['slo'] - df['total_duration']

    # Get a summary per function - number of runs per function, number of completions per function
    print('Summary of Executions per Function')
    for fxn in df['function'].unique():
        df_fxn = df[df['function'] == fxn]
        no_rows = len(df_fxn)
        no_unfinished_rows = len(df_fxn[df_fxn['start_time'] == 'NA'])
        no_finished_rows = no_rows - no_unfinished_rows
        print(f'{fxn}: {no_finished_rows} finished / {no_rows} total')
    print(f'Total invocations: {len(df)}')
    df = df[df['lachesis_end'] != 'NA']
    print(f'Total finished: {len(df)}')
    df['activation_id'] = df['activation_id'].str.strip('"')
    return(df)

def plot_slo_violation_summary(df):

    order = functions
    violation_summary_dict = {}
    for fxn in df['function'].unique():
        df_fxn = df[df['function'] == fxn]
        no_rows = len(df_fxn)
        no_violations = len(df_fxn[df_fxn['slo_violation'] == 1])
        violation_summary_dict[fxn] = [no_violations, no_rows, no_violations/no_rows * 100]
    df_summary = pd.DataFrame.from_dict(violation_summary_dict, orient='index', columns=['no_violations', 'no_rows', 'percentage'])
    df_summary.reset_index(inplace=True)
    df_summary.rename(columns={'index': 'fxn'}, inplace=True)

    custom_order = {func: i for i, func in enumerate(order)}
    df_summary['custom_sort'] = df_summary['fxn'].map(custom_order)
    df_summary = df_summary.sort_values(by='custom_sort')
    df_summary = df_summary.drop('custom_sort', axis=1)
    df_summary = df_summary.reset_index(drop=True)

    ax = sns.barplot(x='fxn', y='percentage', data=df_summary, order=order)
    plt.xlabel('Function')
    plt.ylabel('Percentage of SLO Violations (%)')
    plt.xticks(rotation=45)
    ax.set_ylim(0, 110)

    for p, no_rows in zip(ax.patches, df_summary['no_rows']):
        ax.annotate(f'{no_rows}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 10), 
                    textcoords='offset points')
    plt.savefig(f'../study-ow-scheduler/summary-slo-violations', bbox_inches='tight', dpi=300)
    plt.close()

def plot_slo_violation_summary_system_comparison(df):
    order = functions
    violation_summary_dict = {}
    for system in df['system'].unique():
        df_system = df[df['system'] == system]
        for fxn in df_system['function'].unique():
            df_fxn = df_system[df_system['function'] == fxn]
            no_rows = len(df_fxn)
            no_violations = len(df_fxn[df_fxn['slo_violation'] == 1])
            violation_summary_dict[(fxn, system)] = [no_violations, no_rows, no_violations/no_rows * 100]
    df_summary = pd.DataFrame.from_dict(violation_summary_dict, orient='index', columns=['no_violations', 'no_rows', 'percentage'])
    df_summary.reset_index(inplace=True)
    df_summary.rename(columns={'index': 'fxn-system'}, inplace=True)
    df_summary[['fxn', 'system']] = pd.DataFrame(df_summary['fxn-system'].tolist(), index=df_summary.index)

    custom_order = {func: i for i, func in enumerate(order)}
    df_summary['custom_sort'] = df_summary['fxn'].map(custom_order)
    df_summary = df_summary.sort_values(by='custom_sort')
    df_summary = df_summary.drop('custom_sort', axis=1)
    df_summary = df_summary.reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    hue_order = ['LRA/LSCHED', 'LRA/DEFAULT', 'S4/LSCHED', 'S4/DEFAULT', 'S12/LSCHED', 'S12/DEFAULT', 'S20/LSCHED', 'S20/DEFAULT']
    ax = sns.barplot(x='fxn', y='percentage', data=df_summary, hue='system', hue_order=hue_order, order=order)
    plt.xlabel('Function')
    plt.ylabel('Percentage of SLO Violations (%)')
    plt.legend(ncol=2)
    plt.xticks(rotation=45)
    ax.set_ylim(0, 110)

    # for p, no_rows in zip(ax.patches, df_summary['no_rows']):
    #     ax.annotate(f'{no_rows}', 
    #                 (p.get_x() + p.get_width() / 2., p.get_height()), 
    #                 ha='center', va='center', fontsize=10, color='black', xytext=(0, 10), 
    #                 textcoords='offset points')
    plt.savefig(f'../study-ow-scheduler/summary-slo-violations-system-comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_cold_start_slo_summary(df):
    # Of the invocations that were an SLO violation, how many had cold starts
    order = functions
    df_violations = df[df['slo_violation'] == 1]
    cold_start_summary = {}
    for fxn in df_violations['function'].unique():
        df_fxn = df_violations[df_violations['function'] == fxn]
        no_rows = len(df_fxn)
        no_cold_starts = len(df_fxn[df_fxn['cold_start_latency'] != 0])
        cold_start_summary[fxn] = [no_cold_starts, no_rows, no_cold_starts/no_rows * 100]
    df_summary = pd.DataFrame.from_dict(cold_start_summary, orient='index', columns=['no_cold_starts', 'no_slo_violations', 'percentage'])
    df_summary.reset_index(inplace=True)
    df_summary.rename(columns={'index': 'fxn'}, inplace=True)
    
    custom_order = {func: i for i, func in enumerate(order)}
    df_summary['custom_sort'] = df_summary['fxn'].map(custom_order)
    df_summary = df_summary.sort_values(by='custom_sort')
    df_summary = df_summary.drop('custom_sort', axis=1)
    df_summary = df_summary.reset_index(drop=True)

    ax = sns.barplot(x='fxn', y='percentage', data=df_summary, order=order)
    plt.xlabel('Function')
    plt.ylabel('Percentage of SLO Violations\nWith Cold Starts(%)')
    plt.xticks(rotation=45)
    ax.set_ylim(0, 110)

    for p, no_slo_violations in zip(ax.patches, df_summary['no_slo_violations']):
        ax.annotate(f'{no_slo_violations}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 10), 
                    textcoords='offset points')
    plt.savefig(f'../study-ow-scheduler/summary-violation-cold-starts', bbox_inches='tight', dpi=300)
    plt.close()

def plot_cold_start_slo_summary_system_comparison(df):
    # Of the invocations that were an SLO violation, how many had cold starts
    order = functions
    df_violations = df[df['slo_violation'] == 1]
    cold_start_summary = {}
    for system in df_violations['system'].unique():
        df_system = df_violations[df_violations['system'] == system]
        for fxn in df_system['function'].unique():
            df_fxn = df_system[df_system['function'] == fxn]
            no_rows = len(df_fxn)
            no_cold_starts = len(df_fxn[df_fxn['cold_start_latency'] != 0])
            cold_start_summary[(fxn, system)] = [no_cold_starts, no_rows, no_cold_starts/no_rows * 100]
    df_summary = pd.DataFrame.from_dict(cold_start_summary, orient='index', columns=['no_cold_starts', 'no_slo_violations', 'percentage'])
    df_summary.reset_index(inplace=True)
    df_summary.rename(columns={'index': 'fxn-system'}, inplace=True)
    df_summary[['fxn', 'system']] = pd.DataFrame(df_summary['fxn-system'].tolist(), index=df_summary.index)
    
    custom_order = {func: i for i, func in enumerate(order)}
    df_summary['custom_sort'] = df_summary['fxn'].map(custom_order)
    df_summary = df_summary.sort_values(by='custom_sort')
    df_summary = df_summary.drop('custom_sort', axis=1)
    df_summary = df_summary.reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    hue_order = ['LRA/LSCHED', 'LRA/DEFAULT', 'S4/LSCHED', 'S4/DEFAULT', 'S12/LSCHED', 'S12/DEFAULT', 'S20/LSCHED', 'S20/DEFAULT']
    ax = sns.barplot(x='fxn', y='percentage', data=df_summary, hue='system', hue_order=hue_order, order=order)
    plt.xlabel('Function')
    plt.ylabel('Percentage of SLO Violations\nWith Cold Starts(%)')
    plt.legend(ncol=2)
    plt.xticks(rotation=45)
    ax.set_ylim(0, 110)

    # for p, no_slo_violations in zip(ax.patches, df_summary['no_slo_violations']):
    #     ax.annotate(f'{no_slo_violations}', 
    #                 (p.get_x() + p.get_width() / 2., p.get_height()), 
    #                 ha='center', va='center', fontsize=10, color='black', xytext=(0, 10), 
    #                 textcoords='offset points')
    plt.savefig(f'../study-ow-scheduler/summary-violation-cold-starts-system-comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_idle_cores_summary(df):
    df = df[df['max_cpu'] > 0]
    order = functions
    ax = sns.barplot(x='function', y='idle_cores', data=df, order=order)
    plt.xlabel('Function')
    plt.ylabel('Num. Idle Cores')
    plt.xticks(rotation=45)
    ax.set_ylim(0, 32)
    plt.savefig(f'../study-ow-scheduler/summary-idle-cores', bbox_inches='tight', dpi=300)
    plt.close()

def plot_idle_cores_summary_system_comparison(df):
    df = df[df['max_cpu'] > 0]
    # print(df[df['function'] == 'linpack'][['inputs', 'duration', 'cold_start_latency', 'activation_id', 'invoker_name', 'idle_cores', 'cpu_limit', 'scheduled_cpu', 'p90_cpu', 'p95_cpu', 'p99_cpu', 'max_cpu', 'system']].to_string())
    order = functions
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x='function', y='idle_cores', hue='system', hue_order=['lachesis', 'static-4', 'static-12', 'static-20'], order=order, data=df)
    plt.xlabel('Function')
    plt.ylabel('Num. Idle Cores')
    plt.xticks(rotation=45)
    ax.set_ylim(0, 32)
    plt.savefig(f'../study-ow-scheduler/summary-idle-cores-system-comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_assigned_cpu_limit_timeline(df):
     # Look at assigned CPU limit over time
    for fxn in df['function'].unique():
        df_fxn = df[df['function'] == fxn]
        plt.figure(figsize=(10, 6))  
        sns.lineplot(data=df_fxn, x=df_fxn.index, y='cpu_limit', label='CPU Limit')
        sns.lineplot(data=df_fxn, x=df_fxn.index, y='max_cpu', label='Max CPU')
        plt.xlabel('Time')
        plt.ylabel('CPU Value') 
        plt.legend()
        plt.savefig(f'../study-ow-scheduler/cpu-limit-timeline-{fxn}.png', bbox_inches='tight', dpi=300)
        plt.close()

def plot_assigned_mem_limit_timeline(df):

    for fxn in df['function'].unique():
        df_fxn = df[df['function'] == fxn]
        df_fxn = df_fxn[df_fxn['max_mem'] > 0]
        df_fxn['max_mem'] = df_fxn['max_mem'].where(df_fxn['max_mem'] <= df_fxn['scheduled_mem'], df_fxn['scheduled_mem'] - 128)

        plt.figure(figsize=(10, 6))
        
        # Calculate the top bar values (mem_limit - max_mem)
        df_fxn['top_bar'] = df_fxn['scheduled_mem'] - df_fxn['max_mem']
        
        # Create a stacked barplot
        x_values = range(len(df_fxn))
        plt.bar(x_values, df_fxn['max_mem'], label='Max Mem Used')
        plt.bar(x_values, df_fxn['top_bar'], bottom=df_fxn['max_mem'], label='Mem Limit')
        
        plt.xlabel('Invocation #')
        plt.ylabel('Mem Value (MB)')
        plt.legend()
        
        plt.savefig(f'../study-ow-scheduler/mem-limit-usage-timeline-{fxn}.png', bbox_inches='tight', dpi=300)
        plt.close()

def plot_slack_timeline(df):
    # Print slack over time
    for fxn in df['function'].unique():
        df_fxn = df[df['function'] == fxn]
        plt.figure(figsize=(10, 6))  
        sns.lineplot(data=df_fxn, x=df_fxn.index, y='slack')
        plt.axhline(y=0, linestyle='--', color='orange', linewidth=1)
        plt.xlabel('Time')
        plt.ylabel('Slack (Negative = SLO violation)') 
        plt.savefig(f'../study-ow-scheduler/slack-timeline-{fxn}.png', bbox_inches='tight', dpi=300)
        plt.close()

def plot_invoker_aggreagte_cpu_limit_usage_timeline(df):
     # Analyze the cpu limits and cpu usage on each invoker
    for name in df['invoker_name'].unique():
        df_sub = df[df['invoker_name'] == name]
        # Convert 'end_time' column to datetime objects
        df_sub['end_time'] = pd.to_datetime(df_sub['end_time'], format="%Y-%m-%dT%H:%M:%S.%fZ")

        # Step 1: Sort the DataFrame by 'end_time' in ascending order
        df_sub = df_sub.sort_values(by='end_time')

        # Step 2 and 3: Iterate through the DataFrame in 5-second intervals and sum 'CPU_limit' and 'max_cpu'
        interval_length = timedelta(seconds=0.5)
        results = []

        current_time = df_sub['end_time'].iloc[0]
        counter = 5
        while current_time <= df_sub['end_time'].iloc[-1]:
            interval_df = df_sub[(df_sub['end_time'] >= current_time) & (df_sub['end_time'] < current_time + interval_length)]
            cpu_limit_sum = interval_df['cpu_limit'].sum()
            max_cpu_sum = interval_df['max_cpu'].sum()
            
            results.append({
                'Invoker': name,
                'Time': counter,
                'Sum CPU_limit': cpu_limit_sum,
                'Sum max_cpu': max_cpu_sum
            })
            
            current_time += interval_length
            counter += 5

        result_df = pd.DataFrame(results)
        # print(result_df)

        plt.figure(figsize=(10, 6))  
        sns.lineplot(data=result_df, x='Time', y='Sum CPU_limit', label='summed cpu limits')
        sns.lineplot(data=result_df, x='Time', y='Sum max_cpu', label='summed max cpu used')
        plt.axhline(y=80, linestyle='--', color='orange', linewidth=1)
        plt.xlabel('Time')
        plt.ylabel('Summed CPU Values') 
        plt.legend()
        plt.savefig(f'../study-ow-scheduler/summed-cpu-limits-invoker-{name}-timeline.png', bbox_inches='tight', dpi=300)
        plt.close()

def plot_invoker_aggregate_cpu_limit_timeline(df):
    for name in df['invoker_name'].unique():
        df_sub = df[df['invoker_name'] == name].copy()
        df_sub['start_time'] = pd.to_datetime(df_sub['start_time'], format="%Y-%m-%dT%H:%M:%S.%fZ")
        df_sub['end_time'] = pd.to_datetime(df_sub['end_time'], format="%Y-%m-%dT%H:%M:%S.%fZ")

        # Sort the DataFrame by 'start_time'
        df_sub = df_sub.sort_values(by='start_time')

        # Initialize variables
        curr_time = df_sub['start_time'].iloc[0]
        end_times = df_sub['end_time'].values

        # Create an empty list to store the results
        results = []

        # Loop until curr_time is less than the largest end_time
        while curr_time < max(end_times):
            # Filter rows based on the conditions
            relevant_rows = df_sub[(df_sub['start_time'] <= curr_time) & (curr_time < df_sub['end_time'])]
            
            # Calculate the sum of 'cpu_limit' for the relevant rows
            cpu_sum = relevant_rows['cpu_limit'].sum()
            
            # Append the result to the list
            results.append({'Time': curr_time, 'Aggregate CPU Assignment': cpu_sum})
            
            # Increment curr_time by 0.5 seconds
            curr_time += pd.Timedelta(seconds=0.5)

        # Create a new DataFrame from the results
        result_df = pd.DataFrame(results)

        plt.figure(figsize=(10, 6))  
        sns.lineplot(data=result_df, x='Time', y='Aggregate CPU Assignment')
        plt.axhline(y=90, linestyle='--', color='orange', linewidth=1) 
        plt.savefig(f'../study-ow-scheduler/summed-cpu-limits-invoker-{name}-timeline.png', bbox_inches='tight', dpi=300)
        plt.close()

def plot_slo_violation_breakdown(df):
    df_activations = create_activation_df()
    df_plot = df.merge(df_activations[['activation_id', 'wait_time']], on='activation_id', how='left')
    df_plot['duration - slo'] = df_plot['slack'].abs()
    df_plot['cold_start_latency + wait_time'] = df_plot['cold_start_latency'] + df_plot['wait_time']

    df_melted = df_plot.melt(id_vars=['function'], value_vars=['duration - slo', 'cold_start_latency', 'wait_time', 'cold_start_latency + wait_time'])

    plt.figure(figsize=(12, 6))
    # sns.set(style="whitegrid")
    p1 = sns.barplot(x="function", y="value", hue="variable", data=df_melted)

    plt.xlabel("Function")
    plt.ylabel("Amount of Time (ms)")
    plt.savefig(f'../study-ow-scheduler/slack-breakdown.png', bbox_inches='tight', dpi=300)
    plt.close()
    # print(df_plot[['duration', 'slack', 'cold_start_latency', 'wait_time', 'activation_id']].to_string())

def plot_per_invoker_data(df):
    
    # Plot breakdown of invocations per invoker per function
    df = df.sort_values(by=['invoker_name', 'function'])
    plt.figure(figsize=(16, 6))
    sns.set(style="whitegrid")
    sns.countplot(x="invoker_name", hue="function", data=df, order=['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8'])
    plt.xlabel("Invoker Name")
    plt.ylabel("# of Invocations")
    plt.savefig(f'../study-ow-scheduler/function-breakdown-per-invoker.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Plot SLO violation percentage per invoker
    order = functions

    # Create a dictionary to store the violation summary
    violation_summary_dict = {}

    # Group the data by 'invoker_name' and 'function' and compute the SLO violation percentage
    for invoker in df['invoker_name'].unique():
        for fxn in df['function'].unique():
            df_invoker_fxn = df[(df['invoker_name'] == invoker) & (df['function'] == fxn)]
            no_rows = len(df_invoker_fxn)
            no_violations = len(df_invoker_fxn[df_invoker_fxn['slo_violation'] == 1])
            percentage = 0  
            if no_rows != 0:
                percentage = (no_violations / no_rows) * 100
            violation_summary_dict[f'{invoker}_{fxn}'] = [no_violations, no_rows, percentage]

    # Create a DataFrame from the dictionary
    df_summary = pd.DataFrame.from_dict(violation_summary_dict, orient='index', columns=['no_violations', 'no_rows', 'percentage'])
    df_summary.reset_index(inplace=True)
    df_summary.rename(columns={'index': 'invoker_fxn'}, inplace=True)
    df_summary[['invoker_name', 'function']] = df_summary['invoker_fxn'].str.split('_', expand=True)

    # Sort the DataFrame based on the 'fxn' column using the custom order
    df_summary = df_summary.sort_values(by=['invoker_name', 'function'])
    
    plt.figure(figsize=(16, 6))
    ax = sns.barplot(x='invoker_name', y='percentage', hue='function', data=df_summary, order=['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8'])
    plt.xlabel('Invoker Name')
    plt.ylabel('Percentage of SLO Violations (%)')
    ax.set_ylim(0, 110)

    # Annotate the bars with the number of rows
    # for p, no_rows in zip(ax.patches, df_summary['no_rows']):
    #     ax.annotate(f'{no_rows}', 
    #                 (p.get_x() + p.get_width() / 2., p.get_height()), 
    #                 ha='center', va='center', fontsize=10, color='black', xytext=(0, 10), 
    #                 textcoords='offset points')
    
    plt.savefig(f'../study-ow-scheduler/slo-violation-breakdown-per-invoker.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_cpu_util_timeline():

    subprocess.run(['bash', '../scripts/lachesis/copy-data.sh'])
    directory_path = './tmp-daemon-data'
    for filename in os.listdir(directory_path):
        invoker = filename.split('.')[0]
        file_path = os.path.join(directory_path, filename)

        conn = sqlite3.connect(file_path)
        df_invoker = pd.read_sql_query("SELECT * FROM function_utilization_advanced;", conn)

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
            ax.axhline(y=70, color='orange', linestyle='--', label='70 Cores')
            plt.xlabel('Timestamp')
            plt.ylabel('Aggregated CPU Utilization')
            plt.savefig(f'../study-ow-scheduler/invoker-{invoker}-cpu-util-timeline.png', bbox_inches='tight', dpi=300)
            plt.close()

            # Close the SQLite connection
            conn.close()

    # List all files in the directory
    files = os.listdir(directory_path)

    # Iterate over the files and delete each one
    for file in files:
        file_path = os.path.join(directory_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")

def plot_core_breakdown(df):

    grouped = df.groupby('function')
    for function, group_data in grouped:

        # ax = plt.gca()

        data_to_plot = group_data[['cpu_limit', 'scheduled_cpu']]
        data_to_plot['scheduled_cpu'] = data_to_plot['scheduled_cpu'] - data_to_plot['cpu_limit']
        ax = data_to_plot.plot(kind='bar', stacked=True, color=['skyblue', 'green'], figsize=(12,6), xticks=np.arange(0, len(data_to_plot), 5.0))

        ax.set_xlabel('Invocation')
        ax.set_ylabel('Number of Cores')

        ax.legend()
        plt.xticks(rotation=0)
        plt.savefig(f'../study-ow-scheduler/core-assigned-{function}-scheduled.png', bbox_inches='tight', dpi=300)
        plt.close() 

def plot_cold_start_percentages(curr_exp):
    df2 = create_experiment_data(curr_exp)
    df2['activation_id'] = df2['activation_id'].str.strip('"')
    
    data = []
    for function in df2['function'].unique():
        no_invocations = len(df2[(df2['function'] == function)])
        percentage_cold_starts = (len(df2[(df2['function'] == function) & (df2['cold_start_latency'] > 0)]) / no_invocations) * 100
        data.append({'function': function, 'no_invocations': no_invocations, 'percentage_cold_starts': percentage_cold_starts, 'scheduler': 'Container Aware'})

    df_plot = pd.DataFrame(data)
    df_plot.reset_index(drop=True, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.set(style='whitegrid')
    sns.barplot(data=df_plot, x='function', y='percentage_cold_starts')

    plt.xlabel('Function')
    plt.ylabel('Percentage of Invocations\n with Cold Starts')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)

    plt.savefig(f'../study-ow-scheduler/scheduler-cold-start.png', bbox_inches='tight', dpi=300)
    plt.close() 

def plot_cold_start_percentages_system_comparison(df):
    data = []
    for system in df['system'].unique():
        df2 = df[df['system'] == system]
        for function in df2['function'].unique():
            no_invocations = len(df2[(df2['function'] == function)])
            percentage_cold_starts = (len(df2[(df2['function'] == function) & (df2['cold_start_latency'] > 0)]) / no_invocations) * 100
            data.append({'function': function, 'no_invocations': no_invocations, 'percentage_cold_starts': percentage_cold_starts, 'system': system})
    
    df_plot = pd.DataFrame(data)
    df_plot.reset_index(drop=True, inplace=True)

    plt.figure(figsize=(14,6))
    hue_order = ['LRA/LSCHED', 'LRA/DEFAULT', 'S4/LSCHED', 'S4/DEFAULT', 'S12/LSCHED', 'S12/DEFAULT', 'S20/LSCHED', 'S20/DEFAULT']
    sns.barplot(data=df_plot, x='function', y='percentage_cold_starts', hue='system', hue_order=hue_order)

    plt.xlabel('Function')
    plt.ylabel('Percentage of Invocations\nw/ Cold Starts')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.legend(ncol=2)

    plt.savefig(f'../study-ow-scheduler/scheduler-cold-start-system-comparison.png', bbox_inches='tight', dpi=300)
    plt.close() 

def plot_idle_core_cdf(df):
    plt.figure(figsize=(14,6))
    sns.ecdfplot(data=df, x='idle_cores')
    plt.xlabel('Number of Idle Cores')
    plt.ylabel('Cumulative Distribution')
    plt.xticks(rotation=45)

    plt.savefig(f'../study-ow-scheduler/idle-cold-start.png', bbox_inches='tight', dpi=300)
    plt.close() 

def plot_idle_core_cdf_system_comparison(df):
    plt.figure(figsize=(14,6))
    hue_order = ['LRA/LSCHED', 'LRA/DEFAULT', 'S4/LSCHED', 'S4/DEFAULT', 'S12/LSCHED', 'S12/DEFAULT', 'S20/LSCHED', 'S20/DEFAULT']
    sns.ecdfplot(data=df, x='idle_cores', hue='system', hue_order=hue_order)

    plt.xlabel('Number of Idle Cores')
    plt.ylabel('Cumulative Distribution')
    plt.xticks(rotation=45)
    # plt.legend(ncol=2)

    plt.savefig(f'../study-ow-scheduler/idle-cold-start-system-comparison.png', bbox_inches='tight', dpi=300)
    plt.close() 

def explore_function(df, function):
    df = df[df['function'] == function]
    df = df[df['max_mem'] > 0]
    print(df[['inputs', 'duration', 'cold_start_latency', 'activation_id', 'invoker_name', 'idle_cores', 'cpu_limit', 'mem_limit', 'scheduled_cpu', 'scheduled_mem', 'p90_cpu', 'p95_cpu', 'p99_cpu', 'max_cpu', 'max_mem']].to_string())

def explore_invoker(df, invoker_name):
    df = df[df['invoker_name'] == invoker_name]
    print(df)

def plot_results():

    # exp = 'exp_4_slo_0.2_invoker_mem_75gb'
    # exp = 'exp_3_slo_0.1_invoker_mem_75gb'
    # exp = 'exp_2_slo_0.1'
    # exp = 'exp_12_slo_0.5_invoker_mem_75gb_mem_load_balancer_rps_2'
    # exp = 'exp_13_slo_0.5_quantile_50_invoker_mem_75gb_mem_load_balancer_rps_2'
    # exp = 'exp_14_slo_0.25_quantile_50_invoker_mem_75gb_mem_load_balancer_rps_2'
    # exp = 'exp_15_slo_0.4_quantile_50_invoker_mem_75gb_mem_load_balancer_rps_2'
    # exp = 'exp_16_slo_0.4_quantile_50_invoker_mem_75gb_full_load_balancer_rps_2'
    # exp = 'exp_17_slo_0.4_quantile_50_invoker_mem_75gb_cold_start_scheduler_rps_2'
    # exp = 'exp_18_slo_0.4_quantile_50_invoker_mem_125gb_cold_start_scheduler_rps_2'
    # exp = 'exp_19_slo_0.4_quantile_50_invoker_mem_125gb_cold_start_scheduler_rps_2'
    # exp = 'exp_20_slo_0.4_quantile_50_invoker_mem_125gb_cold_start_scheduler_custom_mem_rps_2'
    # exp = 'exp_21_slo_0.4_quantile_50_invoker_mem_125gb_cold_start_scheduler_custom_mem_rps_2_admission_control'
    # exp = 'exp_23_slo_0.4_quantile_50_invoker_mem_125gb_cpu_140_cold_start_scheduler_custom_mem_rps_2_admission_control'
    # exp = 'exp_24_slo_0.4_quantile_50_invoker_mem_125gb_cpu_120_cold_start_scheduler_custom_mem_rps_2_admission_control'
    # exp = 'exp_26_slo_0.4_quantile_50_invoker_mem_125gb_cpu_120_cold_start_scheduler_custom_mem_rps_2'
    # exp_lachesis = 'exp_27_slo_0.4_quantile_50_invoker_mem_125gb_cpu_90_cold_start_scheduler_custom_mem_rps_2'
    # exp = 'exp_44_slo_0.4_quantile_50_invoker_mem_125gb_lachesis_scheduler_lachesis_ra_custom_mem_rps_2' # linpack 50 memory severity
    # exp = 'exp_45_slo_0.4_quantile_50_invoker_mem_125gb_lachesis_scheduler_lachesis_ra_custom_mem_rps_2' # floatmatmult 50 memory severity
    # exp = 'exp_46_linpack_severity_35_15_invocation_memory_prediction_test' # linpack 35 memory severity
    # exp = 'exp_47_linpack_severity_40_15_invocation_memory_prediction_test' # linpack 40 memory severity
    exp = 'exp_48_slo_0.4_quantile_50_rps_2_full_lachesis'
    df = create_experiment_data(exp)

    # exp_lachesis_full = 'exp_37_slo_0.4_quantile_50_invoker_mem_125gb_cpu_90_cold_start_scheduler_background_custom_mem_rps_2'
    # exp_lachesis_default = 'exp_41_slo_0.4_quantile_50_invoker_mem_125gb_default_scheduler_lachesis_ra_custom_mem_rps_2'
    # exp_static_small_lachesis_scheduler = 'exp_36_slo_0.4_quantile_50_invoker_mem_125gb_cpu_90_lachesis_scheduler_custom_mem_rps_2_static-small'
    # exp_static_small_default_scheduler = 'exp_40_slo_0.4_quantile_50_invoker_mem_125gb_default_scheduler_custom_mem_rps_2_static-small'
    # exp_static_medium_lachesis_scheduler = 'exp_35_slo_0.4_quantile_50_invoker_mem_125gb_cpu_90_lachesis_scheduler_custom_mem_rps_2_static-medium'
    # exp_static_medium_default_scheduler = 'exp_39_slo_0.4_quantile_50_invoker_mem_125gb_default_scheduler_custom_mem_rps_2_static-medium'
    # exp_static_large_lachesis_scheduler = 'exp_34_slo_0.4_quantile_50_invoker_mem_125gb_cpu_90_lachesis_scheduler_custom_mem_rps_2_static-large'
    # exp_static_large_default_scheduler = 'exp_38_slo_0.4_quantile_50_invoker_mem_125gb_default_scheduler_custom_mem_rps_2_static-large'

    # df_lachesis_full = create_experiment_data(exp_lachesis_full)
    # df_lachesis_full['system'] = 'LRA/LSCHED'
    # df_lachesis_default = create_experiment_data(exp_lachesis_default)
    # df_lachesis_default['system'] = 'LRA/DEFAULT'
    # df_small_lsched = create_experiment_data(exp_static_small_lachesis_scheduler)
    # df_small_lsched['system'] = 'S4/LSCHED'
    # df_small_default = create_experiment_data(exp_static_small_default_scheduler)
    # df_small_default['system'] = 'S4/DEFAULT'
    # df_medium_lsched = create_experiment_data(exp_static_medium_lachesis_scheduler)
    # df_medium_lsched['system'] = 'S12/LSCHED'
    # df_medium_default = create_experiment_data(exp_static_medium_default_scheduler)
    # df_medium_default['system'] = 'S12/DEFAULT'
    # df_large_lsched = create_experiment_data(exp_static_large_lachesis_scheduler)
    # df_large_lsched['system'] = 'S20/LSCHED'
    # df_large_default = create_experiment_data(exp_static_large_default_scheduler)
    # df_large_default['system'] = 'S20/DEFAULT'
    # df = pd.concat([df_lachesis_full, df_lachesis_default, df_small_lsched, df_small_default, df_medium_lsched, df_medium_default, df_large_lsched, df_large_default])

    # plot_slo_violation_summary(df)
    # plot_cold_start_slo_summary(df)
    # plot_idle_cores_summary(df)
    # plot_assigned_cpu_limit_timeline(df_lachesis)
    # plot_assigned_mem_limit_timeline(df)
    # plot_slack_timeline(df_lachesis)
    # plot_invoker_aggreagte_cpu_limit_usage_timeline(df_lachesis)
    # plot_slo_violation_breakdown(df_lachesis)
    # plot_per_invoker_data(df_large)
    # plot_cpu_util_timeline()
    # plot_invoker_aggregate_cpu_limit_timeline(df_lachesis)
    # plot_core_breakdown(df_lachesis)
    # plot_cold_start_percentages(exp)
    # plot_idle_core_cdf(df)
    explore_function(df, 'lrtrain')
    # explore_invoker(df, 'w3')

    # plot_slo_violation_summary_system_comparison(df)
    # plot_cold_start_slo_summary_system_comparison(df)
    # plot_idle_cores_summary_system_comparison(df)
    # plot_cold_start_percentages_system_comparison(df)
    # plot_idle_core_cdf_system_comparison(df)

'''
Lachesis Experiment Functions 
'''
def register_functions():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        # Image Process
        function_metadata = ['docker:psinha25/main-python']
        parameters = ['endpoint:\"10.52.3.142:9002\"', 'access_key:\"testkey\"', 'secret_key:\"testsecret\"', 'bucket:\"openwhisk\"']
        response = stub.Register(lachesis_pb2.RegisterRequest(function='imageprocess', 
                                                            function_path='~/lachesis/benchmarks/functions/image-processing', 
                                                            function_metadata=function_metadata, 
                                                            parameters=parameters))
        print(response)

        # Floatmatmult
        response = stub.Register(lachesis_pb2.RegisterRequest(function='floatmatmult',
                                                            function_path='~/lachesis/benchmarks/functions/matmult',
                                                            function_metadata=function_metadata,
                                                            parameters=parameters))
        print(response)

        # Video Process
        function_metadata = ['docker:psinha25/video-ow']
        response = stub.Register(lachesis_pb2.RegisterRequest(function='videoprocess',
                                                            function_path='~/lachesis/benchmarks/functions/video-processing',
                                                            function_metadata=function_metadata,
                                                            parameters=parameters))
        print(response)

        # Mobiletnet
        function_metadata = ['docker:psinha25/mobilenet-ow']
        response = stub.Register(lachesis_pb2.RegisterRequest(function='mobilenet',
                                                            function_path='~/lachesis/benchmarks/functions/mobilenet',
                                                            function_metadata=function_metadata,
                                                            parameters=parameters))
        print(response)

        # Sentiment
        function_metadata = ['docker:psinha25/sentiment-ow']
        response = stub.Register(lachesis_pb2.RegisterRequest(function='sentiment',
                                                            function_path='~/lachesis/benchmarks/functions/sentiment',
                                                            function_metadata=function_metadata,
                                                            parameters=parameters))
        print(response)

        # Linpack
        function_metadata = ['docker:psinha25/main-python']
        response = stub.Register(lachesis_pb2.RegisterRequest(function='linpack',
                                                            function_path='~/lachesis/benchmarks/functions/linpack',
                                                            function_metadata=function_metadata,
                                                            parameters=parameters))
        print(response)

        # Encryption
        function_metadata = ['docker:psinha25/main-python']
        response = stub.Register(lachesis_pb2.RegisterRequest(function='encrypt',
                                                            function_path='~/lachesis/benchmarks/functions/encryption',
                                                            function_metadata=function_metadata,
                                                            parameters=parameters))
        print(response)

        # Logistic Regression Training
        function_metadata = ['docker:psinha25/lr-train-ow']
        response = stub.Register(lachesis_pb2.RegisterRequest(function='lrtrain',
                                                            function_path='~/lachesis/benchmarks/functions/logistic-regression-training',
                                                            function_metadata=function_metadata,
                                                            parameters=parameters))
        print(response)

def test_invocations():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        # matmult_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='floatmatmult', slo=15000, parameters=['matrix1_4000_0.7.txt', 'matrix2_4000_0.7.txt']))
        # image_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='imageprocess', slo=4000, parameters=[feature_dict['imageprocess'].iloc[4]['file_name']]))
        # video_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='videoprocess', slo=10000, parameters=[feature_dict['videoprocess'].iloc[4]['file_name']]))
        # sentiment_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='sentiment', slo=10000, parameters=[feature_dict['sentiment'].iloc[4]['file_name']]))
        # linpack_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='linpack', slo=10000, parameters=['5000']))
        # lrtrain_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='lrtrain', slo=10000, parameters=[feature_dict['lrtrain'].iloc[2]['file_name']]))
        # mobilenet_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='mobilenet', slo=10000, parameters=[feature_dict['mobilenet'].iloc[4]['file_name']]))
        encrypt_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='encrypt', slo=150.75, parameters=['10000', '30']))

        # matmult_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='floatmatmult', slo=15000, parameters=['matrix1_4000_0.7.txt', 'matrix2_4000_0.7.txt']))
        # image_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='imageprocess', slo=4000, parameters=[feature_dict['imageprocess'].iloc[4]['file_name']]))
        # video_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='videoprocess', slo=10000, parameters=[feature_dict['videoprocess'].iloc[4]['file_name']]))
        # sentiment_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='sentiment', slo=10000, parameters=[feature_dict['sentiment'].iloc[4]['file_name']]))
        # linpack_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='linpack', slo=10000, parameters=['5000']))
        # lrtrain_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='lrtrain', slo=10000, parameters=[feature_dict['lrtrain'].iloc[2]['file_name']]))
        # mobilenet_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='mobilenet', slo=10000, parameters=[feature_dict['mobilenet'].iloc[4]['file_name']]))
        # encrypt_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='encrypt', slo=150.75, parameters=['50', '10']))

def launch_slo_invocations():

    for function in feature_dict:
        df = feature_dict[function]
        for file_name in df['file_name'].unique():
            if (function != 'lrtrain') and (function != 'resnet50') and (function != 'floatmatmult') and (function != 'imageprocess'):
                for cpu in range(1, 33):
                    fxn_invocation_command = None
                    if (function == 'floatmatmult'):
                        fxn_invocation_command = f'wsk -i action invoke {function}_{cpu} \
                                                  --param input1 {file_name} \
                                                  --param input2 {file_name} \
                                                  --param cpu {cpu} \
                                                  -r -v\n'
                    else:
                        fxn_invocation_command = f'wsk -i action invoke {function}_{cpu} \
                                                   --param input1 {file_name} \
                                                   --param cpu {cpu} \
                                                   -r -v\n'

                    tmp = subprocess.Popen(fxn_invocation_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    fxn_inv_out, fxn_inv_err = tmp.communicate()
                    fxn_inv_out = fxn_inv_out.decode()
                    fxn_inv_err = fxn_inv_err.decode()
            elif function == 'lrtrain':
                for cpu in range (20, 33):
                    fxn_invocation_command = f'wsk -i action invoke {function}_{cpu} \
                                               --param input1 {file_name} \
                                               --param cpu {cpu} \
                                               -r -v\n'
                    tmp = subprocess.Popen(fxn_invocation_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    fxn_inv_out, fxn_inv_err = tmp.communicate()
                    fxn_inv_out = fxn_inv_out.decode()
                    fxn_inv_err = fxn_inv_err.decode()


    # Encrypt - no input file
    for length in [500, 1000, 5000, 10000, 35000, 50000]:
        for iteration in [10, 25, 30, 50, 75, 100]:
            for cpu in range(1, 33):
                fxn_invocation_command = f'wsk -i action invoke encrypt_{cpu} \
                                           --param input1 {length} \
                                           --param input2 {iteration} \
                                           --param cpu {cpu}\
                                           -r -v\n'
                tmp = subprocess.Popen(fxn_invocation_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                fxn_inv_out, fxn_inv_err = tmp.communicate()
                fxn_inv_out = fxn_inv_out.decode()
                fxn_inv_err = fxn_inv_err.decode()


    # Linpack - no input file
    for size in [500, 1000, 2000, 3500, 5000, 7500, 10000]:
        for cpu in range(1, 33):
            fxn_invocation_command = f'wsk -i action invoke linpack_{cpu} \
                                       --param input1 {size} \
                                       --param cpu {cpu} \
                                       -r -v\n'
            tmp = subprocess.Popen(fxn_invocation_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            fxn_inv_out, fxn_inv_err = tmp.communicate()
            fxn_inv_out = fxn_inv_out.decode()
            fxn_inv_err = fxn_inv_err.decode()

def obtain_input_duration(quantile=0.5):
    df_0 = pd.read_csv('../data/slos/invoker_0_slos.csv')
    df_1 = pd.read_csv('../data/slos/invoker_1_slos.csv')
    df_2 = pd.read_csv('../data/slos/invoker_2_slos.csv')

    df = pd.concat([df_0, df_1, df_2], axis=0)
    
    for fxn in functions:
        df_fxn = df[df['function'].str.contains(fxn, case=False, na=False)]
        # Normal case for all other functions
        if (fxn != 'encrypt') and (fxn != 'linpack'):
            input_durs = {}
            for i in df_fxn['inputs'].unique():
                df_sub = df_fxn[df_fxn['inputs'] == i]
                input_value = str(json.loads(i)[0])[1:-1]
                duration_value = df_sub['duration'].quantile(quantile)
                input_durs[input_value] = duration_value
            df_inputs = pd.read_csv(f'../data/vw-prediction-inputs/{fxn}-inputs.csv')
            df_inputs['duration'] = df_inputs['file_name'].map(input_durs)
            df_inputs.to_csv(f'../data/vw-prediction-inputs/{fxn}-inputs.csv', index=False)
        # Special case for encrypt   
        elif fxn == 'encrypt':
            first_inputs = []
            second_inputs = []
            durations = []
            for i in df_fxn['inputs'].unique():
                df_sub = df_fxn[df_fxn['inputs'] == i]
                first_inputs.append(str(json.loads(i)[0]))
                second_inputs.append(str(json.loads(i)[1]))
                durations.append(df_sub['duration'].quantile(quantile))
            df_inputs = pd.DataFrame()
            df_inputs['length'] = first_inputs
            df_inputs['iterations'] = second_inputs
            df_inputs['duration'] = durations
            df_inputs.to_csv('../data/vw-prediction-inputs/encrypt-inputs.csv', index=False)
        # Special case for linpack
        elif fxn == 'linpack':
            sizes = []
            durations = []
            for i in df_fxn['inputs'].unique():
                df_sub = df_fxn[(df_fxn['inputs'] == i)]
                sizes.append(str(json.loads(i)[0]))
                durations.append(df_sub['duration'].quantile(quantile))
            df_inputs = pd.DataFrame()
            df_inputs['matrix_size'] = sizes
            df_inputs['duration'] = durations

            matrix_sizes = df_inputs['matrix_size'].values.reshape(-1, 1)
            durations = df_inputs['duration'].values

            model = LinearRegression()
            model.fit(matrix_sizes, durations)

            new_matrix_sizes = [250, 750, 1500, 2500, 3000, 4000, 4500, 5500, 6000, 6500, 7000, 8000, 8500, 9000, 9500]
            new_durations = model.predict(np.array(new_matrix_sizes).reshape(-1, 1))

            new_data = pd.DataFrame({'matrix_size': new_matrix_sizes, 'duration': new_durations})
            df_inputs = pd.concat([df_inputs, new_data], ignore_index=True)
            df_inputs.to_csv('../data/vw-prediction-inputs/linpack-inputs.csv', index=False)

def run_experiment():
    
    function_counters = {func: 0 for func in functions}

    # Calculate total number of requests (RPS = 2 for 10 minutes)
    request_duration = 10 * 60
    total_requests = 2 * request_duration

    
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        start_time = time.time()
        for _ in range(total_requests):
            
            # Randomly select a function and get a row from the dataframe
            selected_function = random.choice(functions)
            current_counter = function_counters[selected_function]
            df = feature_dict[selected_function]
            selected_row = df.iloc[current_counter]

            # Increment function row counter
            function_counters[selected_function] = (function_counters[selected_function] + 1) % len(df)

            # Construct parameter list
            parameter_list = []
            if selected_function == 'linpack':
                parameter_list.append(str(selected_row['matrix_size']))
            elif selected_function == 'floatmatmult':
                parameter_list.append(selected_row['file_name'])
                parameter_list.append(selected_row['file_name'])
            elif selected_function == 'encrypt':
                parameter_list.append(str(selected_row['length']))
                parameter_list.append(str(selected_row['iterations']))
            else:
                parameter_list.append(selected_row['file_name'])
            slo = float(selected_row['duration']) * (1 + SLO_MULTIPLIER)
            
            # Make gRPC invocation request
            response = stub.Invoke(lachesis_pb2.InvokeRequest(function=selected_function, slo=slo, parameters=parameter_list))
            print(f'Response for function {selected_function}: {response}')

            # Control the request rate to achieve 2 requests per second
            elapsed_time = time.time() - start_time
            time.sleep(max(0, 0.5 - elapsed_time % 0.5))

def test_linpack():
    df = feature_dict['linpack']
    print(df)

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        start_time = time.time()
        for j in range (0, 10):
            for i in range(0, len(df)):
                selected_row = df.iloc[i]
                parameter_list = [str(selected_row['matrix_size'])]
                slo = float(selected_row['duration']) * (1 + SLO_MULTIPLIER)

                response = stub.Invoke(lachesis_pb2.InvokeRequest(function='linpack', slo=slo, parameters=parameter_list))
                print(f'Response for function linpack: {response}')

                # Control the request rate to achieve 2 requests per second
                elapsed_time = time.time() - start_time
                time.sleep(max(0, 1 - elapsed_time % 1))
        print('Completed linpack invocations')

        # wsk -i action invoke linpack_13_128 --param input1 8000

def test_floatmatmult():
    df = feature_dict['floatmatmult']

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)
        
        start_time = time.time()
        for j in range(0, 10):
            for i in range(0, len(df)):
                selected_row = df.iloc[i]
                parameter_list = []
                parameter_list.append(selected_row['file_name'])
                parameter_list.append(selected_row['file_name'])
                slo = float(selected_row['duration']) * (1 + SLO_MULTIPLIER)

                response = stub.Invoke(lachesis_pb2.InvokeRequest(function='floatmatmult', slo=slo, parameters=parameter_list))
                print(f'Resposne for function floatmatmult: {response}')
                # Control the request rate to achieve 2 requests per second
                elapsed_time = time.time() - start_time
                time.sleep(max(0, 1 - elapsed_time % 1))

def test_image_process():
    df = feature_dict['imageprocess']

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        start_time = time.time()
        for i in range(0, len(df)):
            selected_row = df.iloc[i]
            parameter_list = [str(selected_row['file_name'])]
            slo = int(selected_row['slo'])

            response = stub.Invoke(lachesis_pb2.InvokeRequest(function='imageprocess', slo=slo, parameters=parameter_list))
            print(f'Response for function imageprocess: {response}')

            # Control the request rate to achieve 2 requests per second
            elapsed_time = time.time() - start_time
            time.sleep(max(0, 0.5 - elapsed_time % 0.5))

def test_sentiment():
    df = feature_dict['sentiment']
    print(df)

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        start_time = time.time()
        for j in range(0, 20):
            for i in range(0, len(df)):
                selected_row = df.iloc[i]
                parameter_list = [str(selected_row['file_name'])]
                slo = int(selected_row['slo'])

                response = stub.Invoke(lachesis_pb2.InvokeRequest(function='sentiment', slo=slo, parameters=parameter_list))
                print(f'Response for function sentiment: {response}')

                # Control the request rate to achieve 2 requests per second
                elapsed_time = time.time() - start_time
                time.sleep(max(0, 0.5 - elapsed_time % 0.5))
        print('Completed sentiment invocations')

def test_encrypt():
    df = feature_dict['encrypt']
    print(df)
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        start_time = time.time()
        for j in range (0,15):
            for i in range(0, len(df)):
                selected_row = df.iloc[i]
                parameter_list = [str(selected_row['length']), str(selected_row['iterations'])]
                slo = float(selected_row['duration']) * (1 + SLO_MULTIPLIER)
                
                response = stub.Invoke(lachesis_pb2.InvokeRequest(function='encrypt', slo=slo, parameters=parameter_list))
                print(f'Response for function encrypt: {response}')

                # Control the request rate to achieve 2 requests per second
                elapsed_time = time.time() - start_time
                time.sleep(max(0, 0.5 - elapsed_time % 0.5))
        print('Completed encrypt invocations')

'''
Cypress Experiment Functions
'''
def run_cypress_experiments():
    function_counters = {func: 0 for func in functions}

    # Calculate total number of requests (RPS = 2 for 10 minutes)
    request_duration = 10 * 60
    total_requests = 2 * request_duration

    
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = cypress_pb2_grpc.CypressStub(channel)

        start_time = time.time()
        for _ in range(total_requests):
            
            # Randomly select a function and get a row from the dataframe
            selected_function = random.choice(functions)
            current_counter = function_counters[selected_function]
            df = feature_dict[selected_function]
            selected_row = df.iloc[current_counter]

            # Increment function row counter
            function_counters[selected_function] = (function_counters[selected_function] + 1) % len(df)

            # Construct parameter list
            parameter_list = []
            if selected_function == 'linpack':
                parameter_list.append(str(selected_row['matrix_size']))
            elif selected_function == 'floatmatmult':
                parameter_list.append(selected_row['file_name'])
                parameter_list.append(selected_row['file_name'])
            elif selected_function == 'encrypt':
                parameter_list.append(str(selected_row['length']))
                parameter_list.append(str(selected_row['iterations']))
            else:
                parameter_list.append(selected_row['file_name'])
            slo = float(selected_row['duration']) * (1 + SLO_MULTIPLIER)
            
            # Make gRPC invocation request
            response = stub.Invoke(cypress_pb2.InvokeRequest(function=selected_function, slo=slo, parameters=parameter_list))
            print(f'Response for function {selected_function}: {response}')

            # Control the request rate to achieve 2 requests per second
            elapsed_time = time.time() - start_time
            time.sleep(max(0, 0.5 - elapsed_time % 0.5))

def test_cypress_linpack():
    df = cypress_feature_dict['linpack']

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = cypress_pb2_grpc.CypressStub(channel)

        start_time = time.time()
        for j in range (0, 10):
            for i in range(0, len(df)):
                selected_row = df.iloc[i]
                parameter_list = [str(selected_row['matrix_size'])]
                slo = selected_row['SLO']
                batch_size = selected_row['batch_size']

                response = stub.Invoke(cypress_pb2.InvokeRequest(function='linpack', slo=slo, parameters=parameter_list, batch_size=batch_size))
                print(f'Response for function linpack: {response}')

                # Control the request rate to achieve 2 requests per second
                elapsed_time = time.time() - start_time
                time.sleep(max(0, 2 - elapsed_time % 2))
        print('Completed linpack invocations')

def test_cypress_imageprocess():
    df = cypress_feature_dict['imageprocess']

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = cypress_pb2_grpc.CypressStub(channel)

        start_time = time.time()
        for j in range (0, 10):
            for i in range(0, len(df)):
                selected_row = df.iloc[i]
                parameter_list = [str(selected_row['file_name'])]
                slo = selected_row['SLO']
                batch_size = selected_row['batch_size']

                response = stub.Invoke(cypress_pb2.InvokeRequest(function='linpack', slo=slo, parameters=parameter_list, batch_size=batch_size))
                print(f'Response for function linpack: {response}')

                # Control the request rate to achieve 2 requests per second
                elapsed_time = time.time() - start_time
                time.sleep(max(0, 0.5 - elapsed_time % 0.5))
        print('Completed linpack invocations')

if __name__=='__main__':
    # register_functions()
    # test_invocations()
    # launch_slo_invocations()
    # obtain_input_duration(quantile=0.5)
    # run_experiment()
    # test_linpack()
    # test_floatmatmult()
    # test_sentiment()
    # test_image_process()
    # test_encrypt()
    # plot_results()
    # test_cypress_linpack()
    test_cypress_imageprocess()
    # run_cypress_experiments()