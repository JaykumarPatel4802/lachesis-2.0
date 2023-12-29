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

SLO_MULTIPLIER = 0.4 # originally 0.4

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
    if (exp.split('-')[0] == 'aquatope') or (exp.split('-')[0] == 'cypress'):
        df['total_duration'] = df['duration']
    else:
        df['total_duration'] = df['duration'] + df['cold_start_latency']
    
    
    if ('aquatope-lachesis-scheduler' in exp):
        df['slo_violation'] = df.apply(lambda row: 1 if row['total_duration'] > (row['slo'] / 1.12) else 0, axis=1)
    else:
        df['slo_violation'] = df.apply(lambda row: 1 if (row['total_duration'] > row['slo']) or (row['duration'] < 0) else 0, axis=1)
    df['no_cores'] = np.ceil(df['max_cpu']).astype(int)
    df['idle_cores'] = df['cpu_limit'] - df['no_cores']
    df['idle_cores'] = df['idle_cores'].apply(lambda x: max(0, x))
    df['cpu_utilization'] = df['no_cores'] / df['cpu_limit'] * 100
    if ('cypress' in exp):
        df['idle_mem'] = df['mem_limit'] - df['max_mem']
        df['mem_utilization'] = df['max_mem'] / df['mem_limit'] * 100
    else:
        df['idle_mem'] = df['scheduled_mem'] - df['max_mem']
        df['mem_utilization'] = df['max_mem'] / df['scheduled_mem'] * 100
    df['idle_mem'] = df['idle_mem'].apply(lambda x: max(0, x))
    df['slack'] = df['slo'] - df['total_duration']
    if (exp.split('-')[0] == 'lachesis') and ('mem' not in exp) and ('one-hot' not in exp):
        df = df[df['total_duration'] < 400000]
    
    # if (exp.split('-')[0] == 'lachesis') or (exp.split('-')[0] == 'aquatope') or (exp.split('-')[0] == 'parrotfish'):
    #     print(f'here:')
    #     df = df[df['total_duration'] < 500000]
    # Get a summary per function - number of runs per function, number of completions per function
    print(exp)
    print('----------------------------------')
    print('Summary of Executions per Function')
    for fxn in df['function'].unique():
        df_fxn = df[df['function'] == fxn]
        no_rows = len(df_fxn)
        no_unfinished_rows = len(df_fxn[df_fxn['start_time'] == 'NA'])
        no_finished_rows = no_rows - no_unfinished_rows
        print(f'{fxn}: {no_finished_rows} finished / {no_rows} total')
    print(f'Total invocations: {len(df)}')
    # df = df[df['lachesis_end'] != 'NA']
    print(f'Total finished: {len(df)}')
    print()
    df['activation_id'] = df['activation_id'].str.strip('"')
    return(df)

def create_experiment_data_cypress(exp):
    conn = sqlite3.connect('./datastore/cypress-controller.db')
    df = pd.read_sql_query('SELECT * FROM fxn_exec_data', conn)
    df = df[df['exp_no'] == f'{exp}']
    df['slo_violation'] = df.apply(lambda row: 1 if row['duration'] > row['slo'] else 0, axis=1)
    df['no_cores'] = np.ceil(df['max_cpu']).astype(int)
    df['idle_cores'] = df['cpu_limit'] - df['no_cores']
    df['idle_cores'] = df['idle_cores'].apply(lambda x: max(0, x))
    df['slack'] = df['slo'] - df['duration']

    # Get a summary per function - number of runs, number of completions per function
    print('Summary of Executions per Function')
    for fxn in df['function'].unique():
        df_fxn = df[df['function'] == fxn]
        no_rows = len(df_fxn)
        no_unfinished_rows = len(df_fxn[df_fxn['start_time'] == 'NA'])
        no_finished_rows = no_rows - no_unfinished_rows
        print(f'{fxn}: {no_finished_rows} finished / {no_rows} total')
    print(f'Total invocations: {len(df)}')
    df = df[df['start_time'] != 'NA']
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
    plt.savefig(f'../study-full-experiments/summary-slo-violations', bbox_inches='tight', dpi=300)
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
    plt.savefig(f'../study-full-experiments/summary-slo-violations-system-comparison.png', bbox_inches='tight', dpi=300)
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
    plt.savefig(f'../study-full-experiments/summary-violation-cold-starts', bbox_inches='tight', dpi=300)
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
    plt.savefig(f'../study-full-experiments/summary-violation-cold-starts-system-comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_idle_cores_summary(df):
    df = df[df['max_cpu'] > 0]
    order = functions
    ax = sns.barplot(x='function', y='idle_cores', data=df, order=order)
    plt.xlabel('Function')
    plt.ylabel('Num. Idle Cores')
    plt.xticks(rotation=45)
    ax.set_ylim(0, 32)
    plt.savefig(f'../study-full-experiments/summary-idle-cores', bbox_inches='tight', dpi=300)
    plt.close()

def plot_idle_mem_summary(df):
    order = functions
    ax = sns.boxplot(x='function', y='idle_mem', data=df, order=order)
    plt.xlabel('Function')
    plt.ylabel('Unused Memory (MB)')
    plt.xticks(rotation=45)
    # ax.set_ylim(0, 32)
    plt.savefig(f'../study-full-experiments/summary-idle-mem-scheduled-limit.png', bbox_inches='tight', dpi=300)
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
    plt.savefig(f'../study-full-experiments/summary-idle-cores-system-comparison.png', bbox_inches='tight', dpi=300)
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
        plt.savefig(f'../study-full-experiments/cpu-limit-timeline-{fxn}.png', bbox_inches='tight', dpi=300)
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
        
        plt.savefig(f'../study-full-experiments/mem-limit-usage-timeline-{fxn}.png', bbox_inches='tight', dpi=300)
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
        plt.savefig(f'../study-full-experiments/slack-timeline-{fxn}.png', bbox_inches='tight', dpi=300)
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
        plt.savefig(f'../study-full-experiments/summed-cpu-limits-invoker-{name}-timeline.png', bbox_inches='tight', dpi=300)
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
        plt.savefig(f'../study-full-experiments/summed-cpu-limits-invoker-{name}-timeline.png', bbox_inches='tight', dpi=300)
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
    plt.savefig(f'../study-full-experiments/slack-breakdown.png', bbox_inches='tight', dpi=300)
    plt.close()
    # print(df_plot[['duration', 'slack', 'cold_start_latency', 'wait_time', 'activation_id']].to_string())

def plot_per_invoker_data(df, rps):
    
    # Plot breakdown of invocations per invoker per function
    df = df.sort_values(by=['invoker_name', 'function'])
    plt.figure(figsize=(16, 6))
    sns.set(style="whitegrid")
    sns.countplot(x="invoker_name", hue="function", data=df, order=['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8'])
    plt.xlabel("Invoker Name")
    plt.ylabel("# of Invocations")
    plt.savefig(f'../study-full-experiments/function-breakdown-per-invoker.png', bbox_inches='tight', dpi=300)
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
    
    plt.savefig(f'../study-full-experiments/new-scheduler/slo-violation-breakdown-per-invoker-{rps}.png', bbox_inches='tight', dpi=300)
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
            plt.savefig(f'../study-full-experiments/invoker-{invoker}-cpu-util-timeline.png', bbox_inches='tight', dpi=300)
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
        plt.savefig(f'../study-full-experiments/core-assigned-{function}-scheduled.png', bbox_inches='tight', dpi=300)
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

    plt.savefig(f'../study-full-experiments/scheduler-cold-start.png', bbox_inches='tight', dpi=300)
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

    plt.savefig(f'../study-full-experiments/scheduler-cold-start-system-comparison.png', bbox_inches='tight', dpi=300)
    plt.close() 

def plot_idle_core_cdf(df):
    # Remove columns where prediction != actual limit -- these were initial runs
    df = df[df['cpu_limit'] == df['predicted_cpu']]
    plt.figure(figsize=(14,6))
    sns.ecdfplot(data=df, x='idle_cores')
    plt.xlabel('Number of Idle Cores')
    plt.ylabel('Cumulative Distribution')
    plt.xticks(rotation=45)

    plt.savefig(f'../study-full-experiments/idle-cores-cdf.png', bbox_inches='tight', dpi=300)
    plt.close() 

def plot_idle_mem_cdf(df):
    # Remove columns where prediction != actual limit -- these were initial runs
    df = df[df['mem_limit'] == df['predicted_mem']]
    plt.figure(figsize=(14,6))
    sns.ecdfplot(data=df, x='idle_mem')
    plt.xlabel('Unused Memory (MB)')
    plt.ylabel('Cumulative Distribution')
    plt.xticks(rotation=45)

    plt.savefig(f'../study-full-experiments/idle-mem-cdf.png', bbox_inches='tight', dpi=300)

def plot_idle_core_cdf_system_comparison(df):
    plt.figure(figsize=(14,6))
    hue_order = ['LRA/LSCHED', 'LRA/DEFAULT', 'S4/LSCHED', 'S4/DEFAULT', 'S12/LSCHED', 'S12/DEFAULT', 'S20/LSCHED', 'S20/DEFAULT']
    sns.ecdfplot(data=df, x='idle_cores', hue='system', hue_order=hue_order)

    plt.xlabel('Number of Idle Cores')
    plt.ylabel('Cumulative Distribution')
    plt.xticks(rotation=45)
    # plt.legend(ncol=2)

    plt.savefig(f'../study-full-experiments/idle-cold-start-system-comparison.png', bbox_inches='tight', dpi=300)
    plt.close() 

def explore_function(df, function):
    df = df[df['function'] == function]
    df = df[df['max_mem'] > 0]
    print(df[['inputs', 'slo_violation', 'slo', 'duration', 'cold_start_latency', 'activation_id', 'invoker_name', 'idle_cores', 'cpu_limit', 'mem_limit', 'scheduled_cpu', 'scheduled_mem', 'p90_cpu', 'p95_cpu', 'p99_cpu', 'max_cpu', 'max_mem']].to_string())

def explore_invoker(df, invoker_name):
    df = df[df['invoker_name'] == invoker_name]
    print(df)

def plot_rps_slo_violation_system_comparison(df):
    violation_summary_dict = {}
    for system in df['system'].unique():
        df_system = df[df['system'] == system]
        for rps in df_system['rps'].unique():
            df_rps = df_system[df_system['rps'] == rps]
            no_rows = len(df_rps)
            no_violations = len(df_rps[df_rps['slo_violation'] == 1])
            violation_summary_dict[(rps, system)] = [no_violations, no_rows, no_violations/no_rows * 100]
    df_summary = pd.DataFrame.from_dict(violation_summary_dict, orient='index', columns=['no_violations', 'no_rows', 'percentage'])
    df_summary.reset_index(inplace=True)
    df_summary.rename(columns={'index': 'rps-system'}, inplace=True)
    df_summary[['rps', 'system']] = pd.DataFrame(df_summary['rps-system'].tolist(), index=df_summary.index)

    fig, ax = plt.subplots(ncols=1, figsize=(10, 2.5))
    hue_order = ['small', 'medium', 'large', 'parrotfish', 'aquatope', 'cypress', 'lachesis']
    ax = sns.barplot(ax=ax, x='rps', y='percentage', data=df_summary, hue='system', hue_order=hue_order)
    ax.set_xlabel('RPS', fontsize=17)
    ax.set_ylabel('SLO Violation\nRatio (%)', fontsize=17, labelpad=2)
    ax.tick_params(axis='both', which='major', labelsize=15, colors='dimgray')
    plt.legend(ncol=4, fontsize=14, loc="upper left", bbox_to_anchor=(0.04, 1.05), frameon=False)
    ax.set_ylim(0, 105)
    plt.savefig(f'../study-full-experiments/summary-slo-violations-rps-system-comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_rps_function_breakdown_slo_violation_system_comparison(df):
    for rps in df['rps'].unique():
        df_rps = df[df['rps'] == rps]
        order = functions
        violation_summary_dict = {}
        for system in df_rps['system'].unique():
            df_system = df_rps[df_rps['system'] == system]
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
        hue_order = ['small', 'medium', 'large', 'parrotfish', 'aquatope', 'aquatope-lachesis-scheduler', 'cypress', 'cypress-lachesis-scheduler', 'lachesis']
        ax = sns.barplot(x='fxn', y='percentage', data=df_summary, hue='system', hue_order=hue_order, order=order)
        plt.xlabel('Function')
        plt.ylabel('Percentage of SLO Violations (%)')
        plt.title(f'RPS = {rps}')
        plt.legend(ncol=2)
        plt.xticks(rotation=45)
        ax.set_ylim(0, 110)

        plt.savefig(f'../study-full-experiments/rps-{rps}-slo-violation-function-breakdown-system-comparison.png', bbox_inches='tight', dpi=300)
        plt.close()

def plot_rps_idle_cores_system_comparison(df):
    plt.figure(figsize=(12, 6))
    hue_order = ['small', 'medium', 'large', 'parrotfish', 'aquatope', 'aquatope-lachesis-scheduler', 'cypress', 'cypress-lachesis-scheduler', 'lachesis']
    ax = sns.boxplot(x='rps', y='idle_cores', data=df, hue='system', hue_order=hue_order)
    plt.xlabel('RPS')
    plt.ylabel('Idle Cores')
    plt.legend(ncol=2)
    plt.xticks(rotation=45)
    ax.set_ylim(0, 25)

    plt.savefig(f'../study-full-experiments/summary-idle-cores-rps-system-comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_rps_idle_mem_system_comparison(df):
    plt.figure(figsize=(12, 6))
    hue_order = ['small', 'medium', 'large', 'parrotfish', 'aquatope', 'aquatope-lachesis-scheduler', 'cypress', 'cypress-lachesis-scheduler', 'lachesis']
    ax = sns.boxplot(x='rps', y='idle_mem', data=df, hue='system', hue_order=hue_order)
    plt.xlabel('RPS')
    plt.ylabel('Idle Mem (MB)')
    plt.legend(ncol=2)
    plt.xticks(rotation=45)
    ax.set_ylim(0, 6000)

    plt.savefig(f'../study-full-experiments/summary-idle-mem-rps-system-comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_slo_violation_ans(df):
    # df = df[(df['mem_utilization'] >= 0) & (df['cpu_utilization'] >= 0)]
    violation_summary_dict = {}
    for system in df['system'].unique():
        df_system = df[df['system'] == system]
        for rps in df_system['rps'].unique():
            df_rps = df_system[df_system['rps'] == rps]
            min_cores = df_rps['idle_cores'].min()
            median_cores = df_rps['idle_cores'].quantile(0.5)
            p95_cores = df_rps['idle_cores'].quantile(0.95)
            max_cores = df_rps['idle_cores'].max()

            min_mem = df_rps['idle_mem'].min()
            median_mem = df_rps['idle_mem'].quantile(0.5)
            p95_mem = df_rps['idle_mem'].quantile(0.95)
            max_mem = df_rps['idle_mem'].max()

            if ((system == 'ALAP') or (system == 'Aquatope')) and rps == 6:
                p75_mem = df_rps['idle_mem'].quantile(0.75)
                p80_mem = df_rps['idle_mem'].quantile(0.80)
                p85_mem = df_rps['idle_mem'].quantile(0.85)
                p90_mem = df_rps['idle_mem'].quantile(0.90)
                print(f'p75: {p75_mem}')
                print(f'p80: {p80_mem}')
                print(f'p85: {p85_mem}')
                print(f'p90: {p90_mem}')
            

            no_rows = len(df_rps)
            no_violations = len(df_rps[df_rps['slo_violation'] == 1])
            violation_summary_dict[(rps, system)] = [no_violations, no_rows, no_violations/no_rows * 100, min_cores, median_cores, p95_cores, max_cores, min_mem, median_mem, p95_mem, max_mem]
    df_summary = pd.DataFrame.from_dict(violation_summary_dict, orient='index', columns=['no_violations', 'no_rows', 'percentage', 'min_cores', 'median_cores', 'p95_cores', 'max_cores', 'min_mem', 'median_mem', 'p95_mem', 'max_mem'])
    df_summary.reset_index(inplace=True)
    df_summary.rename(columns={'index': 'rps-system'}, inplace=True)
    df_summary[['rps', 'system']] = pd.DataFrame(df_summary['rps-system'].tolist(), index=df_summary.index)
    
    # Assuming your original DataFrame is named df
    data = {
        'rps-system': ['2-prototype', '3-prototype', '4-prototype', '5-prototype', '6-prototype'],
        'no_violations': [0, 0, 0, 0, 0],
        'no_rows': [0, 0, 0, 0, 0],
        'percentage': [7.2, 10.2, 13, 20.67, 33.52],
        'min_cores': [0, 0, 0, 0, 0],
        'median_cores': [0, 0, 0, 0, 0],
        'p95_cores': [0, 0, 0, 0, 0],
        'max_cores': [0, 0, 0, 0, 0],
        'min_mem': [0, 0, 0, 0, 0],
        'median_mem': [0, 0, 0, 0, 0],
        'p95_mem': [0, 0, 0, 0, 0],
        'max_mem': [0, 0, 0, 0, 0],
        'rps': [2, 3, 4, 5, 6],
        'system': ['Proto', 'Proto', 'Proto', 'Proto', 'Proto']
    }
        
    new_rows_df = pd.DataFrame(data)

    df_summary = pd.concat([df_summary, new_rows_df], axis=0)
    print(df_summary.to_string())


    fig, axs = plt.subplots(1, 1, figsize=(10, 2))
    # # hue_order = ['Medium', 'Large', 'Parrotfish', 'Aquatope', 'Cypress', 'ALAP']
    hue_order = ['Medium', 'Large', 'ALAP', 'Proto']
    final_palette = ['#12e193', '#069af3', '#ad03de', '#ffab0f']
    p0 = sns.barplot(ax=axs, x='rps', y='percentage', data=df_summary, hue='system', hue_order=hue_order, palette=final_palette)

    # plt.subplots_adjust(hspace=0.46)

    axs.set_xlabel('RPS', fontsize=17)
    axs.set_ylabel('% SLO Viol.', fontsize=17, labelpad=2)

    axs.tick_params(axis='both', which='major', labelsize=15)
    axs.locator_params(axis='y', nbins=5)
    
    plt.legend(ncol=4, fontsize=16, loc="upper left", bbox_to_anchor=(0., 1.27), frameon=False)
    plt.savefig(f'../study-full-experiments/new-scheduler/ans-final-results.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_rps_e2e(df):
    # df = df[(df['mem_utilization'] >= 0) & (df['cpu_utilization'] >= 0)]
    violation_summary_dict = {}
    for system in df['system'].unique():
        df_system = df[df['system'] == system]
        for rps in df_system['rps'].unique():
            df_rps = df_system[df_system['rps'] == rps]
            min_cores = df_rps['idle_cores'].min()
            median_cores = df_rps['idle_cores'].quantile(0.5)
            p95_cores = df_rps['idle_cores'].quantile(0.95)
            max_cores = df_rps['idle_cores'].max()

            min_mem = df_rps['idle_mem'].min()
            median_mem = df_rps['idle_mem'].quantile(0.5)
            p95_mem = df_rps['idle_mem'].quantile(0.95)
            max_mem = df_rps['idle_mem'].max()

            if ((system == 'ALAP') or (system == 'Aquatope')) and rps == 6:
                p75_mem = df_rps['idle_mem'].quantile(0.75)
                p80_mem = df_rps['idle_mem'].quantile(0.80)
                p85_mem = df_rps['idle_mem'].quantile(0.85)
                p90_mem = df_rps['idle_mem'].quantile(0.90)
                print(f'p75: {p75_mem}')
                print(f'p80: {p80_mem}')
                print(f'p85: {p85_mem}')
                print(f'p90: {p90_mem}')
            

            no_rows = len(df_rps)
            no_violations = len(df_rps[df_rps['slo_violation'] == 1])
            violation_summary_dict[(rps, system)] = [no_violations, no_rows, no_violations/no_rows * 100, min_cores, median_cores, p95_cores, max_cores, min_mem, median_mem, p95_mem, max_mem]
    df_summary = pd.DataFrame.from_dict(violation_summary_dict, orient='index', columns=['no_violations', 'no_rows', 'percentage', 'min_cores', 'median_cores', 'p95_cores', 'max_cores', 'min_mem', 'median_mem', 'p95_mem', 'max_mem'])
    df_summary.reset_index(inplace=True)
    df_summary.rename(columns={'index': 'rps-system'}, inplace=True)
    df_summary[['rps', 'system']] = pd.DataFrame(df_summary['rps-system'].tolist(), index=df_summary.index)

    print(df_summary.to_string())
    alap_values = df_summary[df_summary['system'] == 'ALAP'][['rps', 'percentage', 'min_cores', 'median_cores', 'p95_cores', 'max_cores', 'min_mem', 'median_mem', 'p95_mem', 'max_mem']]

    slo_reduction = []
    min_cores_reduction = []
    median_cores_reduction = []
    p95_cores_reduction = []
    max_cores_reduction = []
    min_mem_reduction = []
    median_mem_reduction = []
    p95_mem_reduction = []
    max_mem_reduction = []
    for system in df_summary['system'].unique():
        for rps in df_summary['rps'].unique():
            df_sub = df_summary[(df_summary['rps'] == rps) & (df_summary['system'] == system)]
            alap_subset = alap_values[alap_values['rps'] == rps]
            slo_reduction.append((df_sub['percentage'].values[0] - alap_subset['percentage'].values[0]) / df_sub['percentage'].values[0] * 100)

            min_cores_reduction.append((df_sub['min_cores'].values[0] - alap_subset['min_cores'].values[0]) / df_sub['min_cores'].values[0] * 100)
            median_cores_reduction.append((df_sub['median_cores'].values[0] - alap_subset['median_cores'].values[0]) / df_sub['median_cores'].values[0] * 100)
            p95_cores_reduction.append((df_sub['p95_cores'].values[0] - alap_subset['p95_cores'].values[0]) / df_sub['p95_cores'].values[0] * 100)
            max_cores_reduction.append((df_sub['max_cores'].values[0] - alap_subset['max_cores'].values[0]) / df_sub['max_cores'].values[0] * 100)

            min_mem_reduction.append((df_sub['min_mem'].values[0] - alap_subset['min_mem'].values[0]) / df_sub['min_mem'].values[0] * 100)
            median_mem_reduction.append((df_sub['median_mem'].values[0] - alap_subset['median_mem'].values[0]) / df_sub['median_mem'].values[0] * 100)
            p95_mem_reduction.append((df_sub['p95_mem'].values[0] - alap_subset['p95_mem'].values[0]) / df_sub['p95_mem'].values[0] * 100)
            max_mem_reduction.append((df_sub['max_mem'].values[0] - alap_subset['max_mem'].values[0]) / df_sub['max_mem'].values[0] * 100)

    df_summary['slo_reduction'] = slo_reduction
    df_summary['min_cores_reduction'] = min_cores_reduction
    df_summary['median_cores_reduction'] = median_cores_reduction
    df_summary['p95_cores_reduction'] = p95_cores_reduction
    df_summary['max_cores_reduction'] = max_cores_reduction
    df_summary['min_mem_reduction'] = min_mem_reduction
    df_summary['median_mem_reduction'] = median_mem_reduction
    df_summary['p95_mem_reduction'] = p95_mem_reduction
    df_summary['max_mem_reduction'] = max_mem_reduction

    print(df_summary[(df_summary['system'] == 'Parrotfish') | (df_summary['system'] == 'Aquatope') | (df_summary['system'] == 'Cypress')][['rps', 'system', 'slo_reduction', 'min_cores_reduction', 'median_cores_reduction', 'p95_cores_reduction', 'max_cores_reduction', 'min_mem_reduction', 'median_mem_reduction', 'p95_mem_reduction', 'max_mem_reduction']].to_string())
                
        


    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    # hue_order = ['Medium', 'Large', 'Parrotfish', 'Aquatope', 'Cypress', 'ALAP']
    hue_order = ['Medium', 'Large', 'ALAP']
    final_palette = ['#12e193', '#069af3', '#ad03de']
    p0 = sns.barplot(ax=axs[0], x='rps', y='percentage', data=df_summary, hue='system', hue_order=hue_order, palette=final_palette)
    p1 = sns.boxplot(ax=axs[1], x='rps', y='idle_cores', data=df, hue='system', hue_order=hue_order, showfliers=False, palette=final_palette, whis=[5, 95])
    p2 = sns.boxplot(ax=axs[2], x='rps', y='idle_mem', data=df, hue='system', hue_order=hue_order, showfliers=False, palette=final_palette, whis=[5, 95])
    # p3 = sns.barplot(ax=axs[3], x='rps', y='cpu_utilization', data=df, hue='system', hue_order=hue_order, errorbar='sd', estimator=np.mean, palette=['#fc5a50', '#ffab0f', '#fddc5c', '#12e193', '#069af3', '#ad03de'])
    # p4 = sns.barplot(ax=axs[4], x='rps', y='mem_utilization', data=df, hue='system', hue_order=hue_order, errorbar='sd', estimator=np.mean, palette=['#fc5a50', '#ffab0f', '#fddc5c', '#12e193', '#069af3', '#ad03de'])
    p0.get_legend().remove()
    p1.get_legend().remove()
    p2.get_legend().remove()
    # p3.get_legend().remove()
    # p4.get_legend().remove()

    plt.subplots_adjust(hspace=0.46)

    axs[0].set_xlabel('')
    axs[1].set_xlabel('')
    axs[2].set_xlabel('RPS', fontsize=17)
    # axs[3].set_xlabel('')

    axs[0].set_ylabel('% SLO Viol.', fontsize=17, labelpad=2)
    axs[1].set_ylabel('Allocated\nIdle vCPUs', fontsize=17, labelpad=3)
    axs[2].set_ylabel('Allocated\nIdle Mem (MB)', fontsize=17, labelpad=2)
    # axs[3].set_ylabel('Per Invocation\nvCPU Util (%)', fontsize=17, labelpad=2)
    # axs[4].set_ylabel('Per Invocation\nMem Util (%)', fontsize=17, labelpad=2)

    # axs[4].set_xlabel('RPS', fontsize=17)

    axs[0].text(0.5, -0.35, "(a)", size=17, ha="center",
            transform=axs[0].transAxes)
    axs[1].text(0.5, -0.35, "(b)", size=17, ha="center",
            transform=axs[1].transAxes)
    axs[2].text(0.5, -0.52, "(c)", size=17, ha="center",
            transform=axs[2].transAxes)
    # axs[3].text(0.5, -0.35, "(d)", size=17, ha="center",
    #         transform=axs[3].transAxes)
    # axs[4].text(0.5, -0.52, "(e)", size=17, ha="center",
    #         transform=axs[4].transAxes)

    axs[0].tick_params(axis='both', which='major', labelsize=15)
    axs[1].tick_params(axis='both', which='major', labelsize=15)
    axs[2].tick_params(axis='both', which='major', labelsize=15)
    # axs[3].tick_params(axis='both', which='major', labelsize=15)
    # axs[4].tick_params(axis='both', which='major', labelsize=15)

    axs[0].locator_params(axis='y', nbins=5)
    axs[1].locator_params(axis='y', nbins=5)
    axs[2].locator_params(axis='y', nbins=5)
    # axs[3].locator_params(axis='y', nbins=5)
    # axs[4].locator_params(axis='y', nbins=5)

    # axs[3].set_ylim(0, 140)
    # axs[4].set_ylim(0, 140)

    plt.legend(ncol=3, fontsize=16, loc="upper left", bbox_to_anchor=(0.16, 4.2), frameon=False)
    plt.savefig(f'../study-full-experiments/new-scheduler/summary-e2e-rps-system-comparison-ans.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_final_design_exploration(df):
    df = df[(df['mem_utilization'] >= 0) & (df['cpu_utilization'] >= 0)]
    violation_summary_dict = {}
    for system in df['system'].unique():
        df_system = df[df['system'] == system]
        for rps in df_system['rps'].unique():
            df_rps = df_system[df_system['rps'] == rps]
            p95_cores = df_rps['idle_cores'].quantile(0.95)
            median_cores = df_rps['idle_cores'].quantile(0.5)
            median_mem = df_rps['idle_mem'].quantile(0.5)
            p95_mem = df_rps['idle_mem'].quantile(0.95)
            no_rows = len(df_rps)
            no_violations = len(df_rps[df_rps['slo_violation'] == 1])
            violation_summary_dict[(rps, system)] = [no_violations, no_rows, no_violations/no_rows * 100, p95_cores, median_cores, median_mem, p95_mem]
    df_summary = pd.DataFrame.from_dict(violation_summary_dict, orient='index', columns=['no_violations', 'no_rows', 'percentage', 'p95_cores', 'median_cores', 'median_mem', 'p95_mem'])
    df_summary.reset_index(inplace=True)
    df_summary.rename(columns={'index': 'rps-system'}, inplace=True)
    df_summary[['rps', 'system']] = pd.DataFrame(df_summary['rps-system'].tolist(), index=df_summary.index)

    fig, axs = plt.subplots(1, 2, figsize=(9, 2))
    order0 = ['Proportional', 'Absolute']
    order1 = ['Packing', 'Hashing']
    p0 = sns.barplot(ax=axs[0], x='rps', y='percentage', hue='system', hue_order = order0, data=df_summary, palette=['#069af3', '#ad03de'])
    p1 = sns.barplot(ax=axs[1], x='rps', y='percentage', hue='system', hue_order = order1, data=df_summary, palette=['#069af3', '#ad03de'])

    axs[0].set_xlabel('RPS', fontsize=17)
    axs[1].set_xlabel('RPS', fontsize=17)

    axs[0].set_ylabel('% SLO Viol.', fontsize=17, labelpad=2)
    axs[1].set_ylabel('', fontsize=17, labelpad=1)

    axs[0].set_ylim((0, 50))
    axs[1].set_ylim((0, 50))

    axs[0].text(0.5, -0.5, "(a)", size=17, ha="center",
            transform=axs[0].transAxes)
    axs[1].text(0.5, -0.5, "(b)", size=17, ha="center",
            transform=axs[1].transAxes)
    
    axs[0].legend(ncol=1, fontsize=12, title='Cost Function', title_fontsize=12, loc="upper left", frameon=False)
    axs[1].legend(ncol=1, fontsize=12, title='Scheduler', title_fontsize=12, loc="upper left", frameon=False)
    
    axs[0].tick_params(axis='both', which='major', labelsize=15)
    axs[1].tick_params(axis='both', which='major', labelsize=15)

    plt.savefig(f'../study-full-experiments/new-scheduler/final-design-exploration.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_design_exploration(df):
    df = df[(df['mem_utilization'] >= 0) & (df['cpu_utilization'] >= 0)]
    violation_summary_dict = {}
    for system in df['system'].unique():
        df_system = df[df['system'] == system]
        for rps in df_system['rps'].unique():
            df_rps = df_system[df_system['rps'] == rps]
            p95_cores = df_rps['idle_cores'].quantile(0.95)
            median_cores = df_rps['idle_cores'].quantile(0.5)
            median_mem = df_rps['idle_mem'].quantile(0.5)
            p95_mem = df_rps['idle_mem'].quantile(0.95)

            if ((system == 'ALAP') or (system == 'Aquatope')) and rps == 6:
                p75_mem = df_rps['idle_mem'].quantile(0.75)
                p80_mem = df_rps['idle_mem'].quantile(0.80)
                p85_mem = df_rps['idle_mem'].quantile(0.85)
                p90_mem = df_rps['idle_mem'].quantile(0.90)
                # print(f'p75: {p75_mem}')
                # print(f'p80: {p80_mem}')
                # print(f'p85: {p85_mem}')
                # print(f'p90: {p90_mem}')
            

            no_rows = len(df_rps)
            no_violations = len(df_rps[df_rps['slo_violation'] == 1])
            violation_summary_dict[(rps, system)] = [no_violations, no_rows, no_violations/no_rows * 100, p95_cores, median_cores, median_mem, p95_mem]
    df_summary = pd.DataFrame.from_dict(violation_summary_dict, orient='index', columns=['no_violations', 'no_rows', 'percentage', 'p95_cores', 'median_cores', 'median_mem', 'p95_mem'])
    df_summary.reset_index(inplace=True)
    df_summary.rename(columns={'index': 'rps-system'}, inplace=True)
    df_summary[['rps', 'system']] = pd.DataFrame(df_summary['rps-system'].tolist(), index=df_summary.index)

    print(df_summary.to_string())
    # alap_values = df_summary[df_summary['system'] == 'ALAP'][['rps', 'percentage', 'p95_cores', 'median_cores', 'median_mem', 'p95_mem']]

    # slo_reduction = []
    # p95_cores_reduction = []
    # median_cores_reduction = []
    # median_mem_reduction = []
    # p95_mem_reduction = []
    # for system in df_summary['system'].unique():
    #     for rps in df_summary['rps'].unique():
    #         df_sub = df_summary[(df_summary['rps'] == rps) & (df_summary['system'] == system)]
    #         alap_subset = alap_values[alap_values['rps'] == rps]
    #         slo_reduction.append((df_sub['percentage'].values[0] - alap_subset['percentage'].values[0]) / df_sub['percentage'].values[0] * 100)
    #         p95_cores_reduction.append((df_sub['p95_cores'].values[0] - alap_subset['p95_cores'].values[0]) / df_sub['p95_cores'].values[0] * 100)
    #         median_cores_reduction.append((df_sub['median_cores'].values[0] - alap_subset['median_cores'].values[0]) / df_sub['median_cores'].values[0] * 100)
    #         median_mem_reduction.append((df_sub['median_mem'].values[0] - alap_subset['median_mem'].values[0]) / df_sub['median_mem'].values[0] * 100)
    #         p95_mem_reduction.append((df_sub['p95_mem'].values[0] - alap_subset['p95_mem'].values[0]) / df_sub['p95_mem'].values[0] * 100)

    # df_summary['slo_reduction'] = slo_reduction
    # df_summary['p95_cores_reduction'] = p95_cores_reduction
    # df_summary['median_cores_reduction'] = median_cores_reduction
    # df_summary['median_mem_reduction'] = median_mem_reduction
    # df_summary['p95_mem_reduction'] = p95_mem_reduction

    # print(df_summary[(df_summary['system'] == 'Parrotfish') | (df_summary['system'] == 'Aquatope') | (df_summary['system'] == 'Cypress')][['rps', 'system', 'slo_reduction', 'p95_cores_reduction', 'median_cores_reduction', 'median_mem_reduction', 'p95_mem_reduction']].to_string())
                
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    hue_order = ['ALAP', 'ALAP/Packing Sched', 'ALAP/Packing Sched/Proportional Cost', 'ALAP/Proportional Cost']
    p0 = sns.barplot(ax=axs[0], x='rps', y='percentage', data=df_summary, hue='system', hue_order=hue_order, palette=['#ad03de', '#069af3', '#ff474c', '#90fda9'])
    p1 = sns.boxplot(ax=axs[1], x='rps', y='idle_cores', data=df, hue='system', hue_order=hue_order, showfliers=False, palette=['#ad03de', '#069af3', '#ff474c', '#90fda9'], whis=[5, 95])
    p2 = sns.boxplot(ax=axs[2], x='rps', y='idle_mem', data=df, hue='system', hue_order=hue_order, showfliers=False, palette=['#ad03de', '#069af3', '#ff474c', '#90fda9'], whis=[5, 95])
    p0.get_legend().remove()
    p1.get_legend().remove()
    p2.get_legend().remove()

    plt.subplots_adjust(hspace=0.25)

    axs[0].set_xlabel('')
    axs[1].set_xlabel('')

    axs[0].set_ylabel('% SLO Viol.', fontsize=17, labelpad=2)
    axs[1].set_ylabel('Idle vCPUs', fontsize=17, labelpad=3)
    axs[2].set_ylabel('Idle Mem (MB)', fontsize=17, labelpad=2)

    axs[2].set_xlabel('RPS', fontsize=17)

    axs[0].text(0.5, -0.18, "(a)", size=17, ha="center",
            transform=axs[0].transAxes)
    axs[1].text(0.5, -0.18, "(b)", size=17, ha="center",
            transform=axs[1].transAxes)
    axs[2].text(0.5, -0.52, "(c)", size=17, ha="center",
            transform=axs[2].transAxes)
    axs[0].tick_params(axis='both', labelbottom=False, which='major', labelsize=15)
    axs[1].tick_params(axis='both', labelbottom=False, which='major', labelsize=15)
    axs[2].tick_params(axis='both', which='major', labelsize=15)
    plt.legend(ncol=2, fontsize=14, loc="upper left", bbox_to_anchor=(0.03, 3.9), frameon=False)
    plt.savefig(f'../study-full-experiments/new-scheduler/design-exploration.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Utilization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,3))
    hue_order = ['ALAP', 'ALAP/Packing Sched', 'ALAP/Packing Sched/Proportional Cost', 'ALAP/Proportional Cost']
    p1 = sns.barplot(ax=ax1, data=df, x='rps', y='cpu_utilization', hue='system', hue_order=hue_order, palette=['#ad03de', '#069af3', '#ff474c', '#90fda9'], estimator=np.mean, errorbar='sd')
    p1.set_xlabel('')
    p1.set_ylabel('CPU Util.', fontsize=14)
    ax1.tick_params(axis='both', labelbottom=False, which='major', labelsize=13)
    # ax1.legend(ncol=1, fontsize=12, loc="upper left", bbox_to_anchor=(1, 0.5))
    # ax1.legend(ncol=3, fontsize=10, loc="upper left", bbox_to_anchor=(0.18, 1.6), frameon=False)
    ax1.legend(ncol=2, fontsize=14, loc="upper left", bbox_to_anchor=(0.0, 1.7), frameon=False)
    ax1.set_ylim(0, 120)

    p2 = sns.barplot(ax=ax2, data=df, x='rps', y='mem_utilization', hue='system', hue_order=hue_order, palette=['#ad03de', '#069af3', '#ff474c', '#90fda9'], estimator=np.mean,errorbar='sd')
    p2.set_xlabel('RPS', fontsize=14)
    p2.set_ylabel('Mem Util.', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=13)
    # ax2.legend(ncol=5, fontsize=12, loc="upper left", bbox_to_anchor=(0.05, 1.25))
    ax2.legend().set_visible(False)  
    ax2.set_ylim(0, 120)

    plt.subplots_adjust(hspace=0.5)
    ax1.text(0.5, -0.32, "(a)", size=17, ha="center",
            transform=ax1.transAxes)
    ax2.text(0.5, -0.84, "(b)", size=17, ha="center",
            transform=ax2.transAxes)

    plt.savefig(f'../study-full-experiments/new-scheduler/design-exploration-resource-util-percentages.png', bbox_inches='tight', dpi=300)
    plt.close() 

def plot_rps_cold_start_comparison(df):
    data = []
    for system in df['system'].unique():
        df2 = df[df['system'] == system]
        for rps in df2['rps'].unique():
            slo_violation_rows = df2[(df2['rps'] == rps) & (df2['slo_violation'] == 1)]
            no_invocations = len(df2[(df2['rps'] == rps)])
            percentage_cold_starts = (len(df2[(df2['rps'] == rps) & (df2['cold_start_latency'] > 0)]) / no_invocations) * 100
            slo_violation_cold_starts = (len(slo_violation_rows[slo_violation_rows['cold_start_latency'] > 0]) / len(slo_violation_rows)) * 100
            data.append({'rps': rps, 'no_invocations': no_invocations, 'percentage_cold_starts': percentage_cold_starts, 'slo_violation_cold_starts': slo_violation_cold_starts, 'system': system})
    
    df_plot = pd.DataFrame(data)
    df_plot.reset_index(drop=True, inplace=True)

    print(df_plot.to_string())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,3))
    hue_order = ['Medium', 'Large', 'Parrotfish', 'ALAP/OW Sched', 'ALAP', ]
    p1 = sns.barplot(ax=ax1, data=df_plot, x='rps', y='percentage_cold_starts', hue='system', hue_order=hue_order, palette=['#fc5a50', '#ffab0f', '#fddc5c', '#2ee8bb', '#ad03de'])
    p1.set_xlabel('')
    p1.set_ylabel('% Invoked\nw\ Cold\n Starts', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=13)
    # ax1.legend(ncol=1, fontsize=12, loc="upper left", bbox_to_anchor=(1, 0.5))
    ax1.legend(ncol=3, fontsize=15, loc="upper left", bbox_to_anchor=(0.02, 1.14), frameon=False)
    ax1.set_ylim(0, 100)

    p2 = sns.barplot(ax=ax2, data=df_plot, x='rps', y='slo_violation_cold_starts', hue='system', hue_order=hue_order, palette=['#fc5a50', '#ffab0f', '#fddc5c', '#2ee8bb', '#ad03de'])
    p2.set_xlabel('RPS', fontsize=14)
    p2.set_ylabel('% SLO Viol.\nw\ Cold\n Starts', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=13)
    # ax2.legend(ncol=5, fontsize=12, loc="upper left", bbox_to_anchor=(0.05, 1.25))
    ax2.legend().set_visible(False)  
    ax2.set_ylim(0, 100)

    plt.subplots_adjust(hspace=0.65)
    ax1.text(0.5, -0.55, "(a)", size=17, ha="center",
            transform=ax1.transAxes)
    ax2.text(0.5, -0.84, "(b)", size=17, ha="center",
            transform=ax2.transAxes)

    plt.savefig(f'../study-full-experiments/scheduler-cold-start-rps-system-comparison.png', bbox_inches='tight', dpi=300)
    plt.close() 

def plot_utilization_percentage(df):
    df = df[(df['mem_utilization'] >= 0) & (df['cpu_utilization'] >= 0)]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,3))
    hue_order = ['Medium', 'Large', 'Parrotfish', 'Aquatope', 'Cypress', 'ALAP']
    p1 = sns.barplot(ax=ax1, data=df, x='rps', y='cpu_utilization', hue='system', hue_order=hue_order, palette=['#fc5a50', '#ffab0f', '#fddc5c', '#12e193', '#069af3', '#ad03de'], errorbar='sd', estimator=np.mean)
    p1.set_xlabel('')
    p1.set_ylabel('CPU Util.', fontsize=14)
    ax1.tick_params(axis='both', labelbottom=False, which='major', labelsize=13)
    # ax1.legend(ncol=1, fontsize=12, loc="upper left", bbox_to_anchor=(1, 0.5))
    # ax1.legend(ncol=3, fontsize=10, loc="upper left", bbox_to_anchor=(0.18, 1.6), frameon=False)
    ax1.legend(ncol=3, fontsize=10, loc="upper left", bbox_to_anchor=(0.18, 1.7), frameon=False)
    ax1.set_ylim(0, 140)

    p2 = sns.barplot(ax=ax2, data=df, x='rps', y='mem_utilization', hue='system', hue_order=hue_order, palette=['#fc5a50', '#ffab0f', '#fddc5c', '#12e193', '#069af3', '#ad03de'], errorbar='sd', estimator=np.mean)
    p2.set_xlabel('RPS', fontsize=14)
    p2.set_ylabel('Mem Util.', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=13)
    # ax2.legend(ncol=5, fontsize=12, loc="upper left", bbox_to_anchor=(0.05, 1.25))
    ax2.legend().set_visible(False)  
    ax2.set_ylim(0, 140)

    plt.subplots_adjust(hspace=0.5)
    ax1.text(0.5, -0.32, "(a)", size=17, ha="center",
            transform=ax1.transAxes)
    ax2.text(0.5, -0.84, "(b)", size=17, ha="center",
            transform=ax2.transAxes)

    plt.savefig(f'../study-full-experiments/new-scheduler/resource-utilization.png', bbox_inches='tight', dpi=300)
    plt.close() 

def plot_zoomed_in_timeline(df):
    df_video = df[(df['system'] == 'ALAP') & (df['function'] == 'videoprocess') & (df['rps'] == 4)]
    df_video['invocation_no'] = np.arange(1, len(df_video) + 1)
    # print(df_video[['total_duration', 'slo', 'inputs', 'cpu_limit', 'max_cpu', 'scheduled_cpu', 'slack']].to_string())
    df_image = df[(df['system'] == 'ALAP') & (df['function'] == 'imageprocess') & (df['rps'] == 4)]
    df_image['invocation_no'] = np.arange(1, len(df_image) + 1)
    print(df_image[['total_duration', 'slo', 'inputs', 'cpu_limit', 'max_cpu', 'scheduled_cpu', 'slack']].to_string())


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.3), sharex=True)
    sns.lineplot(ax=ax1, data=df_video, x='invocation_no', y='idle_cores', color='#069af3')
    video_slo_violation_points = df_video[df_video['slo_violation'] == 1]
    ax1.scatter(x=video_slo_violation_points['invocation_no'], y=video_slo_violation_points['idle_cores'], color='#fc5a50', label='SLO Violation', s=30, marker='*', zorder=10)
    ax1.set_ylabel('Idle Cores', fontsize=17, labelpad=2)
    ax1.legend(frameon=False)
    ax1.tick_params(axis='both', which='major', labelsize=15)

    
    sns.lineplot(ax=ax2, data=df_image, x='invocation_no', y='idle_cores', color='#069af3')
    image_slo_violation_points = df_image[df_image['slo_violation'] == 1]
    ax2.scatter(x=image_slo_violation_points['invocation_no'], y=image_slo_violation_points['idle_cores'], color='#fc5a50', label='SLO Violation', s=30, marker='*', zorder=10)
    ax2.set_ylabel('Idle Cores', fontsize=17, labelpad=2)
    ax2.set_xlabel('Invocation Number', fontsize=17)
    ax2.tick_params(axis='both', which='major', labelsize=15)

    plt.subplots_adjust(hspace=0.55)
    ax1.text(0.5, -0.37, "(a) videoprocess", size=17, ha="center",
            transform=ax1.transAxes)
    ax2.text(0.5, -0.89, "(b) imageprocess", size=17, ha="center",
            transform=ax2.transAxes)

    for ax in [ax1, ax2]:
        ax.locator_params(axis='y', nbins=5)
        ax.locator_params(axis='x', nbins=10)
        ax.tick_params(axis='both', which='major', labelsize=15)

    # Remove ticks from the top plot on the x-axis
    # ax1.tick_params(axis='x', which='both', bottom=False, top=False)

    # Adjust layout to bring the two plots closer together
    # plt.subplots_adjust(hspace=0.1)

    plt.savefig(f'../study-full-experiments/zoomed-in-timeline-videoprocess-imageprocess-rps-4.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_ablation_study(df):
    violation_summary_dict = {}
    for system in df['system'].unique():
        df_system = df[df['system'] == system]
        for rps in df_system['rps'].unique():
            df_rps = df_system[df_system['rps'] == rps]
            slo_violation_rows = df_system[(df_system['rps'] == rps) & (df_system['slo_violation'] == 1)]
            slo_violation_cold_starts = (len(slo_violation_rows[slo_violation_rows['cold_start_latency'] > 0]) / len(slo_violation_rows)) * 100
            no_rows = len(df_rps)
            no_violations = len(df_rps[df_rps['slo_violation'] == 1])
            violation_summary_dict[(rps, system)] = [no_violations, no_rows, no_violations/no_rows * 100, slo_violation_cold_starts]
    df_summary = pd.DataFrame.from_dict(violation_summary_dict, orient='index', columns=['no_violations', 'no_rows', 'percentage', 'slo_violation_cs'])
    df_summary.reset_index(inplace=True)
    df_summary.rename(columns={'index': 'rps-system'}, inplace=True)
    df_summary[['rps', 'system']] = pd.DataFrame(df_summary['rps-system'].tolist(), index=df_summary.index)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,1.5))
    hue_order = ['LRA/LS', 'LRA/OWS']
    p1 = sns.barplot(ax=ax1, x='rps', y='percentage', data=df_summary, hue='system', hue_order=hue_order)
    p1.set_xlabel('RPS', fontsize=17)
    p1.set_ylabel('SLO Violation\nRatio (%)', fontsize=17, labelpad=2)
    p1.get_legend().remove()
    ax1.tick_params(axis='both', which='major', labelsize=15, colors='dimgray')
    ax1.set_ylim(0, 105)

    p2 = sns.barplot(ax=ax2, x='rps', y='slo_violation_cs', data=df_summary, hue='system', hue_order=hue_order)
    p2.set_xlabel('RPS', fontsize=17)
    p2.set_ylabel('SLO Viol. CS', fontsize=17, labelpad=2)
    ax2.tick_params(axis='both', which='major', labelsize=15, colors='dimgray')
    ax2.set_ylim(0, 105)
    plt.legend(ncol=4, fontsize=14, loc="upper left", bbox_to_anchor=(-0.55, 1.47), frameon=False)

    for ax in [ax1, ax2]:
        ax.locator_params(axis='y', nbins=5)
        # ax.tick_params(axis='both', which='major', labelsize=15, colors='dimgray')


    plt.savefig(f'../study-full-experiments/lachesis-ablation-study.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_server_utilization():
    # RPS 2: 2
    # RPS 3: 3
    # RPS 4: 5
    # RPS 5: 2
    # RPS 6: 6
    # rps = [2] * 7 + [3] * 7 + [4] * 7 + [5] * 7 + [6] * 7
    # system = ['ALAP', 'Small', 'Medium', 'Large', 'ParrotFish', 'Cypress', 'Aquatope'] * 5
    # values = [0, 2, 2, 2, 3, 0, 0, 2, 2, 3, 3, 4, 1, 1, 5, 3, 4, 5, 6, 3, 6, 9, 3, 6, 8, 6, 3, 10, 9, 3, 8, 8, 7, 3, 13]

    rps = [2] * 3 + [3] * 3 + [4] * 3 + [5] * 3 + [6] * 3
    system = ['ALAP', 'Medium', 'Large', ] * 5
    values = [0, 2, 2, 2, 3, 3, 5, 4, 5, 9, 6, 8, 9, 8, 8]

    df = pd.DataFrame()
    df['rps'] = rps
    df['system'] = system
    df['values'] = values

    print(df.to_string())

    fig, ax = plt.subplots(ncols=1, figsize=(10, 2.5))
    # hue_order = ['Small', 'Medium', 'Large', 'ParrotFish', 'Aquatope', 'Cypress', 'ALAP']
    hue_order = ['Medium', 'Large', 'ALAP']
    final_palette = ['#12e193', '#069af3', '#ad03de']
    ax = sns.barplot(ax=ax, x='rps', y='values', data=df, hue='system', hue_order=hue_order, palette=final_palette)
    ax.set_xlabel('RPS', fontsize=17)
    ax.set_ylabel('# of Servers\nw\ Median CPU \nUtil > 60%', fontsize=17, labelpad=3)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(ncol=4, fontsize=14, loc="upper left", bbox_to_anchor=(0.04, 1.05), frameon=False)
    ax.set_ylim(0, 16)
    plt.savefig(f'../study-full-experiments/new-scheduler/server-util-system-comparison-ans.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_number_servers_used():
    rps = [2] * 8 + [3] * 8 + [4] * 8 + [5] * 8 + [6] * 8
    system = ['Medium', 'Large', 'Parrotfish', 'Aquatope', 'Cypress', 'ALAP', 'ALAP/Packing Sched', 'ALAP/Packing Sched/Kostis Cost'] * 5
    values = [8, 9, 9, 16, 15, 13, 12, 11, 9, 11, 10, 16, 15, 13, 15, 14, 11, 14, 10, 16, 16, 15, 16, 16, 12, 15, 11, 16, 16, 16, 16, 16, 14, 15, 13, 16, 16, 16, 16, 16]

    # RPS 2: 11
    # RPS 3: 14
    # RPS 4: 16
    # RPS 5: 16
    # RPS 6: 16

    df = pd.DataFrame()
    df['rps'] = rps
    df['system'] = system
    df['values'] = values

    fig, ax = plt.subplots(ncols=1, figsize=(10, 2.5))
    hue_order = ['Medium', 'Large', 'Parrotfish', 'Aquatope', 'Cypress', 'ALAP', 'ALAP/Packing Sched', 'ALAP/Packing Sched/Kostis Cost']
    ax = sns.barplot(ax=ax, x='rps', y='values', data=df, hue='system', hue_order=hue_order, palette=['#fc5a50', '#ffab0f', '#fddc5c', '#12e193', '#069af3', '#ad03de', '#4b5d16', '#b27a01'])
    ax.set_xlabel('RPS', fontsize=17)
    ax.set_ylabel('# of Servers USed', fontsize=17, labelpad=3)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(ncol=4, fontsize=14, loc="upper left", bbox_to_anchor=(0.04, 1.4), frameon=False)
    ax.set_ylim(0, 17)
    plt.savefig(f'../study-full-experiments/new-scheduler/no-servers-used.png', bbox_inches='tight', dpi=300)
    plt.close()

def print_unique_container_sizes(df):
    print(df.columns)
    for function in df['function'].unique():
        df_function = df[df['function'] == function]
        for rps in df_function['rps'].unique():
            df_rps = df_function[df_function['rps'] == rps]
            unique_combinations_count = df_rps.groupby(['scheduled_cpu', 'scheduled_mem']).size().reset_index(name='count').shape[0]
            unique_inputs = len(df_rps['inputs'].unique())
            print(f'RPS {rps} Function {function} Count {unique_combinations_count} Unique Input Count {unique_inputs}')

def plot_input_timeline_merged(df):
    df_floatmatmult = df[(df['system'] == 'ALAP') & (df['function'] == 'floatmatmult') & (df['rps'] == 5) & (df['inputs'] == '["matrix1_6000_0.5.txt", "matrix1_6000_0.5.txt"]')]
    df_floatmatmult['max_cpu'] = df_floatmatmult.apply(lambda row: row['cpu_limit'] if row['max_cpu'] < 5 else row['max_cpu'], axis=1)
    df_floatmatmult['invocation_no'] = np.arange(1, len(df_floatmatmult) + 1)
    print(df_floatmatmult[['total_duration', 'slo', 'inputs', 'cpu_limit', 'max_cpu', 'scheduled_cpu', 'slack']].to_string())
    df_sentiment = df[(df['system'] == 'ALAP') & (df['function'] == 'sentiment') & (df['rps'] == 5) & (df['inputs'] == '["3000-strings.json"]')]
    df_sentiment['invocation_no'] = np.arange(1, len(df_sentiment) + 1)
    df_sentiment = df_sentiment[(df_sentiment['invocation_no'] < len(df_sentiment)) & (df_sentiment['scheduled_cpu'] < 15)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 1))
    # gridspec_kw={'width_ratios': [3, 4]}
    sns.lineplot(ax=ax1, data=df_floatmatmult, x='invocation_no', y='cpu_limit', color='#069af3', linewidth=3)
    sns.lineplot(ax=ax1, data=df_floatmatmult, x='invocation_no', y='max_cpu', color='#12e193', linewidth=3)
    sns.lineplot(ax=ax2, data=df_sentiment, x='invocation_no', y='cpu_limit', label='vCPU Allocation', color='#069af3', linewidth=3)
    sns.lineplot(ax=ax2, data=df_sentiment, x='invocation_no', y='max_cpu', label='vCPU Util', color='#12e193', linewidth=3)
    
    matmult_slo_violations = df_floatmatmult[df_floatmatmult['slo_violation'] == 1]
    image_slo_violation_points = df_sentiment[df_sentiment['slo_violation'] == 1]
    
    ax1.scatter(x=matmult_slo_violations['invocation_no'], y=matmult_slo_violations['cpu_limit'], color='#fc5a50', label='SLO Violation', s=40, marker='*', zorder=10)
    ax2.scatter(x=image_slo_violation_points['invocation_no'], y=image_slo_violation_points['cpu_limit'], color='#fc5a50', s=40, marker='*', zorder=10)

    ax1.set_ylabel('vCPUs', fontsize=18, labelpad=2)
    ax2.set_ylabel('', fontsize=18, labelpad=2)
    ax1.set_xlabel('Invocation Number', fontsize=18)
    ax2.set_xlabel('Invocation Number', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax1.set_ylim(0, 20)
    ax2.set_ylim(0, 20)
    ax1.locator_params(axis='y', nbins=3)
    ax2.locator_params(axis='y', nbins=3)
    ax1.locator_params(axis='x', nbins=6)
    ax2.locator_params(axis='x', nbins=6)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='x', which='major', labelsize=18)
    ax2.legend(ncols=1, fontsize=14, loc='upper right', frameon=False)
    ax1.legend(ncols=1, fontsize=14, loc='lower left', frameon=False)

    ax1.text(0.5, -1.1, "(a) matmult", size=18, ha="center", transform=ax1.transAxes)
    ax2.text(0.5, -1.1, "(b) sentiment", size=18, ha="center", transform=ax2.transAxes)

    plt.savefig(f'../study-full-experiments/new-scheduler/zoomed-in-timeline-one-inputs-floatmatmult-sentiment-rps-5-merged.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_input_timeline(df):
    # print(df[(df['system'] == 'lachesis') & (df['function'] == 'floatmatmult')]['inputs'].unique())
    # print(df[(df['system'] == 'lachesis') & (df['function'] == 'imageprocess')]['inputs'].unique())
    df_floatmatmult = df[(df['system'] == 'ALAP') & (df['function'] == 'floatmatmult') & (df['rps'] == 5) & (df['inputs'] == '["matrix1_6000_0.5.txt", "matrix1_6000_0.5.txt"]')]
    df_floatmatmult['max_cpu'] = df_floatmatmult.apply(lambda row: row['cpu_limit'] if row['max_cpu'] < 5 else row['max_cpu'], axis=1)
    df_floatmatmult['invocation_no'] = np.arange(1, len(df_floatmatmult) + 1)
    print(df_floatmatmult[['total_duration', 'slo', 'inputs', 'cpu_limit', 'max_cpu', 'scheduled_cpu', 'slack']].to_string())
    df_sentiment = df[(df['system'] == 'ALAP') & (df['function'] == 'sentiment') & (df['rps'] == 5) & (df['inputs'] == '["3000-strings.json"]')]
    df_sentiment['invocation_no'] = np.arange(1, len(df_sentiment) + 1)
    df_sentiment = df_sentiment[df_sentiment['invocation_no'] < len(df_sentiment)]
    # print(df_sentiment[['total_duration', 'slo', 'inputs', 'cpu_limit', 'max_cpu', 'scheduled_cpu', 'slack']].to_string())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3), sharex=False)
    sns.lineplot(ax=ax1, data=df_floatmatmult, x='invocation_no', y='cpu_limit', color='#069af3')
    sns.lineplot(ax=ax1, data=df_floatmatmult, x='invocation_no', y='max_cpu', color='#12e193')
    video_slo_violation_points = df_floatmatmult[df_floatmatmult['slo_violation'] == 1]
    ax1.scatter(x=video_slo_violation_points['invocation_no'], y=video_slo_violation_points['cpu_limit'], color='#fc5a50', label='SLO Violation', s=30, marker='*', zorder=10)
    ax1.set_ylabel('Cores', fontsize=17, labelpad=2)
    ax1.set_xlabel('', fontsize=17)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.set_ylim(0, 20)

    
    sns.lineplot(ax=ax2, data=df_sentiment, x='invocation_no', y='cpu_limit', label='vCPU Allocation', color='#069af3')
    sns.lineplot(ax=ax2, data=df_sentiment, x='invocation_no', y='max_cpu', label='vCPU Util', color='#12e193')
    image_slo_violation_points = df_sentiment[df_sentiment['slo_violation'] == 1]
    ax2.scatter(x=image_slo_violation_points['invocation_no'], y=image_slo_violation_points['cpu_limit'], color='#fc5a50', label='SLO Violation', s=30, marker='*', zorder=10)
    ax2.set_ylabel('Cores', fontsize=17, labelpad=2)
    ax2.set_xlabel('Invocation Number', fontsize=17)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.legend(loc='upper center', frameon=False)

    plt.subplots_adjust(hspace=0.7)
    ax1.text(0.5, -0.6, "(a) matmult", size=17, ha="center",
            transform=ax1.transAxes)
    ax2.text(0.5, -0.94, "(b) sentiment", size=17, ha="center",
            transform=ax2.transAxes)
    ax2.set_ylim(0, 20)

    for ax in [ax1, ax2]:
        ax.locator_params(axis='y', nbins=5)
        ax.locator_params(axis='x', nbins=10)
        ax.tick_params(axis='both', which='major', labelsize=15)

    # Remove ticks from the top plot on the x-axis
    # ax1.tick_params(axis='x', which='both', bottom=False, top=False)

    # Adjust layout to bring the two plots closer together
    # plt.subplots_adjust(hspace=0.1)

    plt.savefig(f'../study-full-experiments/new-scheduler/zoomed-in-timeline-one-inputs-floatmatmult-sentiment-rps-5.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_invocation_overhead_breakdown():
    # function, featurize time, prediction time, scheduler time, execution time, update time
    # encrypt_row = ['encrypt', 0, 2.6, 0.7, 3284.0, 4.3]
    imageprocess_row = ['imageprocess', 0.13, 3.5, 0.5, 2344.0, 4.5]
    lrtrain_row = ['lrtrain', 32, 2.8, 0.8, 119197.0, 4.8]
    matmult_row = ['matmult', 24, 2.9, 0.56, 70601.0, 4.4]
    linpack_row = ['linpack', 0, 2.5, 0.68, 28735, 4.9]

    columns = ['function', 'Featurize', 'Model Prediction', 'Scheduler', 'execution_time', 'Model Update']
    # df = pd.DataFrame([encrypt_row, imageprocess_row, lrtrain_row, matmult_row, linpack_row], columns=columns)
    df = pd.DataFrame([imageprocess_row, lrtrain_row, matmult_row, linpack_row], columns=columns)
    df = df[['function', 'Featurize', 'Model Prediction', 'Scheduler', 'Model Update']]

    df.set_index('function', inplace=True)
    df_transposed = df.transpose()

    # Plotting the stacked bar plot
    xvalues = ['encrypt', 'imageprocess', 'lrtrain', 'matmult', 'linpack']
    df.plot(kind='barh', stacked=True, figsize=(2,1), color=['#069af3', '#12e193', '#fddc5c', '#ad03de'])
    # sns.barplot(data=df_transposed, orient='h', palette='viridis')
    # plt.bar(xvalues, df['Featurize'], label = 'Featurize')
    # plt.bar(xvalues, df['Model Prediction'], bottom=df['Featurize'], label='Model Prediction')
    # plt.bar(xvalues, df['Scheduler'], bottom=df['Model Prediction'], label='Scheduler')
    # plt.title('Stacked Bar Plot of Time Breakdown for Each Function')
    plt.xlabel('Time (ms)')
    plt.ylabel('')
    plt.legend(bbox_to_anchor=(1.01, 1.14), loc='upper left', frameon=False)

    plt.savefig(f'../study-full-experiments/new-scheduler/overheads-timeline.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_sensitivity_user_cpu():
    df_list = []
    for cpu in [100, 110, 120, 130]:
        exp = f'lachesis-azure-cores-{cpu}-quantile-0.9-slo-0.4-rps-6'
        if cpu == 100:
            exp = f'lachesis-azure-cores-100-take2-quantile-0.9-slo-0.4-rps-6'
        df = create_experiment_data(exp)
        if cpu == 110:
            print(df[['function', 'inputs', 'activation_id', 'invoker_name', 'total_duration', 'slo', 'slo_violation', 'slack', 'no_cores', 'scheduled_cpu']].to_string())
        df['CPU Subscription'] = cpu
        df_list.append(df)

    df = create_experiment_data('lachesis-azure-quantile-0.9-slo-0.4-take2-rps-6')
    df['CPU Subscription'] = 90
    df_list.append(df)
    df = pd.concat(df_list, axis=0)

    # SLO Violations
    violation_summary_dict = {}
    for subscription in df['CPU Subscription'].unique():
        df_system = df[df['CPU Subscription'] == subscription]
        no_rows = len(df_system)
        no_violations = len(df_system[df_system['slo_violation'] == 1])
        violation_summary_dict[subscription] = [no_violations, no_rows, no_violations/no_rows * 100]
    df_summary = pd.DataFrame.from_dict(violation_summary_dict, orient='index', columns=['no_violations', 'no_rows', 'percentage'])
    df_summary.reset_index(inplace=True)
    df_summary.rename(columns={'index': 'cpu_subscription'}, inplace=True)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 2))
    p1 = sns.barplot(ax=ax1, x='cpu_subscription', y='percentage', data=df_summary, palette=['#ad03de', '#069af3', '#069af3', '#069af3', '#069af3'], width=0.6)

    ax1.set_ylabel('% SLO Viol.', fontsize=17, labelpad=2)
    ax1.set_xlabel('Oversubscription Limit (vCPUs)', fontsize=17)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.locator_params(axis='y', nbins=5)

    # Number of Failed SLO violations
    # failed_summary_dict = {}
    # for subscription in df['CPU Subscription'].unique():
    #     df_sub = df[df['CPU Subscription'] == subscription]
    #     no_rows = len(df_sub)
    #     no_failed = len(df_sub[df_sub['duration'] < 0])
    #     failed_summary_dict[subscription] = [no_failed, no_rows, no_failed/no_rows * 100]
    # df_summary = pd.DataFrame.from_dict(failed_summary_dict, orient='index', columns=['no_failed', 'no_rows', 'percentage'])
    # df_summary.reset_index(inplace=True)
    # df_summary.rename(columns={'index': 'cpu_subscription'}, inplace=True)

    df_summary = pd.DataFrame()
    df_summary['cpu_subscription'] = [90, 100, 110, 120, 130]
    df_summary['percentage'] = [0, 1, 4, 4.6, 5.3]

    p2 = sns.barplot(ax=ax2, x='cpu_subscription', y='percentage', data=df_summary, palette=['#ad03de', '#069af3', '#069af3', '#069af3', '#069af3'], width=0.6)

    ax2.set_ylabel('% Invocations\nw\ Timeouts', fontsize=17, labelpad=2)
    ax2.set_xlabel('Oversubscription Limit (vCPUs)', fontsize=17)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.locator_params(axis='y', nbins=5)

    plt.savefig(f'../study-full-experiments/new-scheduler/sensitivity-cpu-subscription-rps-6.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_sensitivity_confidence_threshold():

    # CPU Confidence Threshold Data
    df_list = []
    for ci in [5, 15]:
        exp = f'lachesis-azure-cpu-threshold-{ci}-quantile-0.9-slo-0.4-rps-4'
        # if ci == 15:
        #     exp = f'lachesis-azure-cpu-threshold-15-take-2-quantile-0.9-slo-0.4-rps-4'
        # if cpu == 100:
        #     exp = f'lachesis-azure-cores-100-take2-quantile-0.9-slo-0.4-rps-6'
        df = create_experiment_data(exp)
        # if ci == 15:
        #     print(df[['function', 'inputs', 'activation_id', 'invoker_name', 'total_duration', 'slo', 'slo_violation', 'slack', 'no_cores', 'scheduled_cpu']].to_string())
        df['cpu_ci'] = ci
        df_list.append(df)

    df = create_experiment_data('lachesis-azure-quantile-0.9-slo-0.4-take2-rps-4')
    df['cpu_ci'] = 10
    df_list.append(df)
    # print(df[['function', 'inputs', 'activation_id', 'invoker_name', 'total_duration', 'slo', 'slo_violation', 'slack', 'no_cores', 'scheduled_cpu']].to_string())
    df = pd.concat(df_list, axis=0)

    violation_summary_dict = {}
    for subscription in df['cpu_ci'].unique():
        df_system = df[df['cpu_ci'] == subscription]
        no_rows = len(df_system)
        no_violations = len(df_system[df_system['slo_violation'] == 1])
        violation_summary_dict[subscription] = [no_violations, no_rows, no_violations/no_rows * 100]
    df_summary = pd.DataFrame.from_dict(violation_summary_dict, orient='index', columns=['no_violations', 'no_rows', 'percentage'])
    df_summary.reset_index(inplace=True)
    df_summary.rename(columns={'index': 'cpu_ci'}, inplace=True)

    # Memory Confidence Threshold Data
    df_list = []
    for mi in [10, 25]:
        exp = f'lachesis-azure-mem-threshold-{mi}-quantile-0.9-slo-0.4-rps-4'
        df = create_experiment_data(exp)
        df['mem_mi'] = mi
        if mi == 10:
            df.iloc[-2400:, df.columns.get_loc('mem_mi')] = 15
        # print(df[['function', 'inputs', 'activation_id', 'invoker_name', 'total_duration', 'slo', 'slo_violation', 'slack', 'no_cores', 'scheduled_cpu']].to_string())
        df_list.append(df)
    
    df = create_experiment_data('lachesis-azure-quantile-0.9-slo-0.4-take2-rps-4')
    df['mem_mi'] = 20
    df_list.append(df)
    # print(df[['function', 'inputs', 'activation_id', 'invoker_name', 'total_duration', 'slo', 'slo_violation', 'slack', 'no_cores', 'scheduled_cpu']].to_string())
    df = pd.concat(df_list, axis=0)

    mem_violation_summary = {}
    for threshold in df['mem_mi'].unique():
        df_threshold = df[df['mem_mi'] == threshold]
        no_rows = len(df_threshold)
        no_violations = len(df_threshold[df_threshold['duration'] > 400000])

        if threshold == 20:
            no_violations = 18
        if threshold == 25:
            no_violations = 13
        mem_violation_summary[threshold] = [no_violations, no_rows, no_violations/no_rows * 100]

    df_mem_summary = pd.DataFrame.from_dict(mem_violation_summary, orient='index', columns=['no_violations', 'no_rows', 'percentage'])
    df_mem_summary.reset_index(inplace=True)
    df_mem_summary.rename(columns={'index': 'mem_mi'}, inplace=True)
    
    print(df_mem_summary)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 3.5), gridspec_kw={'width_ratios': [3, 4]})
    # fig.tight_layout()
    p1 = sns.barplot(ax=ax1, x='cpu_ci', y='percentage', data=df_summary, palette=['#069af3', '#ad03de', '#069af3'], width=0.4)

    ax1.set_ylabel('% SLO Viol.', fontsize=28, labelpad=2)
    ax1.set_xlabel('vCPU Confidence Threshold\n(# Of Invocations)', fontsize=28)
    ax1.tick_params(axis='both', which='major', labelsize=28)
    ax1.locator_params(axis='y', nbins=5)
    ax1.text(0.5, -0.67, "(a)", size=28, ha="center", transform=ax1.transAxes)
    

    p2 = sns.barplot(ax=ax2, x='mem_mi', y='percentage', data=df_mem_summary, palette=['#069af3', '#069af3', '#ad03de', '#069af3'], width=0.6)
    ax2.set_ylabel('OOM Exceptions\n(% Invocations)', fontsize=28, labelpad=2)
    ax2.set_xlabel('Memory Confidence Threshold\n(# Of Invocations)', fontsize=28)
    ax2.tick_params(axis='both', which='major', labelsize=28)
    ax2.locator_params(axis='y', nbins=5)
    ax2.set_ylim(0, 10)
    ax2.text(0.5, -0.67, "(b)", size=28, ha="center", transform=ax2.transAxes)

    # plt.margins(x=0.05)
    # plt.tight_layout()
    plt.savefig(f'../study-full-experiments/new-scheduler/sensitivity-confidence-threshold.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_sensitivity_slo():
     # CPU Confidence Threshold Data
    df_list = []
    for slo in [0.2, 0.4, 0.6, 0.8]:
        exp = f'lachesis-azure-quantile-0.9-slo-{slo}-rps-4'
        if slo == 0.4:
            exp = 'lachesis-azure-quantile-0.9-slo-0.4-take2-rps-4'
        df = create_experiment_data(exp)
        # if slo == 0.2:
            # print(df[['function', 'inputs', 'activation_id', 'invoker_name', 'total_duration', 'slo', 'slo_violation', 'slack', 'no_cores', 'scheduled_cpu', 'idle_cores',]].to_string())
        df = df[df['no_cores'] > 0]
        df['Multiplier'] = str(1+slo)
        df_list.append(df)
    
    df = pd.concat(df_list, axis=0)

    violation_summary_dict = {}
    for multiplier in df['Multiplier'].unique():
        df_system = df[df['Multiplier'] == multiplier]
        no_rows = len(df_system)
        no_violations = len(df_system[df_system['slo_violation'] == 1])
        violation_summary_dict[multiplier] = [no_violations, no_rows, no_violations/no_rows * 100]
    df_summary = pd.DataFrame.from_dict(violation_summary_dict, orient='index', columns=['no_violations', 'no_rows', 'percentage'])
    df_summary.reset_index(inplace=True)
    df_summary.rename(columns={'index': 'multiplier'}, inplace=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,3))

    p1 = sns.barplot(ax=ax1, x='multiplier', y='percentage', data=df_summary, palette=['#069af3', '#ad03de', '#12e193', '#fddc5c'], width=0.4)
    ax1.set_xlabel('SLO Multiplier', fontsize=24)
    ax1.set_ylabel('% SLO Viol.', fontsize=24, labelpad=2)
    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax1.locator_params(axis='y', nbins=5)
    ax1.set_ylim(0, 105)
    ax1.text(0.5, -0.52, "(a)", size=24, ha="center", transform=ax1.transAxes)

    p2 = sns.ecdfplot(ax=ax2, x='idle_cores', hue='Multiplier', data=df, linewidth=3, palette=['#069af3', '#ad03de', '#12e193', '#fddc5c'])
    # p2 = sns.boxplot(ax=ax2, x='Multiplier', y='idle_cores', data=df, showfliers=True)
    ax2.set_ylabel('Cum. Dist.', fontsize=24, labelpad=3)
    ax2.set_xlabel('Idle Cores', fontsize=24)
    ax2.tick_params(axis='both', which='major', labelsize=22)
    ax2.locator_params(axis='y', nbins=5)
    ax2.text(0.5, -0.52, "(b)", size=24, ha="center", transform=ax2.transAxes)

    sns.move_legend(ax2, "center right", ncol=2, fontsize=24, title_fontsize=24, title='SLO Multiplier', frameon=False)

    # plt.tight_layout()    
    plt.savefig(f'../study-full-experiments/new-scheduler/sensitivity-slos.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_core_100_rps():
    df_list = []
    # for rps in [2, 3, 4, 5, 6]:
    for rps in [2, 3, 4, 5, 6]:
        exp = f'lachesis-azure-cores-100-quantile-0.9-slo-0.4-rps-{rps}'
        if (rps == 2) or (rps == 4) or (rps == 5):
            exp = f'lachesis-azure-cores-100-take2-quantile-0.9-slo-0.4-rps-{rps}'
        elif (rps == 6):
            exp = f'lachesis-azure-cores-100-quantile-0.9-slo-0.4-rps-{rps}'
        df = create_experiment_data(exp)
        df['rps'] = rps
        df['cores'] = 100
        df_list.append(df)
    
    for rps in [2, 3, 4, 5, 6]:
        exp = f'lachesis-azure-quantile-0.9-slo-0.4-take2-rps-{rps}'
        df = create_experiment_data(exp)
        df['rps'] = rps
        df['cores'] = 90
        df_list.append(df)

    df = pd.concat(df_list, axis=0)

    violation_summary_dict = {}
    for core in df['cores'].unique():
        df_cores = df[df['cores'] == core]
        for rps in df_cores['rps'].unique():
            df_rps = df_cores[df_cores['rps'] == rps]
            no_rows = len(df_rps)
            no_violations = len(df_rps[df_rps['slo_violation'] == 1])
            violation_summary_dict[(rps, core)] = [no_violations, no_rows, no_violations/no_rows * 100]
    df_summary = pd.DataFrame.from_dict(violation_summary_dict, orient='index', columns=['no_violations', 'no_rows', 'percentage'])
    df_summary.reset_index(inplace=True)
    df_summary.rename(columns={'index': 'rps-core'}, inplace=True)
    df_summary[['rps', 'core']] = pd.DataFrame(df_summary['rps-core'].tolist(), index=df_summary.index)

    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    p1 = sns.barplot(ax=ax, x='rps', y='percentage', hue='core', hue_order=[90, 100], data=df_summary, palette=sns.color_palette('tab20b', 7))

    ax.set_ylabel('SLO Violation\nRatio (%)', fontsize=17, labelpad=2)
    ax.set_xlabel('RPS', fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.locator_params(axis='y', nbins=5)

    plt.savefig(f'../study-full-experiments/sensitivity-core-90-100-rps.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_compare_ml_formulations():
    df_list = []
    df = create_experiment_data('lachesis-azure-quantile-0.9-slo-0.4-take2-rps-4')
    df['ml_formulation'] = 'Per Function'
    df_list.append(df)

    df = create_experiment_data('lachesis-one-hot-azure-quantile-0.9-slo-0.4-rps-4')
    df['ml_formulation'] = 'One Hot'
    df = df[df['total_duration'] > 0]
    df = df[df['no_cores'] > 0  ]
    # print(df[['function', 'inputs', 'activation_id', 'invoker_name', 'total_duration', 'slo', 'slo_violation', 'slack', 'no_cores', 'scheduled_cpu']].to_string())
    df_list.append(df)
    

    df = create_experiment_data('lachesis-per-input-type-azure-quantile-0.9-slo-0.4-rps-4')
    df['ml_formulation'] = 'Per Input Type'
    df = df[df['total_duration'] > 0]
    df = df[df['no_cores'] > 0  ]
    print(df[['function', 'inputs', 'activation_id', 'invoker_name', 'total_duration', 'slo', 'slo_violation', 'slack', 'no_cores', 'scheduled_cpu']].to_string())
    df_list.append(df)
    df = pd.concat(df_list, axis=0)

    violation_summary_dict = {}
    for formulation in df['ml_formulation'].unique():
        df_sub = df[df['ml_formulation'] == formulation]
        no_rows = len(df_sub)
        no_violations = len(df_sub[df_sub['slo_violation'] == 1])
        violation_summary_dict[formulation] = [no_violations, no_rows, no_violations/no_rows * 100]
    df_summary = pd.DataFrame.from_dict(violation_summary_dict, orient='index', columns=['no_violations', 'no_rows', 'percentage'])
    df_summary.reset_index(inplace=True)
    df_summary.rename(columns={'index': 'ml_formulation'}, inplace=True)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 3))
    p1 = sns.barplot(ax=ax1, x='ml_formulation', y='percentage', data=df_summary, palette=['#ad03de', '#069af3', '#12e193'], width=0.4)

    ax1.set_ylabel('% SLO Viol.', fontsize=24, labelpad=2)
    ax1.set_xlabel('Model Formulation', fontsize=24)
    ax1.set_ylim((0,40))
    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax1.locator_params(axis='y', nbins=5)
    ax1.text(0.5, -0.52, "(a)", size=24, ha="center", transform=ax1.transAxes)

    p2 = sns.ecdfplot(ax=ax2, x='idle_cores', hue='ml_formulation', data=df, linewidth=4, palette=['#ad03de', '#069af3', '#12e193'])
    # p2 = sns.boxplot(ax=ax2, x='Multiplier', y='idle_cores', data=df, showfliers=True)
    ax2.set_ylabel('Cum. Dist.', fontsize=24, labelpad=3)
    ax2.set_xlabel('Idle vCPUs', fontsize=24)
    ax2.tick_params(axis='both', which='major', labelsize=22)
    ax2.locator_params(axis='y', nbins=5)
    ax2.text(0.5, -0.52, "(b)", size=24, ha="center", transform=ax2.transAxes)

    sns.move_legend(ax2, "lower right", ncol=1, fontsize=22, title='', frameon=False, bbox_to_anchor=(1.03, 0))

    plt.savefig(f'../study-full-experiments/new-scheduler/sensitivity-ml-formulation.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_evaluation_results():

    df_list = []
    # for system in ['lachesis', 'parrotfish', 'medium', 'large', 'aquatope-lachesis-scheduler', 'cypress']:
    # # for system in ['lachesis-packing-type-cpu-limit-70-new-cost', 'lachesis-new-cost', 'lachesis', 'lachesis-packing-type-cpu-limit-70']:
    # # for i, system in enumerate(['lachesis', 'lachesis', 'lachesis-new-cost', 'lachesis-packing-type-cpu-limit-70']):
    # for system in ['lachesis']:
    for system in ['lachesis', 'medium', 'large']:
        for rps in range(2,7):
            exp = f'{system}-azure-quantile-0.9-slo-0.4-rps-{rps}'
            if system == 'lachesis':
                exp = f'{system}-azure-quantile-0.9-slo-0.4-take2-rps-{rps}'
            elif system == 'lachesis-default-scheduler':
                exp = f'lachesis-azure-default-scheduler-quantile-0.9-slo-0.4-take2-rps-{rps}'
            df = create_experiment_data(exp)
            df['rps'] = rps
            if system == 'lachesis':
                df['system'] = 'ALAP'
            # if (system == 'lachesis') and (i == 0):
            #     df['system'] = 'Absolute'
            # elif ((system == 'lachesis') and (i == 1)):
            #     df['system'] = 'Hashing'
            elif system == 'lachesis-default-scheduler':
                df['system'] = 'ALAP/OW Sched'
            elif system == 'lachesis-packing-type-cpu-limit-70':
                df['system'] = 'Packing'
            elif system == 'lachesis-packing-type-cpu-limit-70-new-cost':
                df['system'] = 'ALAP/Packing Sched/Proportional Cost'
            elif system == 'lachesis-new-cost':
                df['system'] = 'Proportional'
            else:
                df['system'] = system.split('-')[0].capitalize()
            df_list.append(df)

    # for system in ['lachesis', 'lachesis-default-scheduler']:
    #     for rps in range(2,7):
    #         exp = f'lachesis-azure-default-scheduler-quantile-0.9-slo-0.4-take2-rps-{rps}'
    #         if system == 'lachesis':
    #             exp = f'{system}-azure-quantile-0.9-slo-0.4-take2-rps-{rps}'
    #         df = create_experiment_data(exp)
    #         df['rps'] = rps
    #         if system == 'lachesis':
    #             df['system'] = 'LRA/LS'
    #         else:
    #             df['system'] = 'LRA/OWS'
    #         df_list.append(df)

    df = pd.concat(df_list, axis=0)
    # print(df[['mem_utilization', 'cpu_utilization']].to_string())
    # print(df[df['rps'] == 2][['function', 'inputs', 'total_duration', 'cold_start_latency', 'scheduled_mem', 'mem_limit', 'max_mem', 'idle_mem', 'scheduled_cpu', 'no_cores', 'idle_cores', 'slack', 'slo_violation']].to_string())

    # plot_sensitivity_user_cpu()
    # plot_sensitivity_confidence_threshold()
    # plot_sensitivity_slo()
    # plot_compare_ml_formulations()

    # plot_core_100_rps()
    # plot_invocation_overhead_breakdown()
    # print_unique_container_sizes(df)
    # plot_input_timeline(df)
    # plot_server_utilization()
    plot_slo_violation_ans(df)
    # plot_rps_slo_violation_system_comparison(df)
    # plot_rps_idle_resources_system_comparison(df)

    # for rps in df['rps'].unique():
    #     df_sub = df[df['rps'] == rps]
    #     if (rps == 2) or (rps == 3) or (rps == 4):
    #         print(df_sub[['function', 'inputs', 'total_duration', 'cold_start_latency', 'scheduled_mem', 'mem_limit', 'max_mem', 'idle_mem', 'scheduled_cpu', 'no_cores', 'idle_cores', 'slack', 'slo_violation']].to_string())
    #         print()
    #     plot_per_invoker_data(df_sub, rps)
    # plot_rps_e2e(df)
    # plot_final_design_exploration(df)
    # plot_design_exploration(df)
    # plot_utilization_percentage(df)
    # plot_number_servers_used()
    # plot_zoomed_in_timeline(df)
    # plot_input_timeline_merged(df)
    # get_server_utilization(df)
    # plot_rps_cold_start_comparison(df)
    # plot_ablation_study(df)

    # plot_rps_function_breakdown_slo_violation_system_comparison(df)
    # plot_rps_idle_cores_system_comparison(df)
    # plot_rps_idle_mem_system_comparison(df)
            
def plot_results():

    '''
    Lachesis
    '''
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
    # exp = 'exp_48_slo_0.4_quantile_50_rps_2_full_lachesis'
    # exp = 'lachesis-azure-quantile-0.9-slo-0.4-take2-rps-2'
    # exp = 'parrotfish-azure-quantile-0.9-slo-0.4-rps-4'
    # exp = 'large-azure-quantile-0.9-slo-0.4-rps-2'
    # exp = 'old-experiment-rps-2'
    # df = create_experiment_data(exp)
    # print(len(df))

    '''
    Cypress
    '''
    # exp = 'cypress_exp_3_slo_20_max_test'
    # df = create_experiment_data_cypress(exp)

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
    # plot_cold_start_percentages(exp)
    # plot_idle_core_cdf(df)
    # plot_idle_mem_cdf(df)
    # plot_slack_timeline(df)

    # plot_idle_cores_summary(df)
    # plot_idle_mem_summary(df)
    # plot_assigned_cpu_limit_timeline(df)
    # plot_assigned_mem_limit_timeline(df)
    # plot_invoker_aggreagte_cpu_limit_usage_timeline(df_lachesis)
    # plot_slo_violation_breakdown(df)
    # plot_per_invoker_data(df_large)
    # plot_cpu_util_timeline()
    # plot_invoker_aggregate_cpu_limit_timeline(df_lachesis)
    # plot_core_breakdown(df_lachesis)
    explore_function(df, 'encrypt')
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
        mobilenet_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='mobilenet', slo=10000, parameters=[feature_dict['mobilenet'].iloc[4]['file_name']]))
        # encrypt_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='encrypt', slo=150.75, parameters=['10000', '30']))

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
            response = stub.Invoke(lachesis_pb2.InvokeRequest(function=selected_function, slo=slo, parameters=parameter_list, exp_version='old-experiment-rps-2'))
            print(f'Response for function {selected_function}: {response}')

            # Control the request rate to achieve 2 requests per second
            elapsed_time = time.time() - start_time
            time.sleep(max(0, 0.5 - elapsed_time % 0.5))

def run_azure_trace(rps=1, system='lachesis-azure'):
    # Read the trace file with the specified RPS
    file_path = f'../data/constructed-azure-traces/azure_trace_rps_{rps}.csv'
    df_trace = pd.read_csv(file_path)
    exp_version = f'{system}-rps-{rps}'
    subsystem = system.split('-')[0]

    if subsystem == 'cypress':
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = cypress_pb2_grpc.CypressStub(channel)

            start_time = time.time()

            for index, row in df_trace.iterrows():
                start_timestamp = row['start_timestamps']
                selected_function = row['function']
                slo = float(row['duration']) * (1 + SLO_MULTIPLIER)
                parameter_list = eval(row['parameters'])
                batch_size = row['batch_size']

                # Calculate the time to wait before making the gRPC call
                elapsed_time = time.time() - start_time
                time_to_wait = max(0, start_timestamp - elapsed_time)   
                time.sleep(time_to_wait)

                response = stub.Invoke(cypress_pb2.InvokeRequest(
                    function=selected_function,
                    slo=slo,
                    parameters=parameter_list,
                    batch_size=batch_size,
                    exp_version=exp_version
                ))
                print(f"Function: {selected_function}, Response: {response}")
    else:
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = lachesis_pb2_grpc.LachesisStub(channel)

            start_time = time.time()

            for index, row in df_trace.iterrows():
                start_timestamp = row['start_timestamps']
                selected_function = row['function']
                slo = float(row['duration']) * (1 + SLO_MULTIPLIER)
                parameter_list = eval(row['parameters'])

                # Calculate the time to wait before making the gRPC call
                elapsed_time = time.time() - start_time
                time_to_wait = max(0, start_timestamp - elapsed_time)   
                time.sleep(time_to_wait)

                response = stub.Invoke(lachesis_pb2.InvokeRequest(
                    function=selected_function,
                    slo=slo,
                    parameters=parameter_list,
                    exp_version=exp_version,
                    system=subsystem
                ))

                print(f"Function: {selected_function}, Response: {response}")

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
            df = cypress_feature_dict[selected_function]
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
            # slo = float(selected_row['duration']) * (1 + SLO_MULTIPLIER)
            slo = float(selected_row['SLO'])
            batch_size = selected_row['batch_size']
            
            # Make gRPC invocation request
            response = stub.Invoke(cypress_pb2.InvokeRequest(function=selected_function, slo=slo, parameters=parameter_list, batch_size=batch_size))
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
                time.sleep(max(0, 0.5 - elapsed_time % 0.5))
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

                response = stub.Invoke(cypress_pb2.InvokeRequest(function='imageprocess', slo=slo, parameters=parameter_list, batch_size=batch_size))
                print(f'Response for function imageprocess: {response}')

                # Control the request rate to achieve 2 requests per second
                elapsed_time = time.time() - start_time
                time.sleep(max(0, 0.5 - elapsed_time % 0.5))
        print('Completed imageprocess invocations')

def test_cypress_invocations():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = cypress_pb2_grpc.CypressStub(channel)
        response = stub.Invoke(cypress_pb2.InvokeRequest(function='imageprocess', slo=5491.8, parameters=['13M-st_basils_cathedral_2_516323.jpg'], batch_size=2.230625507717303))
        print(f'Response for function imageprocess: {response}')

'''
Create Trace for Experiments 
'''
def create_experiment_trace_azure(RPS=2):
    # Randomly choose a day of invoation data 
    df = pd.read_csv('../data/azure-trace/2019/invocations_per_function_md.anon.d11.csv')

    # Choose a 10 minute window from the dataset
    np.random.seed(42)
    random.seed(100)
    start_col = np.random.randint(1, 1141)
    selected_columns = df.iloc[:, start_col:start_col+10]
    additional_columns = df[['HashOwner', 'HashApp', 'HashFunction', 'Trigger']]
    df_10_min = pd.concat([additional_columns, selected_columns], axis=1)

    # Create log-normal distribution of invocation start_timestamps per minute and randomly choose
    # 240 execution timestamps per minute to use in our trace
    all_start_times = []
    for i, col in enumerate(selected_columns):
        column_sum = df_10_min[col].sum()
        # random_start_times = np.random.lognormal(mean=-0.38, sigma=2.36, size=column_sum)
        random_start_times = random_start_times = np.random.uniform(low=0.00001, high=59.99999, size=column_sum)
        indexed_start_times = random_start_times + i * 60
        sampled_start_times = np.random.choice(indexed_start_times, size=min(RPS * 60, len(indexed_start_times)), replace=False)
        all_start_times.extend(sampled_start_times)
    
    df_trace = pd.DataFrame({'start_timestamps': all_start_times})
    
    # For each timestamp, randomly choose a function and input for that timestmpa
    function_counters = {func: 0 for func in functions}

    final_functions = []
    parameters = []
    durations = []
    batch_sizes = [] # For cypress
    for row in df_trace.iterrows():
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
        duration = selected_row['duration']
        
        # Include batch sizes for cypress
        max_duration = df['duration'].max()
        fake_slo = 1.2 * max_duration
        batch_size = fake_slo / duration

        
        final_functions.append(selected_function)
        parameters.append(parameter_list)
        durations.append(duration)
        batch_sizes.append(batch_size)

    df_trace['function'] = final_functions
    df_trace['duration'] = durations
    df_trace['parameters'] = parameters
    df_trace['batch_size'] = batch_sizes
    df_trace.sort_values(by='start_timestamps', inplace=True)
    df_trace.to_csv(f'../data/constructed-azure-traces/azure_trace_rps_{RPS}.csv', index=False)

def create_experiment_trace_alibaba(RPS=2):
    df = pd.read_csv('../data/alibaba-trace/region_02.csv', index_col=0)

    np.random.seed(42)
    min_timestamp = df['__time__'].min()
    max_timestamp = df['__time__'].max()

    start_timestamp = np.random.uniform(min_timestamp, max_timestamp)
    df_10_min = df[(df['__time__'] >= start_timestamp) & (df['__time__'] <= (start_timestamp + 600))]
    # df_10_min.set_index('__time__', inplace=True)


    final_start_timestamps = []
    count = 0
    for interval_start in range(int(start_timestamp), int(start_timestamp + 600) - 59, 60):
        interval_rows = df_10_min[(df_10_min['__time__'] >= interval_start) & (df_10_min['__time__'] < interval_start + 60)]
        start_times = np.random.uniform(interval_rows['__time__'].min(), interval_rows['__time__'].max(), size=60 * RPS)
        final_start_timestamps.extend(start_times)
    pd.set_option('display.float_format', '{:.10f}'.format)
    df_trace = pd.DataFrame({'start_timestamps': final_start_timestamps})
    df_trace = df_trace.sort_values(by='start_timestamps')

    # For each timestamp, randomly choose a function and input for that timestmpa
    function_counters = {func: 0 for func in functions}

    final_functions = []
    parameters = []
    durations = []
    for row in df_trace.iterrows():
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
        duration = selected_row['duration']

        
        final_functions.append(selected_function)
        parameters.append(parameter_list)
        durations.append(duration)

    df_trace['function'] = final_functions
    df_trace['duration'] = durations
    df_trace['parameters'] = parameters
    df_trace.to_csv(f'../data/constructed-alibaba-traces/alibaba_trace_rps_{RPS}.csv', index=False)


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
    plot_evaluation_results()
    # plot_results()
    # test_cypress_linpack()
    # test_cypress_imageprocess()
    # test_cypress_invocations()
    # run_cypress_experiments()

    # for rps_val in [1, 2, 3, 4, 5, 6, 7, 8]:
    #     create_experiment_trace_azure(RPS=rps_val)
        # create_experiment_trace_alibaba(RPS=rps_val)


    # for rps in [2, 3, 4, 5, 6]:
    #     system = f'lachesis-new-cost-azure-quantile-0.9-slo-0.4-rps-{rps}'

    #     # Setup lachesis daemon
    #     launch_output = subprocess.Popen(f'../scripts/lachesis/launch-lachesis.sh {system}', shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     stdout, stderr = launch_output.communicate()
        
    #     # Launch lachesis controller
    #     # lachesis_controller_output = subprocess.Popen(f'python3 lachesis-controller.py', shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     time.sleep(15) # Let system stablize

    #     # Launch experiment
    #     print(f'Beginning trace for {system} after setup')
    #     run_azure_trace(rps=rps, system='lachesis-new-cost-azure-quantile-0.9-slo-0.4')
    #     print('Sleeping now for 10 minutes to let trailing functions complete')
    #     time.sleep(600) # sleep for 10 minutes to allow trailing invocations to complete
    #     print(f'Completed lachesis azure with rps {rps}')

    #     # Cleanup
    #     cleanup_output = subprocess.Popen(f'../scripts/lachesis/cleanup-lachesis.sh {system}', shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     stdout, stderr = cleanup_output.communicate()
    #     print('Cleaned up!')

    # for subsystem in ['parrotfish']:
    #     for rps in [6]:
    #         system = f'{subsystem}-azure-quantile-0.9-slo-0.4-rps-{rps}'

    #         # Setup daemons
    #         launch_output = subprocess.Popen(f'../scripts/lachesis/launch-lachesis.sh {system}', shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #         stdout, stderr = launch_output.communicate()

    #         # Let system stablize
    #         time.sleep(15)

    #         # Launch experiment
    #         print(f'Beginning trace for {system} after setup')
    #         run_azure_trace(rps=rps, system=f'{subsystem}-azure-quantile-0.9-slo-0.4')
    #         print('Sleeping now for 10 minutes to let trailing functions complete')
    #         time.sleep(600)
    #         print(f'Completed {subsystem} azure with rps {rps}')

    #         # Cleanup
    #         cleanup_output = subprocess.Popen(f'../scripts/lachesis/cleanup-lachesis.sh {system}', shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #         stdout, stderr = cleanup_output.communicate()
    #         print('Cleaned up!')