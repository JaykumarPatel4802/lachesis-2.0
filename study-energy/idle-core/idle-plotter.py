import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Plot power consumption as the number of online cores per socket varies 
def pwr_online_cores():
    df = pd.read_csv('~/lachesis/data/energy-study/data-energy-idle.csv')
    df.columns = ['run_no', 'no_online_cpu', 'total_pwr', 'pwr_s0', 'pwr_s1']

    plot_df = pd.DataFrame(columns=['run_no', 'no_online_cpu', 'pwr', 'socket'])
    counter = 1
    run_nos = []
    no_online_cpus = []
    pwrs = []
    sockets = []
    for no_online_cpu in df['no_online_cpu'].unique():
        sub_df = df[df['no_online_cpu'] == no_online_cpu]

        s0_cpu = [counter] * len(sub_df)
        s1_cpu = [counter] * len(sub_df)
        s0_pwr = sub_df['pwr_s0'].tolist()
        s1_pwr = sub_df['pwr_s1'].tolist()
        s0_socket = ['socket 0'] * len(sub_df)
        s1_socket = ['socket 1'] * len(sub_df)

        run_nos.extend(sub_df['run_no'].tolist())
        run_nos.extend(sub_df['run_no'].tolist())
        no_online_cpus.extend(s0_cpu)
        no_online_cpus.extend(s1_cpu)
        pwrs.extend(s0_pwr)
        pwrs.extend(s1_pwr)
        sockets.extend(s0_socket)
        sockets.extend(s1_socket)

        counter += 1
    
    plot_df['run_no'] = run_nos
    plot_df['no_online_cpu'] = no_online_cpus
    plot_df['pwr'] = pwrs
    plot_df['socket'] = sockets

    p1 = sns.lineplot(data=plot_df, x='no_online_cpu', y='pwr', hue='socket')
    p1.set_xlabel('Online Cores per Socket')
    p1.set_ylabel('Power Consumption (Watts)')
    # p1.set_ylim(0, 37)
    plt.savefig('./plots/online-cores-vs-pwr-zoomed-view.png', bbox_inches='tight', dpi=300)
    plt.close() 


def c6_timeseries():

    timestamps = []
    cpu_nos = []
    c6_states = []
    timestamp = 5
    with open('../../data/energy-study/raw-cstate-data.txt') as f:
        lines = f.readlines()
        for line in lines:
            if ('CPU' not in line) and ('-' not in line):
                split_line = line.split('\t')
                if len(split_line) == 1:
                    timestamp += 5
                else:
                    timestamps.append(int(timestamp))
                    cpu_nos.append('cpu' + split_line[0])
                    c6_states.append(float(split_line[1][:-2]))

    df = pd.DataFrame()
    df['timestamp'] = timestamps
    df['cpu_no'] = cpu_nos
    df['c6_state'] = c6_states
    print(df['timestamp'].to_string())

    plot_df = df[(df['cpu_no'] == 'cpu4') | (df['cpu_no'] == 'cpu13') | (df['cpu_no'] == 'cpu25') | (df['cpu_no'] == 'cpu46')]
    print(plot_df)
    p1 = sns.lineplot(data=plot_df, x='timestamp', y='c6_state', hue='cpu_no')
    p1.set_xlabel('Time (s)')
    p1.set_ylabel('Percentage of 5s Interval\nCore is in C6 Power State')
    # p1.set_ylim(0, 101)
    plt.savefig('./plots/c6-state-timeseries-zoomed-view.png', bbox_inches='tight', dpi=300)
    plt.close()


                



if __name__=='__main__':
    # pwr_online_cores()
    c6_timeseries()
