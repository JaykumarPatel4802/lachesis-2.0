import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_data():
    df = pd.read_csv('../../data/energy-study/energy-container-creation.csv')
    df.columns = ['container', 'cpu_lim', 'memory_lim', 'runtime', 'energy_s0', 'pwr_s0', 'energy_s1', 'pwr_s1']
    docker_image_size_dict = {'psinha25-python': '1.96GB', 'psinha25-resnet': '3.46GB', 'psinha25-mobilenet': '3.26GB', 'psinha25-redis': '138MB', 'psinha25-nginx': '187MB', 'psinha25-mysql': '577MB', 'psinha25-prometheus': '245MB'}
    df['docker_image_size'] = df['container'].map(docker_image_size_dict)
    df['total_pwr'] = df['pwr_s0'] + df['pwr_s1']
    df['total_energy'] = df['energy_s0'] + df['energy_s1']
    return df

def cpu_energy(df):
    plot_df = df[df['memory_lim'] == 2048]
    p1 = sns.barplot(data=plot_df, x='cpu_lim', y='total_energy', hue='docker_image_size')
    p1.set_xlabel('CPU Limit to Assign Container \n(Memory Fixed: 2GB)')
    p1.set_ylabel('Energy Consumed to Create Container (J)')
    p1.set_ylim(0, plot_df['total_energy'].max() + 2)
    plt.legend(title='Container Image Size', bbox_to_anchor=(0.5, 1.15), loc='center', ncol=3)
    plt.savefig('./plots/cpu-limit-energy-consumption.png', bbox_inches='tight', dpi=300)
    plt.close()


def cpu_pwr(df):
    plot_df = df[df['memory_lim'] == 2048]
    p1 = sns.barplot(data=plot_df, x='cpu_lim', y='total_pwr', hue='docker_image_size')
    p1.set_xlabel('CPU Limit to Assign Container \n(Memory Fixed: 2GB)')
    p1.set_ylabel('Power Consumed to Create Container (W)')
    p1.set_ylim(0, plot_df['total_pwr'].max() + 2)
    plt.legend(title='Container Image Size', bbox_to_anchor=(0.5, 1.15), loc='center', ncol=3)
    plt.savefig('./plots/cpu-limit-pwr-consumption.png', bbox_inches='tight', dpi=300)
    plt.close()

def mem_energy(df):
    plot_df = df[df['cpu_lim'] == 16]
    p1 = sns.barplot(data=plot_df, x='memory_lim', y='total_energy', hue='docker_image_size')
    p1.set_xlabel('Memory Limit to Assign Container (MB) \n(CPU Fixed: 16)')
    p1.set_ylabel('Energy Consumed to Create Container (J)')
    p1.set_ylim(0, plot_df['total_energy'].max() + 2)
    plt.legend(title='Container Image Size', bbox_to_anchor=(0.5, 1.15), loc='center', ncol=3)
    plt.savefig('./plots/mem-limit-energy-consumption.png', bbox_inches='tight', dpi=300)
    plt.close()

def mem_pwr(df):
    plot_df = df[df['cpu_lim'] == 16]
    p1 = sns.barplot(data=plot_df, x='memory_lim', y='total_pwr', hue='docker_image_size')
    p1.set_xlabel('Memory Limit to Assign Container (MB) \n(CPU Fixed: 16)')
    p1.set_ylabel('Power Consumed to Create Container (W)')
    p1.set_ylim(0, plot_df['total_pwr'].max() + 2)
    plt.legend(title='Container Image Size', bbox_to_anchor=(0.5, 1.15), loc='center', ncol=3)
    plt.savefig('./plots/mem-limit-pwr-consumption.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__=='__main__':
    df = preprocess_data()
    cpu_energy(df)
    cpu_pwr(df)
    mem_energy(df)
    mem_pwr(df)
