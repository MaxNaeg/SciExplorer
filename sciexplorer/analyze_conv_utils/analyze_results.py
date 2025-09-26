import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

import os
import json

def analyze_folder(saved_path:str, 
                   metric_key:str, 
                   average_func:callable=np.mean, 
                   save_fig:bool=False,
                   contained_in_name:list=[],
                   title='',
                   bar_colors:list=None,
                   print_data:bool=False):
    '''Plot bar plots for the results produced by runner in the folder.'''

    if not contained_in_name:
        contained_in_name = ['agent_run', 'no_tool_run']
    
    # list fo all files (not directories) in the saved_path
    all_files = sorted([f for f in os.listdir(saved_path) if os.path.isfile(os.path.join(saved_path, f))])
    # filter files that contain the specified strings in their names
    filtered_files = [[f for f in all_files if c in f] for c in contained_in_name]

    def extract_metric(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        result = data.get('result', {})
        mse = result.get(metric_key, None)
        return mse
    
    data_unproc = [np.array([extract_metric(os.path.join(saved_path, f)) for f in files]) for files in filtered_files]

    # remove empty data from values 
    contained_in_name = [c for i, c in enumerate(contained_in_name) if data_unproc[i].size > 0]
    data_unproc = [d for d in data_unproc if d.size > 0]
    

    if metric_key == 'R2':
        data = [np.clip(np.nan_to_num(d, nan=0), 0, 1) for d in data_unproc]
    elif metric_key == 'accuracy':
        data = [np.nan_to_num(d, nan=0) for d in data_unproc]
    else:
        data = [np.array(d) for d in data_unproc]
    if metric_key == 'fidelity':
        data = [np.clip(np.nan_to_num(d, nan=-np.inf), -1., 0.) for d in data_unproc]

    if data == []:
        return
    
    fig, ax = plt.subplots(figsize=(6, 5))

    if not bar_colors:
        bar_colors = ['tomato', 'royalblue']

    x_positions = np.arange(len(data))
    bar_width = 0.4

    means = []
    cis_lower = []
    cis_upper = []

    for i, values in enumerate(data):
        ci = 0.9
        mean = average_func(values)
        if print_data:
            print(f"{contained_in_name[i]}: {values}, mean: {mean}")
        if values.size>1:
            res = bootstrap((values,), average_func, confidence_level=ci, n_resamples=10000, method='percentile', random_state=42)
            ci_low, ci_high = res.confidence_interval
        else:
            ci_low, ci_high = mean, mean
        means.append(mean)
        cis_lower.append(mean - ci_low)
        cis_upper.append(ci_high - mean)

        # Evenly spaced dots within the bar width
        spread = np.linspace(-bar_width / 2.2, bar_width / 2.2, len(values))
        ax.plot(x_positions[i] + spread, values, 'o', color='black', alpha=0.7)

    # Draw bars with error bars and different colors
    for i in range(len(data)):
        ax.bar(x_positions[i], means[i], 
            yerr=[[cis_lower[i]], [cis_upper[i]]], 
            capsize=10, color=bar_colors[i], edgecolor=None, width=bar_width)
    ax.grid()
    # Labels and layout
    ax.set_xticks(x_positions)
    ax.set_xticklabels(contained_in_name)
    ax.set_ylabel(metric_key)
    ax.set_title(f'{title}, {average_func.__name__}, confidence interval {ci}')
    if metric_key == 'MSE':
        ax.set_yscale('symlog', linthresh=1e-5)
    fig.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(saved_path, f'{metric_key}_comparison.pdf'), bbox_inches='tight')
    else:
        plt.show()
    




