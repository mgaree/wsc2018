# -*- coding: utf-8 -*-
"""
Code for some plots
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)
sns.set(style='whitegrid')


instance_name_map = {
    1: 'Directed Random Tree',
    2: 'Directed Random Tree (Reversed)',
    3: 'Erdos-Renyi Random #1a',
    4: 'Erdos-Renyi Random #1b',
    5: 'Erdos-Renyi Random #2a',
    6: 'Erdos-Renyi Random #2b',
    7: 'Scale-Free #1a',
    8: 'Scale-Free #1b',
    9: 'Scale-Free #2a',
    10: 'Scale-Free #2b',
    11: 'Random k-out (k=3) #a',
    12: 'Random k-out (k=3) #b',
    13: 'Random k-out (k=10) #a',
    14: 'Random k-out (k=10) #b',
}


# copied from design_matrix_tools but tweaked to match current context
factors = dict(
    N=['100', '500', '1000'],
    structure=list(instance_name_map.values()),
    g_level=['1', '5', 'N/4', 'N'],
    randomize_update_seq=['True', 'False'],
    b_distro=['U01', 'U-11', 'N01'],
    b_0_is_zero=['True', 'False'],
    normalize_bs=['no', 'agent', 'network'],
    error_variance=['0.5', '1.0', '2.0'],  # Need to change 1 to 1.0 in percentages_t.csv
    normalize_yis=['True', 'False'],
    uninformed_rate=['0.0', '0.1', '0.25']
    )

label_map = {
    'U-11': '$U(-1, 1)$',
    'U01': '$U(0, 1)$',
    'N01': '$N(0, 1)$',
    'g_level': 'Batches',
    'b_0_is_zero': 'Fix $b_{i0} = 0$',
    'b_distro': '$b_{ij}$ Distribution',
    'error_variance': 'Error Variance',
    'normalize_bs': 'Normalize $b_{i0}$, $b_{ij}$',
    'normalize_yis': 'Normalize $y_i$',
    'randomize_update_seq': 'Update Sequence',
    'uninformed_rate': 'Uninformed Rate',
    '0.25': '25%',
    '0.1': '10%',
    '0.0': '0%',
    'structure': 'Structure',
    'N/4': '$N/4$',
    'N': '$N$',

    'Directed Random Tree': 'Random Tree',
    'Directed Random Tree (Reversed)': 'Random Tree (Reversed)',
    'Erdos-Renyi Random #1a': 'Erdos-Renyi #1a',
    'Erdos-Renyi Random #1b': '#1b',
    'Erdos-Renyi Random #2a': 'Erdos-Renyi #2a',
    'Erdos-Renyi Random #2b': '#2b',
    'Scale-Free #1a': 'Scale-Free #1a',
    'Scale-Free #1b': '#1b',
    'Scale-Free #2a': 'Scale-Free #2a',
    'Scale-Free #2b': '#2b',
    'Random k-out (k=3) #a': 'Random k-out ($k$=3) #a',
    'Random k-out (k=3) #b': '#b',
    'Random k-out (k=10) #a': 'Random k-out ($k$=10) #a',
    'Random k-out (k=10) #b': '#b',
}

def fmt_lbl(s):
    try:
        return str(label_map[s])
    except:
        return str(s)


factor_list = ['N', 'b_0_is_zero', 'b_distro', 'error_variance', 'g_level',
       'normalize_bs', 'normalize_yis', 'randomize_update_seq',
       'uninformed_rate']

percentage_factors = [
    'N',
    'g_level',
    'randomize_update_seq',
    'b_distro',
    'b_0_is_zero',
    'normalize_bs',
    'error_variance',
    'normalize_yis',
    'uninformed_rate',
#    'structure',  # omitting from paper because same name != same network, e.g. Scale-Free 1a (N=100) has different metrics than Scale-Free 1a (N=500)
   ]


def better_dot_plot(df, t):
    columns = ['Linear', 'Not Linear', 'Invalid']
    colors = ['lime', 'slategrey', 'firebrick']
    markers = ['o', 'v', 'x']

    text_labels = []
    factor_positions = []  # where to draw axhlines
    row_indexes = []  # get right data for row

    i = 1
    for f in percentage_factors:
        text_labels.append(fmt_lbl(f))
        factor_positions.append(i)
        row_indexes.append(None)
        i += 1
        for l in factors[f]:
            text_labels.append(fmt_lbl(l))
            row_indexes.append( (f, l) )
            i += 1

    # kludge to make some True/False labels better
    text_labels[10] = 'Random'
    text_labels[11] = 'Fixed'

    # gather data
    data = {'Linear': [], 'Not Linear': [], 'Invalid': [], 'Row Totals': []}
    keys = data.keys()
    for r in row_indexes:
        if not r:
            for k in keys:
                data[k].append(None)
        else:
            for k in keys:
                data[k].append(df.loc[r, k])


    yvals = range(1, len(text_labels)+1)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 5.1))

    for col, color, marker in zip(columns, colors, markers):
        ax.scatter(data[col], yvals, color=color, label=col, marker=marker, linewidths=1).set_clip_on(False)

    ax.set_ylim((1, yvals[-1]+0.5))
    ax.set_yticks(yvals)
    ax.set_yticklabels(text_labels)

    # Embolden factor names on y axis
    ylabs = ax.get_ymajorticklabels()
    for i in factor_positions:
        ylabs[i-1].set_fontweight('bold')

    ax2 = ax.twiny()  # horizontal axis on top of chart

    ax2.set_xlabel('Fraction of all trials with factor-level, by classification (t = {})'.format(t))

    for axis in (ax, ax2):
        axis.set_xlim((0, 0.8))
        axis.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        axis.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8])
        axis.tick_params(axis='x', which='major', pad=4)

        axis.spines['top'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_visible(False)

        axis.xaxis.grid(False)
        axis.yaxis.grid(True)

    # Add divider lines between factors
    ax.hlines(y=factor_positions, xmin=-0.0125, xmax=0.8, color='DodgerBlue',
              alpha=1, linewidth=1.75, linestyles='solid').set_clip_on(False)

    fig.legend(ncol=1, loc='upper left', handletextpad=0.1)

    fig.gca().invert_yaxis()
    fig.tight_layout()

    fig.show()
    return fig


if __name__ == '__main__':
    # percentage table files comes from analysis_wsc.py
    t = 500
    df = pd.read_csv('percentages_{}.csv'.format(t), index_col=[0, 1])

    fig = better_dot_plot(df, t)
