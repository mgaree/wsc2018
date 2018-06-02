# -*- coding: utf-8 -*-
""" Get classification of time step for all trials, all timesteps

This is to support the question, "if linear now, linear later?"

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
sns.set(style='white')


results_filename_template = 'C:/Users/Mike/Desktop/tmp/wsc_data/trial_step_results/{:03}_results.csv'

# Classifier constants
min_linear_R2 = 0.5 # threshold for saying a trial is "Linear"


trial_nums = range(1, 256)
time_steps = range(5, 501)  # start at 5 because no R2 moving average till then

time_classification_data = pd.DataFrame(index=time_steps, columns=trial_nums)

for trial_num in trial_nums:
    trial_results = pd.read_csv(results_filename_template.format(trial_num))
    trial_results.set_index('Step', inplace=True)

    for step in time_steps:
        R2 = trial_results.loc[step, 'MA-R2 Adj.']
        pval_significant = trial_results.loc[step, 'MA-p-value Significant']

        good_R2_val = lambda R2: (min_linear_R2 <= R2 < 1)

        if good_R2_val(R2) and (pval_significant == 1):
            # Good R2 and significant p-value
            classification = 1 #'Linear'

        elif (-np.inf < R2 < min_linear_R2) or (good_R2_val(R2) and pval_significant == 0):
            # Either a bad R2 value; or a good R2 but insignificant p-value
            # The 2nd term keeps us from saying an invalid R2 is Not Linear, as
            # all invalid R2 should have pval_significant = False
            classification = -1 #'Not Linear'

        elif not (-np.inf < R2 < 1) or np.isnan(R2):
            # R2 is not an acceptable value: infinite, null, or == 1
            # R2 exactly = 1 is bad -- occurs when Y very large, Condition very poor, data otherwise suspect
            # R2 null or infinite is bad -- simply unusuable data
            classification = 0 #'Invalid'

        time_classification_data.at[step, trial_num] = classification

time_classification_data = time_classification_data.T.drop_duplicates().T # delete duplicate columns


cmap=plt.get_cmap('jet', 3)
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist[0] = (0.7,0.7,0.7,1)
cmaplist[1] = (1, 1, 1, 1)
cmaplist[2] = (0.1,1,0,1)
cmap = cmap.from_list('x', cmaplist, cmap.N)

df = time_classification_data.T.astype(int)


### for curiousity, filter out trials that end on Invalid
#keep = time_classification_data.loc[496][time_classification_data.loc[496] != 0]
#df = df.loc[keep.index]
##

df.sort_values(by=list((df.columns)), inplace=True)

ax = sns.heatmap(df[df.columns[:-2]], cmap=cmap, cbar=False,
                 linewidths=0,
                 xticklabels=100, yticklabels=False)

ax.figure.set_size_inches(6.5, 3.8, forward=True)
ax.set_xlabel('Time step')
ax.set_xticks([1, 250, 500])
ax.set_xticklabels([1, 250, 500])
ax.set_title('     Trial classification vs Time step', loc='left')

ax.set_ylabel('Classification pattern')

# custom legend
from matplotlib.patches import Patch

legend_elements = [
        Patch(facecolor=cmaplist[0], edgecolor=None, label='Not Linear'),
        Patch(facecolor=cmaplist[1], edgecolor='black', label='Invalid'),
        Patch(facecolor=cmaplist[2], edgecolor=None, label='Linear'),
]

ax.legend(handles=legend_elements, loc='lower right', ncol=3, bbox_to_anchor=(1,1), borderaxespad=0, handletextpad=0.25, columnspacing=1)
ax.figure.tight_layout()
