# -*- coding: utf-8 -*-
"""
Analysis setup for taking snapshot of trials at sampled timesteps, creating percentages
tables, and saving design matrix joined with sampled data.

Intended to be run as cells instead of as a module.

"""

if __name__ == '__main__':
    raise Exception('do not run this as main')

#
# ! Assumes process_trial_multiproc has been ran, and /trial_results/001_results.csv files exist
#

import pandas as pd
from pandas.api.types import CategoricalDtype as ctype
import numpy as np

import statsmodels.formula.api as smf


# this is only useful for top-level aggregate statistics. need to join with trial number for more details,
# but better to just use one dataframe/file at a time.
results_filename_template = 'C:/Users/Mike/Desktop/tmp/wsc_data/trial_step_results/{:03}_results.csv'
all_the_results = pd.read_csv(results_filename_template.format(1))
for i in range(2, 256):
    all_the_results = all_the_results.append(pd.read_csv(results_filename_template.format(i)), ignore_index=True)


######################


def prepare_design_df():
    """
    Return design matrix with some supplemental columns inserted and everything
    properly type-cast for OLS.
    """
    from design_matrix_tools import reverse_g_val

    # Add max degree as an analysis factor
    from network_graphs import instance_name_map, instance_definitions, get_network_instance
    max_degree = dict()
    for N, v in instance_definitions.items():
        for i in v:
            G = get_network_instance(N, i)
            max_degree[(N, i)] = np.max(G.out_degree, axis=0)[1]

    design_filename = 'design_matrix_WSC2018.csv'
    design_df = pd.read_csv(design_filename, dtype={
        'g': 'int64',
        'g_level': 'category',
        'randomize_update_seq': 'category',
        'b_distro': 'category',
        'b_0_is_zero': 'category',
        'normalize_bs': 'category',
        'normalize_yis': 'category',
    })
    design_df.set_index('Trial', inplace=True)

    # Include human-readable structure instance names and map the batch number g to the factor level
    # Also include # of X variables in trial. These may be useful as proxies for original factors.
    design_df.insert(2, 'structure', None)
    design_df.insert(4, 'g_level', None)
    design_df.insert(3, 'num_X_vars', None)
    for i, row in design_df.iterrows():
        design_df.at[i, 'structure'] = instance_name_map[row.structure_instance_num]
        design_df.at[i, 'g_level'] = reverse_g_val(row.g, row.N)
        design_df.at[i, 'num_X_vars'] = max_degree[(row.N, row.structure_instance_num)]
    design_df['structure'] = design_df.structure.astype(ctype())
    design_df['g_level'] = design_df.g_level.astype(ctype())
    design_df['g_rate'] = design_df['g'] / design_df['N']
    design_df['num_X_vars'] = design_df.num_X_vars.astype('int64')

    return design_df


#######################
# Classify trial based on thresholds for a sample of timesteps
#######################

results_filename_template = 'C:/Users/Mike/Desktop/tmp/wsc_data/trial_step_results/{:03}_results.csv'

# Classifier constants

min_linear_R2 = 0.5 # threshold for saying a trial is "Linear"

sampled_timesteps = [100, 250, 500]

###

design_df = prepare_design_df()

for trial_num in design_df.index:
    trial_results = pd.read_csv(results_filename_template.format(trial_num))
    trial_results.set_index('Step', inplace=True)

    for t in sampled_timesteps:
        # Using moving average of R2 & p-value now
        design_df.at[trial_num, 'MA_R2_Adj__t_{}'.format(t)] = trial_results.loc[t, 'MA-R2 Adj.']
        design_df.at[trial_num, 'MA-p_value__t_{}'.format(t)] = trial_results.loc[t, 'MA-p-value Significant']
        design_df.at[trial_num, 'Outliers_Dropped__t_{}'.format(t)] = trial_results.loc[t, 'Num Outliers Dropped']

        # Note that now the determination of "moving average" p-value is set by
        # process_trial(), using p<=0.1 = Significant
        R2 = trial_results.loc[t, 'MA-R2 Adj.']
        pval_significant = trial_results.loc[t, 'MA-p-value Significant']

        good_R2_val = lambda R2: (min_linear_R2 <= R2 < 1)

        if good_R2_val(R2) and (pval_significant == 1):
            # Good R2 and significant p-value
            classification = 'Linear'

        elif (-np.inf < R2 < min_linear_R2) or (good_R2_val(R2) and pval_significant == 0):
            # Either a bad R2 value; or a good R2 but insignificant by p-value
            # The 2nd term keeps us from saying an invalid R2 is Not Linear, as all invalid R2 should have
            # pval_significant = False
            classification = 'Not Linear'

        elif not (-np.inf < R2 < 1) or np.isnan(R2):
            # R2 is not an acceptable value: infinite, null, or == 1
            # R2 exactly = 1 is bad -- occurs when Y very large, Condition very poor, data otherwise suspect
            # R2 null or infinite is bad -- simply unusuable data
            classification = 'Invalid'

        else:
            classification = '???'  # Error trap for logic flow

        # These two were captured for interest and diagnostics, but not using them as of now.
        design_df.at[trial_num, 'Max_Y__{}'.format(t)] = trial_results.loc[t, 'Max Y']
        design_df.at[trial_num, 'Max_Corr_Y__{}'.format(t)] = trial_results.loc[t, 'Max Corr_Y']

        design_df.at[trial_num, 'Classification__t_{}'.format(t)] = classification

# Typecast classification columns to support OLS auto-recognition
classification_cols = filter(lambda s: s.startswith('Classification'), design_df.columns)
for col in classification_cols:
    design_df[col] = design_df[col].astype(ctype())

design_df.to_csv('design_with_trial_results.csv')
print(design_df['Classification__t_100'].value_counts())


###############
# Linear OLS regression on design matrix, with the time-sampled R^2 value as the response
#
# Here, we do not filter out too-low R^2, but do filter out Invalid trials
###############


# head to brute_force_model_selection for a better treatment of this

factors = [
    'N',

#    'structure',
    'num_X_vars',

    'g_level',  # original categorical -- best for main effects
#    'g_rate',   # bad for main effects only R2 (t=500)
#    'g',        # integer number of batches

    'randomize_update_seq',
    'b_distro',
    'b_0_is_zero',
    'normalize_bs',
    'error_variance',
    'normalize_yis',
    'uninformed_rate',
   ]

fits = dict()
for t in [500]:  #sampled_timesteps:
    # Helper vars
    classification_col = 'Classification__t_{}'.format(t)
    r2_col = 'MA_R2_Adj__t_{}'.format(t)

    # Create data table for regression
    data2 = design_df[factors+[r2_col, classification_col]].copy()
    data2 = data2[data2[classification_col] != 'Invalid'].dropna()  # Get rid of nan/inf R^2 & p-values
    data2.drop(columns=classification_col, inplace=True)

    formula = 'MA_R2_Adj__t_{} ~ {}'.format(t, ' + '.join(factors))
    fits[t] = smf.ols(formula, data2).fit()

    print(fits[t].summary())

### vif block


from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

# get y and X dataframes based on this regression:
y, x = dmatrices(formula, data2, return_type='dataframe')

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif["features"] = x.columns
print(vif.round(1))


##############
# Percentages table as function of classification (Invalid, Linear, Not Linear)
#############


# Get list of all (factor, level) pairs present in design matrix
percentage_factors = [
    'N',
    'structure',
    'num_X_vars',
    'g_level',
    'g_rate',
    'randomize_update_seq',
    'b_distro',
    'b_0_is_zero',
    'normalize_bs',
    'error_variance',
    'normalize_yis',
    'uninformed_rate',
   ]

row_labels = []
for factor in percentage_factors:
    levels = design_df[factor].unique()
    row_labels.extend([(factor, level) for level in levels])

percentage_tables = dict()
for t in sampled_timesteps:
    classification_col = 'Classification__t_{}'.format(t)
    classification_counts = design_df[classification_col].value_counts()

    data3 = pd.DataFrame(columns=list(classification_counts.keys())+['Row Totals'],
                         index=pd.MultiIndex.from_tuples(row_labels, names=['Factor', 'Level']))

    # Create frequency table (raw counts)
    # FUTURE: See if I can do this with pandas.crosstab()
    for factor, level in row_labels:
        for classification in classification_counts.keys():
            data3.loc[(factor, level), classification] = \
                len(design_df[(design_df[factor] == level) & (design_df[classification_col] == classification)])
    data3['Row Totals'] = data3.sum(axis=1)

    # Normalize within rows
    for factor, level in row_labels:
        for classification in classification_counts.keys():
            data3.loc[(factor, level), classification] /= data3.loc[(factor, level), 'Row Totals']

    percentage_tables[t] = data3
    data3.to_csv('percentages_{}.csv'.format(t))

###########
# Create dot plot for classifications by factor-level
###########
# goto plots.py
