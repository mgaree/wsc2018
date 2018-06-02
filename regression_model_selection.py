# -*- coding: utf-8 -*-
"""
Code for selecting best or preferred regression models on the experimental design and MA-AR2
"""

from itertools import chain, combinations

from tqdm import tqdm
import statsmodels.formula.api as smf


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def two_ways(iterable):
    s = list(iterable)
    if len(s) == 1:
        return (s + [0],)
    return combinations(s, 2)


# Full set of design factors that could possibly be included in regression fit
    # can toggle between variants to try different things
factors = [
    'N',

#    'structure',
    'num_X_vars',

    'g_level',  # original categorical -- best for main effects
#    'g_rate',   # bad for main effects only R2 (t=500)
#    'g',        # integer number of batches, also bad

    'randomize_update_seq',
    'b_distro',
    'b_0_is_zero',
    'normalize_bs',
    'error_variance',
    'normalize_yis',
    'uninformed_rate',
   ]


def fit_factor_set(factor_set, df, dependent_var_col, main_effects_only=True):
    if main_effects_only:
        formula = '{} ~ {}'.format(dependent_var_col, ' + '.join(factor_set))
    else:
        formula = '{} ~ {}'.format(dependent_var_col, ' + '.join(['{}*{}'.format(i, j) for i, j in two_ways(factor_set)]))

    return smf.ols(formula, df).fit()


def get_dataframe(t):
    global design_df  # assumes analysis_wsc has put design_df in the global NS

    classification_col = 'Classification__t_{}'.format(t)

    data2 = design_df[design_df[classification_col] != 'Invalid'].dropna(subset=[classification_col,])  # Get rid of nan/inf R^2 & p-values
    return data2


def get_best_fit(t, main_effects_only=True):
    dependent_var_col = 'MA_R2_Adj__t_{}'.format(t)

    data2 = get_dataframe(t)

    cur_fit = None
    best_factor_set = None
    best_r2 = -999

    factorsets = list(powerset(factors))
    for factor_set in tqdm(factorsets, total=len(factorsets)):
        # Create data table for regression
        fs = list(factor_set)
        if len(fs) < 1:
            continue

        cur_fit = fit_factor_set(fs, data2, dependent_var_col, main_effects_only)

        if cur_fit.rsquared_adj > best_r2:
            best_r2 = cur_fit.rsquared_adj
            best_factor_set = fs

    # Return best
    best_fit = fit_factor_set(best_factor_set, data2, dependent_var_col, main_effects_only)
    print(best_fit.summary())
    return best_fit, best_factor_set


def backward_elimination_for_interaction_model(t):
    dependent_var_col = 'MA_R2_Adj__t_{}'.format(t)
    data2 = get_dataframe(t)

    remaining = list(two_ways(factors))

    while len(remaining) > 0:
        # Measure model with all of remaining
        formula = '{} ~ {}'.format(dependent_var_col, ' + '.join(
                        ['{}*{}'.format(i, j) for i, j in remaining]))
        cur_fit = smf.ols(formula, data2).fit()

        # If removing another factor doesn't help, then stop
        r2_to_beat = cur_fit.rsquared_adj

        best_r2_this_pass = r2_to_beat
        term_to_remove = None

        for ff in remaining:
            formula = '{} ~ {}'.format(dependent_var_col, ' + '.join(
                    ['{}*{}'.format(i, j) for i, j in [x for x in remaining if x != ff]]))
            cur_fit = smf.ols(formula, data2).fit()
            if cur_fit.rsquared_adj > best_r2_this_pass:
                best_r2_this_pass = cur_fit.rsquared_adj
                term_to_remove = ff

        if term_to_remove is None:
            break  # removing another term doesn't help
        else:
            remaining.remove(term_to_remove)

    formula = '{} ~ {}'.format(dependent_var_col, ' + '.join(
                        ['{}*{}'.format(i, j) for i, j in remaining]))
    best_fit = smf.ols(formula, data2).fit()
    print(best_fit.summary())
    return best_fit, remaining


"""
    Plotting
"""

import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)
sns.set(style='whitegrid')


# selected != best; I found the best and removed some factors; hurts R2 a little, simplifies model a lot
selected_main_effects_model_factorset = [
    'num_X_vars',  # aka max_di
    'g_level',
    'b_distro',
    'normalize_bs',
    'normalize_yis',
   ]

selected_interaction_model_factorset = [
	'g_level',
    'b_distro',
    'normalize_bs',
    'normalize_yis',
    'randomize_update_seq',
    'error_variance',
    'b_0_is_zero',
	]

def get_selected_main_effects_model(t=500):
    global factors
    tmp = list(factors)
    factors = selected_main_effects_model_factorset
    main_effects_model, _ = get_best_fit(t=500, main_effects_only=True)
    factors = tmp
    return main_effects_model

def get_selected_interaction_model(t=500):
    global factors
    tmp = list(factors)
    factors = selected_interaction_model_factorset
    interaction_model, _ = backward_elimination_for_interaction_model(t=500)
    factors = tmp
    return interaction_model

def build_diagnostic_plots(model, t):
    # plots for main effects only
    axes = [
        Rplots.residual_plot(model, get_dataframe(t)),
        Rplots.QQ_plot(model, get_dataframe(t)),
        Rplots.pred_vs_actual_plot(model, get_dataframe(t)),
        ]

    for ax in axes:
        ax.figure.set_size_inches(4,4, forward=True)
        ax.figure.tight_layout()

    return axes


import Rplots

t = 500

main_effects_model = get_selected_main_effects_model(t)
axes = build_diagnostic_plots(main_effects_model, t)

interaction_model = get_selected_interaction_model(t)
axes = build_diagnostic_plots(interaction_model, t)

#============

if __name__ == '__main__':
    factors = selected_main_effects_model_factorset
    best_main_effects_model, best_factor_set = get_best_fit(t=500, main_effects_only=True)

#    best_interaction_model, best_interaction_factor_set = get_best_fit(t=500, main_effects_only=False)

    # interaction terms greedily removed from full interaction model
#    best_backelim_model, best_backelim_fs = backward_elimination_for_interaction_model(t=500)
