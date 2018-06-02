# -*- coding: utf-8 -*-
"""
R-style plots

adapted from https://emredjan.github.io/blog/2017/07/11/emulating-r-plots-in-python/

"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.graphics.gofplots import ProbPlot


plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)
sns.set(style='white')


def residual_plot(model_fit, data):
    fig = plt.figure(1)
    fig.set_figheight(5)
    fig.set_figwidth(5)

    model_fitted_y = model_fit.fittedvalues
    model_residuals = model_fit.resid

    fig.axes[0] = sns.residplot(
        model_fitted_y, model_fit.model.endog_names, data=data,
        lowess=True,
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8}
        )

    ax = fig.axes[0]
    ax.set_title('Residuals vs Fitted')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')

    # annotations
    abs_resid = np.abs(model_residuals).sort_values()
    abs_resid_top_3 = abs_resid[-3:]

    for i in abs_resid_top_3.index:
        ax.annotate(i, xy=(model_fitted_y[i], model_residuals[i]))

    fig.tight_layout()
    return ax


def QQ_plot(model_fit, data):
    model_norm_residuals = model_fit.get_influence().resid_studentized_internal

    QQ = ProbPlot(model_norm_residuals)
    fig = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

    fig.set_figheight(5)
    fig.set_figwidth(5)

    ax = fig.axes[0]
    ax.set_title('Normal Q-Q')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Standardized Residuals')

    # annotations
    abs_norm_resid = np.argsort(np.abs(model_norm_residuals))
    abs_norm_resid_top_3 = abs_norm_resid[-3:]  # indices of 3 most extreme observations

    for i in abs_norm_resid_top_3:
        ax.annotate(i, xy=(
            QQ.theoretical_quantiles[np.where(QQ.sample_quantiles == model_norm_residuals[i])[0][0]],
            model_norm_residuals[i])
        )

    fig.tight_layout()
    return ax


def pred_vs_actual_plot(model_fit, data):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(5)
    fig.set_figwidth(5)

    fig.axes[0] = sns.regplot(model_fit.predict(), model_fit.model.endog, ci=90,
                    scatter_kws={'alpha': 0.5},)

    ax = fig.axes[0]
    ax.set_title('Predicted vs Actual')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')

    fig.tight_layout()
    return ax
