# -*- coding: utf-8 -*-
"""
Process raw simulation data using multiprocessing:
    take Y, X_j values for each time step of a trial (across replications)
    remove outliers & compute linear regression model using OLS
    compute moving average statistics
    save data frame to csv for trial

"""

# Spyder patch to allow multiprocessing to work from within Spyder
__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"


def process_trial(trial_num):
    # Due to an apparent bug in pathos (or induced by Spyder?) I need to put my globals inside the function...
    import pandas
    import numpy as np
    import statsmodels.formula.api as smf

    data_filename_template = 'C:/Users/Mike/Desktop/tmp/wsc_data/trial_data/{:03}.{:03}.csv'
    num_replications = 100
    num_steps = 500

    # Build trial data table from each replication file
    trial_df = pandas.read_csv(data_filename_template.format(trial_num, 1))
    for replication in range(2, num_replications+1):
        trial_df = trial_df.append(pandas.read_csv(data_filename_template.format(trial_num, replication)), ignore_index=True)

    trial_df = trial_df.replace([np.inf, -np.inf], np.nan)

    # Build model strings for linear and quadratic fits
    columns = [item for item in trial_df.columns if item not in ('Step', 'Y')]
    linear_model = 'Y ~ {}'.format(' + '.join(columns))

    df_cols = ['Step', 'R2', 'R2 Adj.', 'p-value', 'Max Corr_Y', 'Max Y', 'Num Outliers Dropped']
    results = np.zeros((num_steps, len(df_cols)))

    #
    # OLS Regression
    # Fit model to each step from across trial replications
    #

    # Some trials have extreme outliers in Y, which kills the regression. Mask the outliers
    # and capture the quantity of outliers in the results data. Dropping outliers based on the
    # standard 1.5 times inter-quartile range (IQR) rule.

    for step in range(num_steps):
        step_df = trial_df.query('Step == {}'.format(step))

        try:
            # Computing IQR
            Q1 = step_df.Y.quantile(0.25)
            Q3 = step_df.Y.quantile(0.75)
            IQR = Q3 - Q1

            # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
            keep = step_df.query('(@Q1 - 1.5 * @IQR) <= Y <= (@Q3 + 1.5 * @IQR)')
            num_outliers_dropped = len(step_df) - len(keep)
            step_df = keep

            linear_fit = smf.ols(formula=linear_model, data=step_df).fit()

            lin_R = linear_fit.rsquared
            lin_R_adj = linear_fit.rsquared_adj
            lin_p = linear_fit.f_pvalue
        except:
            lin_R, lin_R_adj, lin_p = [None, None, None]
            num_outliers_dropped = np.nan

        # These can inform really high condition numbers, but are not explicitly used
        max_Y_corr = step_df.corr()['Y'][2:].max()
        max_Y = step_df['Y'].max()

        # I'm doing step+1 here to make the rest of analysis cleaner; raw data is recorded
        # as Step 0, but we're all thinking in Step 1.
        results[step, :] = [step+1, lin_R, lin_R_adj, lin_p, max_Y_corr, max_Y, num_outliers_dropped]

    trial_result = pandas.DataFrame(results, columns=df_cols)
    trial_result.set_index('Step', inplace=True)
    trial_result.index = trial_result.index.astype('int')  # To get integer formatting

    # Moving average calculation
    trial_result.insert(3, 'MA-R2 Adj.', trial_result['R2 Adj.'].rolling(window=5, center=False).mean())

    trial_result.insert(4, 'MA-p-value Significant', compute_p_value_rolling_value(trial_result['p-value']))
    #trial_result.insert(4, 'MA-p-value', trial_result['p-value'].rolling(window=5, center=False).mean())

    trial_result.to_csv('C:/Users/Mike/Desktop/tmp/wsc_data/trial_step_results/{:03}_results.csv'.format(trial_num),
                        float_format='%g')
    return


def compute_p_value_rolling_value(data):
    """Compute the rolling ("moving average") significance of the p-value.

    We only care if the p-value is significant or not; the absolute value is not
    so important. With simple moving average, a random high value can overly bias
    a series of very low values and may unfairly make a trial Not Linear.

    Instead, do the following:
        - at t = 1, set MA-p-value = Insignificant
        - switch to Significant only after Z sequential observations of
          significant p-values
        - switch back to Insignificant after Z sequential insignificant p-values
    In short, after observing Z good (bad) p-values, declare the current series
    Significant (Insignificant).

    For this paper, set Z = RUN_LENGTH_TO_SWITCH = 3.

    """
    import numpy as np
    data = list(data)

    SIGNIFICANT = 1  # These are binary flags, not p-values
    NOT_SIGNIFCANT = 0

    p_threshold = 0.10  # Greater than this, Not Significant, else Significant
    # nan/inf always Not Significant

    # First classify each data point as significant or not (to simplify smoothing process)
    for i, d in enumerate(data):
        if np.isfinite(d) and d <= p_threshold:
            data[i] = SIGNIFICANT
        else:
            data[i] = NOT_SIGNIFCANT

    # Now apply smoothing, initialized at NOT_SIGNIFICANT (consistent w/ null hypothesis)
    current = NOT_SIGNIFCANT

    run_length = 0
    RUN_LENGTH_TO_SWITCH = 3

    res = np.zeros(len(data))  # Makes res[0] = NOT_SIGNIFICANT by default
    for i, d in enumerate(data):
        if d == current:  # We only need to track things if they could make us switch
            run_length = 0
        else:
            run_length += 1

            if run_length >= RUN_LENGTH_TO_SWITCH:
                # Switch label we're applying and reset counter
                current = SIGNIFICANT if (current == NOT_SIGNIFCANT) else NOT_SIGNIFCANT
                run_length = 0

        res[i] = current

    return res


if __name__ == "__main__":
    # Batch process + multiprocessing
    from tqdm import tqdm

    from pathos.multiprocessing import ProcessPool
    pool = ProcessPool(nodes=3)
    job_queue = []

    for trial_num in range(1, 256):
        job_queue.append(pool.uimap(process_trial, (trial_num,)))

    # Code block to process R & p values for all trials
    with tqdm(total=255, unit='trials') as pbar:
        for task in job_queue:
            list(task)
            pbar.update()
