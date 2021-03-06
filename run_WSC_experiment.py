# -*- coding: utf-8 -*-
"""
Driver for simulation experiment for WSC 2018 paper.
"""

from datetime import datetime
import pandas as pd
import pickle

from DOEMultiProcBatchRunner import DOEMultiProcBatchRunner
from model_wsc import WSCModel
from data_handler import DataHandler


design_filename = 'design_matrix_WSC2018.csv'
output_file_prefix = 'C:/Users/Mike/Desktop/tmp/wsc_data/trial_data/'
collect_stepwise_data = True


def run_experiment(test_run=True):
    # If test_run, do only a few time steps and 2 replications
    if test_run:
        replications = 2
        max_steps = 10
    else:
        replications = 100  # 80:400 is about as much as my 1TB drive can hold of agent step-data
        max_steps = 500

    br = DOEMultiProcBatchRunner(
        WSCModel,
        design_filename,
        collect_stepwise_data,
        iterations=replications,
        max_steps=max_steps,
        data_handler=DataHandler(output_file_prefix),
        # WSCModel uses stepwise datacollection, so don't need br-level
        # agent_ and model_ reporters
        )
    br.run_all(processors=3, iseed=1)  # using all 4 cores leads to system instability
    return br


def run_trial_one_rep(trial_num, seed, test_run=True):
    design = pd.read_csv(design_filename)
    row = design.loc[trial_num-1]  # index is base-0, but Trials is base-1
    params = {**row}
    del params['Trial']

    if test_run:
        max_steps = 3
    else:
        max_steps = 500

    m = WSCModel(seed, True, **params)
    for _ in range(max_steps):
        m.step()

    return m


if __name__ == '__main__':
    print("{} - Experiment started".format(str(datetime.now())))

    res = run_experiment(test_run=True)

    print("{} - Batch run complete".format(str(datetime.now())))

    # I don't have a current need for this data, so just get it saved somewhere
#    with open(output_file_prefix+'tables.pickle.txt', 'wb') as f:
#        pickle.dump(datadict['model_instance_tables'], f)

    # Manifest tracks what RNG seeds were used when
    with open(output_file_prefix+'manifest.pickle.txt', 'wb') as f:
        pickle.dump(res.manifest, f)

    print("{} - Data saved".format(str(datetime.now())))
    print()
    print("Complete.")
