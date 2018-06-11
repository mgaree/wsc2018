# -*- coding: utf-8 -*-
"""
Module for writing per-iteration data from BatchRunner to csv files and
also later processing data to the trial level.

For writing the raw data:
    - use numbers & one column for each X_j element instead of mashing into a string
    - use one table per trial, which contains all steps & replications

"""

from collections import defaultdict
import pandas as pd


class DataHandler:
    """External data handler class for single batch runner replications.

    Args:
        file_prefix (str): Path & prefix for output filenames, such that each
          output filename takes the form <file_prefix><trial num>_<replication num>.csv.

    """
    def __init__(self, file_prefix):
        self.file_prefix = file_prefix

    def __reduce__(self):
        # Used for pickling by multiproc module
        return (self.__class__, (self.file_prefix, ))

    def handle_single_replication(self, model, model_key):
        """Save raw data arrays from model to csv file.

        Current approach: raw arrays -> pandas -> csv

        Args:
            model (mesa.Model): The model object containing replication data.
            model_key (2-tuple): The (trial number, replication number) pair.

        Returns:
            None.

        """
        trial_num, rep_num = model_key
        model_vars = model.datacollector.model_vars

        model_data = defaultdict(dict)

        for var, records in model_vars.items():
            for step, entry in enumerate(records):
                model_data[step][var] = entry

        sw_model_vars_df = pd.DataFrame.from_dict(model_data, orient='index')
        sw_model_vars_df.to_csv('{}{:03}.{:03}.csv'.format(self.file_prefix,
            trial_num, rep_num), index=True, index_label='Step', float_format='%g')


class DataProcessor:
    pass