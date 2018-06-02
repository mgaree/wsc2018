# -*- coding: utf-8 -*-
"""
Batch Runner for WSC 2018 paper. Adapts original mesa.batchrunner by:

- Uses DOE design matrix as input instead of full-factorial parameter list
- Uses multiprocessing
- Supports data collection from model's own DataCollector (e.g. step-wise data)
- TODO: write data to sqlite3 db each iteration, instead of writing whole experiment at end

"""

from collections import defaultdict

import pandas as pd
from pathos.multiprocessing import ProcessPool
from randomgen import RandomGenerator, PCG64
from tqdm import tqdm

from mesa.batchrunner import BatchRunner
from mesa.time import RandomActivation

#from pathos.helpers import freeze_support
#freeze_support()  # uncomment if ever have pickle problems on Windows

"""
Note:
     trial == design point
     replication == iteration
     tick == time step

"""


class DOEMultiProcBatchRunner(BatchRunner):
    """Modifies mesa.BatchRunner to support designed experiments and multiprocessing.

    Normally, BatchRunner will run a full factorial design with the same value
    of replications and time steps per trial. This class allows a Pandas data-
    frame specifying the experimental design to be used instead. The inputs
    `variable_parameters` and `fixed_parameters` are ignored, using `design`
    instead.

    If `design` includes the column `replications`, then the `replications`
    value for a trial will set the number of iterations for that trial only.
    If `design` does not include `replications`, self.iterations will be used.

    Multiprocessing design adapted from https://github.com/projectmesa/mesa/pull/456

    Args:
        model_cls: The class of model to batch-run.
        design (str or obj:Pandas.DataFrame): Table defining experimental design,
            or filename of csv file containing design.
            Each row of design is a single trial. Each column (except `trial`)
            is a factor, and the data is the levels. Columns/factors must be valid
            keyword arguments to self.model.__init__().
        collect_stepwise_data (boolean): If True, model.datacollector is called
            each time step (from within the model class; this arg is passed to
            model constructor) and read from model at end of run, in addition
            to calling the model/agent reporters passed to this constructor.
            If False, data is collected only at end of run via the model/agent
            reporters passed to the batch runner constructor.

    See https://github.com/projectmesa/mesa/blob/master/mesa/batchrunner.py
    for further details.

    """
    def __init__(self, model_cls, design, collect_stepwise_data, **kwargs):
        # I don't use variable_parameters for DOE mode, but need to init to avoid error
        kwargs['variable_parameters'] = dict()

        if type(design) is str:
            design = pd.read_csv(design)
        assert 'Trial' in design.columns

        super().__init__(model_cls, **kwargs)

        self.design = design

        self.collect_stepwise_data = collect_stepwise_data
        if self.collect_stepwise_data:
            self.stepwise_vars = {}

        if 'replications' not in self.design.columns:
            # Insert `replications` column into design and set all values to
            # self.iterations.
            self.design.insert(1, 'replications', self.iterations)

    def run_all(self, processors=1, iseed=1):
        """Run the model for all trials in the designed experiment and store results.

        Model constructor is assumed to take args (seed, collect_stepwise_data,
        trial kwargs).

        Args:
            processors (int, default=1): Number of cpu cores to use for batch run.
            iseed (int, default=1): Initial seed for replication 1 of each trial.
                Seeds for subsequent replications will be PNRG by this class.

        """
        pool = ProcessPool(nodes=processors)
        job_queue = []

        # Generator for initial model seeds. Models use these seeds to manage
        # their own RNGs.
        brng = PCG64(iseed)
        randomgen = RandomGenerator(brng)

        param_names = self.design.columns
        param_names = param_names[(param_names != 'replications') & (param_names != 'Trial')]

        total_iterations = self.design.replications.sum()

        self.manifest = []  # Records what seed was used where

        for row in self.design.itertuples():
            kwargs = {key: getattr(row, key) for key in param_names}

            # Reset model seed generator for next design point
            brng.seed(iseed)

            for rep in range(1, row.replications+1):
                # Advance model seed for next replication
                model_seed = randomgen.randint(10000000)

                model_key = (row.Trial, rep,)
                self.manifest.append((model_key, model_seed))
                job_queue.append(pool.uimap(
                    self._run_single_replication, (model_seed,), (kwargs,), (model_key,)))


        with tqdm(total=total_iterations, desc='Total', unit='dp',
                  disable=not self.display_progress) as pbar_total:
            # empty the queue
            results = []
            for task in job_queue:
                for model_vars, agent_vars, stepwise_vars in list(task):
                    results.append((model_vars, agent_vars, stepwise_vars))
                pbar_total.update()

        # store the results in batchrunner
        for model_vars, agent_vars, stepwise_vars in results:
            if self.model_reporters:
                for model_key, model_val in model_vars.items():
                    self.model_vars[model_key] = model_val
            if self.agent_reporters:
                for agent_key, reports in agent_vars.items():
                    self.agent_vars[agent_key] = reports
            if self.collect_stepwise_data:
                for stepwise_key, stepwise_val in stepwise_vars.items():
                    self.stepwise_vars[stepwise_key] = stepwise_val

    def _run_single_replication(self, seed, kwargs, model_key):
        """ Run a single model for one parameter combination and one iteration.

        model_key = (row.Trial id number, replication number) tuple

        """
        model = self.model_cls(
            seed=seed, collect_stepwise_data=self.collect_stepwise_data, **kwargs)
        self.run_model(model)

        model_vars = {}
        agent_vars = {}
        stepwise_vars = {}

        # Collect and store results:
        if self.model_reporters:
            model_vars[model_key] = dict(
                **self.collect_model_vars(model))
        if self.agent_reporters:
            for agent_id, reports in self.collect_agent_vars(model).items():
                agent_key = model_key + (agent_id,)
                agent_vars[agent_key] = reports
        if self.collect_stepwise_data:
            stepwise_vars[model_key] = dict(
                model_vars=model.datacollector.model_vars,
                agent_vars=model.datacollector.agent_vars,
                tables=model.datacollector.tables,
            )

        return (model_vars, agent_vars, stepwise_vars)

    def run_model(self, model):
        """Run a model object to completion, or until reaching max steps."""
        model.schedule.steps = 0
        while model.running and model.schedule.steps < self.max_steps:
            model.step()

    def get_model_instance_dataframes(self):
        """Create model instance dataframes: step-wise model and agent, plus
        tables.

        (Model tables are not explicitly step-wise, but they are one datatable
        per model instance, whereas self.model_vars are one data entry per
        model instance.)

        """
        result = self._prepare_stepwise_report_tables()
#        result['model_instance_tables'] = self._prepare_model_instance_tables()

        return result

    def _prepare_model_instance_tables(self):
        """Join model instance datacollector `tables` with model_key and return
        as dict of dataframes. (`tables` can have multiple tables)

        """
        tables = defaultdict(dict)
        for model_key, dicts in self.stepwise_vars.items():
            for var, table in dicts['tables'].items():
                tables[model_key][var] = pd.DataFrame(table)
        return tables

    def _prepare_stepwise_report_tables(self):
        """Creates 2 dataframes from collected stepwise data (model_vars, agent_vars)
        and sorts them by 'Trial', 'Replication'.

        """
        index_cols = ['Trial', 'Replication', 'Step']
        model_data = defaultdict(dict)
#        agent_data = defaultdict(dict)

        for model_key, dicts in self.stepwise_vars.items():
            for var, records in dicts['model_vars'].items():
                for step, entry in enumerate(records):
                    model_data[(*model_key, step)][var] = entry

#            for var, records in dicts['agent_vars'].items():
#                for step, entries in enumerate(records):
#                    for entry in entries:
#                        agent_id = entry[0]
#                        val = entry[1]
#                        agent_data[(*model_key, step, agent_id)][var] = val

        sw_model_vars_df = pd.DataFrame.from_dict(model_data, orient='index')
        sw_model_vars_df.index.names = index_cols
        sw_model_vars_df.sort_index(level=index_cols, inplace=True)
#
#        sw_agent_vars_df = pd.DataFrame.from_dict(agent_data, orient='index')
#        sw_agent_vars_df.index.names = index_cols + ['AgentID']
#        sw_agent_vars_df.sort_index(level=index_cols + ['AgentID'], inplace=True)

        return dict(sw_model_df=sw_model_vars_df)#, sw_agent_df=sw_agent_vars_df)

    def _prepare_report_table(self, vars_dict, extra_cols=None):
        """Creates a dataframe from collected records and sorts it by 'Trial',
        'Replication'.

        """
        index_cols = ['Trial', 'Replication'] + (extra_cols or [])

        df = pd.DataFrame.from_dict(vars_dict, orient='index')
        df.index.names = index_cols
        df.sort_index(level=index_cols, inplace=True)
        return df


class _TestModel:
    """Test model"""
    def __init__(self, seed, collect_stepwise_data, N, x, y, z):
        from mesa import Agent as Agent
        self.N, self.x, self.y, self.z = N, x, y, z
        self.schedule = RandomActivation(self)
        self.schedule.add(Agent(2, self))
        self.schedule.add(Agent(9, self))

        from mesa.datacollection import DataCollector
        self.datacollector = DataCollector(
            model_reporters={
            'N': lambda m: m.N,
            'X_1': lambda m: m.x,
            'Y_2': lambda m: m.y,
            'X_3': lambda m: m.z,
            },
        agent_reporters={
            'id': lambda a: a.unique_id,
        })
        self.datacollector.collect(self)
        self.step()
        self.datacollector.collect(self)
        self.step()
        self.datacollector.collect(self)

        self.running = True
    def step(self):
        self.schedule.step()

def _test_batchrunner(design):
    # Create DOEBatchrunner using design
    br = DOEMultiProcBatchRunner(
        _TestModel, design, True,
        max_steps=1,
        model_reporters={
            'N': lambda m: m.N,
            'X_1': lambda m: m.x,
            'Y_2': lambda m: m.y,
            'X_3': lambda m: m.z,
            },
        agent_reporters={
            'id': lambda a: a.unique_id,
        })
    br.run_all(processors=2)
    return br

if __name__ == '__main__':
    # Testing: Create Pandas dataframe for design
    factors = ['Trial', 'N', 'x', 'y', 'z']
    design = pd.DataFrame(columns=factors)
    design['Trial'] = ['1.1', '1.2', '2.1']
    design['N'] = [1, 5, 13]
    design['x'] = [.3, .6, .8]
    design['y'] = ['A', 'B', 'C']
    design['z'] = [True, False, False]

    res = _test_batchrunner(design)
    print(res.get_model_vars_dataframe())
    print(res.get_agent_vars_dataframe())

#    res._run_single_replication(1, dict(N=1, x=2, y=3, z=4), (0, 1))
