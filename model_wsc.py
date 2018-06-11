# -*- coding: utf-8 -*-
"""
Model implementation for WSC paper. Designed for DOE design matrix and
proper RNG stream management.

"""

from functools import partial

from enum import IntEnum, auto
import numpy as np
import mesa
from mesa.datacollection import DataCollector
from randomgen import RandomGenerator, PCG64

import schedulers
from agent_wsc import Agent
from network_graphs import get_network_instance


class RNGStream(IntEnum):
    """Convenience enum for RNG stream management."""
    B_COEFFS = auto()
    ERROR_TERM = auto()
    GROUP_ASSIGNMENT = auto()
    UPDATE_SEQUENCING = auto()
    UNINFORMED_AGENT_SELECTION = auto()


class WSCModel:
    """
    Model class for WSC 2018 paper.

    """
    def __init__(self, seed, collect_stepwise_data, N, structure_instance_num, g,
                 randomize_update_seq, b_distro, b_0_is_zero, normalize_bs,
                 error_variance, normalize_yis, uninformed_rate):
        """ Create a new model, configured based on design point inputs.

        Args:
            seed (int): seed for the random number generator. We use multiple
              streams, and each stream will be initialized with this seed, as in
              PCG64(seed=seed, stream=<stream num>).
            collect_stepwise_data (boolean): Indicates if model data should be collected
              each time step.
            N (int): number of agents.
            structure_instance_num (int): Network structure instance number from
              network_graphs.py.
            g (int): number of groups for dividing agents. Must be evenly divided
              by N.
            randomize_update_seq (boolean): If False, groups are updated in same order
              each time step. If True, group update order is randomized each time.
            b_distro (string): Label for distribution function agents use to
              generate their b_ij values. Allowed values are 'U01', 'U-11', and
              'N01'.
            b_0_is_zero (boolean): If True, set b_i0 to 0 for all i. If False,
              b_i0 is sampled from b_distro.
            normalize_bs ('agent', 'network', or 'no'): If 'no', do not
              normalize b_i* values. If 'agent', normalize b_i* within each agent.
              If 'network', normalize b_i* across whole network.
            error_variance (positive float): Variance value for zero-mean
              normal error term.
            normalize_yis (boolean): If True, normalize y_i for all i each time
              step by dividing by sum(y_i). If False, do not normalize y_i.
            uninformed_rate (float): Fraction of agents to be made initially
              uninformed. Must be in interval [0, 1).

        Attributes:
            schedule (obj): Update scheduler for agents.
            running (boolean): Indicates if the model should continue running.
            datacollector (obj:DataCollector): Object for step-wise data
              collection.
            agents (list of Agents): Model agents.
            rg (list of RandomGenerators): Streams for RNG. See RNGStream enum
              for use cases.
            G (NetworkX.DiGraph): Graph structure of network.
            d_i_max (int): Maximum out-degree of agents in network.
            b_distro (obj:RandomGenerator): RNG for b_i* values of agents.
            b_0_is_zero (boolean): Indicates if b_i0 should be 0.
            error_distro (obj:RandomGenerator): RNG for agent error terms.
            normalize_yis (boolean): Indicates if agent y_i values should be
              normalized each time step.
            uninformed_rate (float): Fraction of agents to be made initially
              uninformed.

        """
        # Minor error checking for easy mistakes (larger mistakes are punished)
        if N % g != 0:
            raise Exception()
        if not (0 <= uninformed_rate < 1):
            raise Exception()

        # Prepare random number streams
        self.iseed = seed
        self.rg = [RandomGenerator(PCG64(self.iseed, i+1)) for i in range(len(RNGStream)+1)]

        # Create network
        self.G = get_network_instance(N, structure_instance_num)
        self.d_i_max = np.max(self.G.out_degree, axis=0)[1]

        # Create "reachback" functions for agents to use.
        if b_distro == 'U01':
            self.b_distro = lambda: self.rg[RNGStream.B_COEFFS].uniform(0, 1)
        elif b_distro == 'U-11':
            self.b_distro = lambda: self.rg[RNGStream.B_COEFFS].uniform(-1, 1)
        elif b_distro == 'N01':
            self.b_distro = lambda: self.rg[RNGStream.B_COEFFS].normal(0, 1)
        else:
            raise Exception()

        self.b_0_is_zero = b_0_is_zero
        self.error_distro = lambda: self.rg[RNGStream.ERROR_TERM].normal(0, error_variance)

        # Create agents
        self.agents = [Agent(i, self) for i in range(N)]
        for agent in self.agents:
            agent.meet_neighbors()  # Must defer until all agents created

        # Create scheduler & groups
        # Agents don't need to know what group they're in, so this data is held
        # only by the scheduler.
        update_rng = self.rg[RNGStream.UPDATE_SEQUENCING]
        if g == 1:
            # Add everyone to same group. Ignores `randomize_update_seq`.
            self.schedule = mesa.time.SimultaneousActivation(self)
            self.schedule.agents = list(self.agents)
        elif g == N:
            # Everyone in own group, so don't actually need groups
            self.schedule = schedulers.SingleAgentSimultaneousActivation(
                self, rg=update_rng, random_order=randomize_update_seq)
            self.schedule.agents = list(self.agents)
        else:
            self.schedule = schedulers.GroupedSimultaneousActivation(
                self, randomize_group_order=randomize_update_seq, rg=update_rng)

            cur_group_num = 0
            agent_nums = list(range(N))
            self.rg[RNGStream.GROUP_ASSIGNMENT].shuffle(agent_nums)
            for i in agent_nums:
                self.schedule.add(self.agents[i], cur_group_num)
                cur_group_num = (cur_group_num + 1) % g

        # Normalize values as needed
        self._normalize_bs(normalize_bs)
        self.normalize_yis = normalize_yis

        self.uninformed_rate = uninformed_rate
        self._make_agents_uninformed()

        # Prepare step-wise data collection
        self.collect_stepwise_data = collect_stepwise_data
        if self.collect_stepwise_data:
            self._build_datacollector()

#            # One-time collection of agent constants b_i*; not currently using
#            for agent in self.agents:
#                self.datacollector.add_table_row(
#                    'Initial agent settings',
#                    {'id': agent.unique_id, 'b_i0': agent.b_0, 'b_ij': agent.b_j}
#                )

        self.running = True

    def _build_datacollector(self):
        """Number of X_j values are variable across networks, so data collector
        must be dynamically created.

        """
        model_reporters = {'Y': 'Y'}
        model_reporters.update(
                {'X_{}'.format(i+1): partial(lambda m, i: m.X_j[i], i=i) for i in range(self.d_i_max)})

        # Commenting out because stepwise agent data is not currently needed
#        agent_reporters = {
#            'y_i': 'y_i',
#            'x_ij': lambda a: _nice_X_j_formatter(a.x_ij),  # can unpack list in analysis as needed
#        }
#        # Minimize redundant data by capturing only if informed is non-constant
#        if self.uninformed_rate > 0:
#            agent_reporters['informed'] = lambda a: a.informed

#        tables = {'Initial agent settings': ['id', 'b_i0', 'b_ij']}

        self.datacollector = DataCollector(
            model_reporters=model_reporters,
#            agent_reporters=agent_reporters,
            agent_reporters={},  # zeroizing for now because I don't have a good plan to handle that much data
#            tables=tables,  # skipping for now due to data mgmt failures
        )

    def _make_agents_uninformed(self):
        rate = self.uninformed_rate
        if rate > 0:
            targets = self.rg[RNGStream.UNINFORMED_AGENT_SELECTION].choice(
                self.agents, int(rate*len(self.agents)), replace=False)
            for agent in targets:
                agent.informed = False
                agent.y_i = 0

    def _normalize(self):
        sum_y = 0
        for agent in self.agents:
            # Compute network-wide denominator
            sum_y += agent.y_i
        for agent in self.agents:
            # Update values
            agent.y_i /= sum_y

    def _normalize_bs(self, mode):
        if mode == 'agent':
            for agent in self.agents:
                sum_b = agent.b_0 + agent.b_j.sum()
                agent.b_0 /= sum_b
                agent.b_j = agent.b_j / sum_b
        elif mode == 'network':
            sum_b = 0
            for agent in self.agents:
                # Compute network-wide denominator
                sum_b += agent.b_0 + agent.b_j.sum()
            for agent in self.agents:
                # Update values
                agent.b_0 /= sum_b
                agent.b_j = agent.b_j / sum_b
        else:
            return

    def _prepare_step_data(self):
        """Build Y, X_1, X_2, ..., X_d_i_max before DataCollector collects.

        This is done for efficiency by avoiding d_i_max passes through the
        agent set. Step values are set as model attributes to leverage
        DataCollector's ability to fetch attribute values.

        """
        self.Y = sum([a.y_i for a in self.agents])

        self.X_j = np.zeros(self.d_i_max)
        for agent in self.agents:
            for i, x_j in enumerate(agent.x_ij):
                self.X_j[i] += x_j

    def step(self):
        """A single step of model."""
        self.schedule.step()

        # Collect step-wise data pre-normalization, else Y = 1
        if self.collect_stepwise_data:
            self._prepare_step_data()
            self.datacollector.collect(self)

        if self.normalize_yis:
            self._normalize()


if __name__ == '__main__':
    # If ran as main, run one specific trial
    m = WSCModel(1, True, 100, 1, 100, True, 'U-11', False, 'agent', 0.5, False, 0.5)
    m.step()
