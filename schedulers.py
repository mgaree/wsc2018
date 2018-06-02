# -*- coding: utf-8 -*-
"""
Agent update schedulers.
"""

from collections import defaultdict

from mesa.time import BaseScheduler


class GroupedSimultaneousActivation(BaseScheduler):
    """Agents are members of a group. Each group is activated with simultaneous
    activations. Groups are activated in fixed or random order based on init
    settings.

    """
    def __init__(self, model, randomize_group_order=False, rg=None):
        super().__init__(model)
        self.groups = defaultdict(list)
        self.randomize_group_order = randomize_group_order
        self.rg = rg  # Randomizer for shuffling agents if needed

    def add(self, agent, group):
        """Add agent to schedule and specified group."""
        self.agents.append(agent)
        self.groups[group].append(agent)

    def remove(self, agent):
        """Remove agent from schedule and all groups."""
        while agent in self.agents:
            self.agents.remove(agent)
        for k, group in self.groups.items():
            while agent in group:
                self.groups[k].remove(agent)

    def step(self):
        """For each group, step all agents, then advance them. (Simultaneous activation)"""
        group_order = list(self.groups.keys())
        if self.randomize_group_order:
            self.rg.shuffle(group_order)
        else:
            group_order.sort()  # dict.keys() does not guarantee a particular ordering

        for group_num in group_order:
            # Slice-copy is used to allow agents to leave the system
            for agent in self.groups[group_num][:]:
                agent.step()
            for agent in self.groups[group_num][:]:
                agent.advance()

        self.steps += 1
        self.time += 1


class SingleAgentSimultaneousActivation(BaseScheduler):
    """Activate SimultaneousActivation-style agents one at a time, in fixed
    or random order.

    """
    def __init__(self, model, rg=None, random_order=False):
        super().__init__(model)
        self.rg = rg
        self.random_order = random_order

    def step(self):
        if self.random_order:
            self.rg.shuffle(self.agents)

        for agent in self.agents[:]:
            agent.step()
            agent.advance()
        self.steps += 1
        self.time += 1
