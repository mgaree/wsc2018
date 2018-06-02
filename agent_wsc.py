# -*- coding: utf-8 -*-
"""
Agent class used by Mesa model for WSC paper.

"""

import numpy as np


class Agent:
    """Base Agent class for regression study.

    Agent's belief value (y_i) is a function of random regression terms and
    the neighbors' influence values.

    Designed for SimultaneousActivation-style scheduling.

    Args:
        unique_id (int): Unique id number associated with instance.
        model (obj:Model): Reference to parent Model.

    Attributes:
        unique_id (int): Unique id number associated with this agent.
        model (obj:Model): Reference to parent Model.
        y_i (float): Belief value.
        b_0 (float): Internal bias.
        b_j (np.array of floats): Coefficients for neighbors belief.
        neighbors (list of Agents): Neighbors of self, based on edges in
          self.model.G of the form (self, other).
        x_ij (np.array of floats): Most recently cached values of neighbors
          belief.

    """
    def __init__(self, unique_id, model):
        """ Create a new agent. """
        self.unique_id = unique_id
        self.model = model

        # Meet the neighbors
        # FUTURE: consider if there's a more efficient way to manage neighbors via iters
        self.neighbor_ids = list(self.model.G.neighbors(self.unique_id))
        self.d_i = len(self.neighbor_ids)

        # Set b_* coefficients
        if self.model.b_0_is_zero:
            self.b_0 = 0
        else:
            self.b_0 = self.model.b_distro()

        self.b_j = np.array([self.model.b_distro() for _ in self.neighbor_ids])

        self.y_i = self.b_0 + self.model.error_distro()  # last term is eps_i
        self._next_y_i = 0

        self.informed = True  # Model will decide if an agent is uninformed

    def meet_neighbors(self):
        self.neighbors = sorted([self.model.agents[nn] for nn in self.neighbor_ids],
                                key=lambda a: a.unique_id)
        del self.neighbor_ids  # free up space

    def get_neighbor_beliefs(self):
        """Return array of belief values for neighbors."""
        return np.array([nn.y_i for nn in self.neighbors])

    def step(self):
        """Update belief level using linear regression equation."""
        self.x_ij = self.get_neighbor_beliefs()

        if not self.informed:
            # Check up on neighbors
            neighbor_status = [nn.informed for nn in self.neighbors]
            if True in neighbor_status:
                self.informed = True
            else:
                return

        eps_i = self.model.error_distro()
        self._next_y_i = self.b_0 + np.dot(self.b_j, self.x_ij) + eps_i

    def advance(self):
        """Complete the SimultaneousActivation-style step."""
        if self.informed is False:
            return

        self.y_i = self._next_y_i
