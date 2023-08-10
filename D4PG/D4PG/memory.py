import numpy as np
import os
from laserhockey.hockey_env import *
import torch
import random

# class to store transitions
class Memory():
    def __init__(self, max_size=100000):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size=max_size

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx,:] = np.asarray(transitions_new, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch_size=1):
        if batch_size > self.size:
            batch_size = self.size
        self.inds=np.random.choice(range(self.size), size=batch_size, replace=False)
        return self.transitions[self.inds,:]

    def get_all_transitions(self):
        return self.transitions[0:self.size]

class PrioritizedExpReplay:
    """
    The ExperienceReplay class implements a base class for an experience replay buffer.

    Parameters
    ----------
    max_size : int
        Specifies the maximum number of (state, action, reward, new_state, done) tuples in the buffer.
    """

    def __init__(self, eps=0.01, alpha=0.6, beta=0.4, max_size=100000):
        # Use a list to store transitions for flexibility
        self._transitions = np.asarray([])
        self._current_idx = 0
        self.size = 0
        self.max_size = max_size

        self.eps = eps  # small amount to avoid zero priority
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # SumTree and MinTree for efficient sampling
        self.priority_max = 1
        self.sum_tree = SumTree(max_size)

    @staticmethod
    def clone_buffer(new_buffer, max_size):
        # Create a new buffer and copy old transitions
        old_transitions = new_buffer._transitions.copy()
        buffer = Memory(max_size=max_size)
        for t in old_transitions:
            buffer.add_transition(t)

        return buffer

    def add_transition(self, transitions_new):

        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self._current_idx,:] = np.asarray(transitions_new, dtype=object)
        self._current_idx = (self._current_idx + 1) % self.max_size

        # Update the SumTree
        #self.sum_tree.add(self.priority_max ** self.alpha, self._current_idx)
        self.sum_tree.add(self.priority_max, self._current_idx)

        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=128):

        if batch_size > self.size:
            batch_size = self.size

        sample_idx_lst, tree_idx_lst = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        segment = self.sum_tree.total / batch_size
        for inp in range(batch_size):
            a, b = segment * inp, segment * (inp + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.sum_tree.get(cumsum)

            priorities[inp] = priority
            tree_idx_lst.append(tree_idx)
            sample_idx_lst.append(sample_idx)

        probs = priorities / self.sum_tree.total
        weights = (self.size * probs) ** -self.beta
        weights = weights / weights.max()

        sample_idx_lst = [np.random.choice(self.size) if v is None else v for v in sample_idx_lst]
        #print("sample_idx_lst", sample_idx_lst)
        return self.transitions[sample_idx_lst, :], weights, tree_idx_lst

    def update_priorities(self, tree_idx_lst, priorities):

        for idx, priority in zip(tree_idx_lst, priorities):
            self.sum_tree.update(idx, (priority + self.eps) ** self.alpha)
            self.priority_max = max(self.priority_max, priority)



class SumTree:
    """
    A binary sum tree data structure for efficient cumulative sum calculations.
    """

    def __init__(self, max_size):
        """
        Initialize a SumTree instance.

        Args:
            max_size (int): The maximum number of elements the sum tree can hold.
        """
        self.tree_nodes = [0] * (2 * max_size - 1)  # Stores the sum tree nodes
        self.data = [None] * max_size  # Stores associated data

        self.max_size = max_size
        self.current_count = 0  # Current count of added elements
        self.real_size = 0  # Actual number of added elements

    @property
    def total(self):
        """
        Get the total sum stored at the root of the sum tree.

        Returns:
            int: The total sum.
        """
        return self.tree_nodes[0]

    def update(self, data_idx, value):
        """
        Update a value in the sum tree and propagate the changes upward.

        Args:
            data_idx (int): Index of the data to be updated.
            value (int): New value to update.

        """
        idx = data_idx + self.max_size - 1  # Index of the leaf node
        change = value - self.tree_nodes[idx]  # Calculate the change to propagate

        self.tree_nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            # Update the parent nodes along the path to the root.
            self.tree_nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        """
        Add a new value and its associated data to the sum tree.

        Args:
            value (int): Value to be added.
            data: Associated data to be stored.

        """
        self.data[self.current_count] = data
        self.update(self.current_count, value)

        self.current_count = (self.current_count + 1) % self.max_size
        self.real_size = min(self.max_size, self.real_size + 1)

    def get(self, target_sum):
        """
        Retrieve data and index based on a target cumulative sum.

        Args:
            target_sum (int): The target cumulative sum.

        Returns:
            Tuple[int, int, Any]: A tuple containing data index, tree node value, and associated data.

        """
        assert target_sum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.tree_nodes):
            left, right = 2 * idx + 1, 2 * idx + 2

            if target_sum <= self.tree_nodes[left]:
                idx = left
            else:
                idx = right
                target_sum = target_sum - self.tree_nodes[left]

        data_idx = idx - self.max_size + 1

        return data_idx, self.tree_nodes[idx], self.data[data_idx]


def custom_reward(transition):

    dist_to_puck = compute_dist_to_puck(transition)
    new_reward = transition[2] + dist_to_puck

    transition = (transition[0], transition[1], new_reward, transition[3], transition[4])

    return transition

def compute_dist_to_puck(transition):
    # Extract observation from the transition
    observation = np.asarray(transition[0])

    # Initialize the reward for closeness to puck
    reward_closeness_to_puck = 0

    # Check if the agent is in its own half and the puck is behind
    if (observation[-6] + CENTER_X) < CENTER_X and observation[-4] <= 0:
        # Calculate distance to puck
        agent_position = observation[:2]
        puck_position = observation[-6:-4]
        dist_to_puck = dist_positions(agent_position, puck_position)

        # Define maximum distance and maximum reward for proxy
        max_dist = 250. / SCALE
        max_reward = -30.  # Max (negative) reward through this proxy

        # Calculate reward factor based on distance
        factor = max_reward / (max_dist * 250 / 2)

        # Add proxy reward for being close to the puck in the agent's own half
        reward_closeness_to_puck += dist_to_puck * factor

    return reward_closeness_to_puck