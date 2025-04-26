import numpy as np
import torch.nn.functional as F
from scipy.stats import entropy

class FL_Environment:
    """Federated Learning Environment for RL-based client selection."""
    def __init__(self, num_clients, global_class_dist):
        self.num_clients = num_clients
        self.global_class_dist = global_class_dist

    def compute_reward(self, prev_acc, new_acc, client_class_dist):
        """Calculate reward based on accuracy improvement and KL divergence."""
        acc_reward = new_acc - prev_acc
        kl_div = entropy(client_class_dist, self.global_class_dist)  # KL-Divergence
        reward = acc_reward / (0.1 * kl_div)  # Normalize KL divergence effect
        return reward

    def step(self, selected_client_indexes, prev_acc, new_acc, client_distributions):
        """Simulate FL training step and compute reward."""
        rewards = [self.compute_reward(prev_acc, new_acc, client_distributions[c]) for c in selected_client_indexes]
        return np.mean(rewards)
