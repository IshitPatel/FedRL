import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class DQN(nn.Module):
    """Neural Network for Q-learning."""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, state):
        return self.fc2(torch.relu(self.fc1(state)))

class DQN_Agent:
    """DQN Agent for Client Selection."""
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.epsilon = 0.5

    def select_clients(self, state, num_clients):
        """Select clients using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            return random.sample(range(num_clients), k=3)
        with torch.no_grad():
            q_values = self.model(torch.tensor(state, dtype=torch.float32))
            return q_values.argsort(descending=True)[:3].tolist()

    def train(self, state, action, reward, next_state):
        """Train the Q-network."""
        state, next_state = torch.tensor(state, dtype=torch.float32), torch.tensor(next_state, dtype=torch.float32)
        state = torch.tensor(state, dtype=torch.float32)  # Convert to float32
        action = torch.tensor(action, dtype=torch.long)   # Actions should be long for indexing
        reward = torch.tensor(reward, dtype=torch.float32) # Convert reward to float32
        next_state = torch.tensor(next_state, dtype=torch.float32)
        target_q = reward + 0.9 * self.target_model(next_state).max().item()
        q_value = self.model(state)[action]
        
        loss = self.loss_fn(q_value, torch.tensor(target_q))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
