import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Neural Network Model with Hidden Layers
class AdvancedNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(AdvancedNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)  # Output matches number of actions
        )

    def forward(self, x):
        return self.model(x)

# Deep Q-Network Agent for Offloading Decision Making
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Stores past experiences
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = AdvancedNN(state_size, action_size)  # DRL model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    # Store past experiences in memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Select an action using ε-greedy policy
    def act(self, state):
        if np.random.rand() <= self.epsilon:  # Exploration
            return random.randrange(self.action_size)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        return torch.argmax(action_values).item()  # Best action

    # Train the model using stored experiences
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                with torch.no_grad():
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                    target += self.gamma * torch.max(self.model(next_state_tensor)).item()

            # Compute Q-value predictions
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            target_f = self.model(state_tensor).clone().detach()  # Get Q-values for current state
            target_f[0, action] = target  # Update the Q-value for the selected action

            # Train the model
            self.optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = self.criterion(output, target_f)
            loss.backward()
            self.optimizer.step()

        # Reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Function to train the DQN agent
def train_model():
    state_size = 5  # Example: state has 5 features (task size, bandwidth, CPU load, etc.)
    action_size = 3  # Example: 3 offloading choices
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = np.random.rand(state_size)  # Replace with real-world data
        done = False
        total_reward = 0

        for time in range(200):
            action = agent.act(state)  # Get action
            next_state = np.random.rand(state_size)  # Simulate next state
            reward = np.random.rand()  # Simulate reward based on offloading efficiency
            done = time == 199  # Episode ends after 200 steps

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
                break

        # Train the agent using replay memory
        agent.replay(batch_size)

# Run the training process
if __name__ == "__main__":
    train_model()
