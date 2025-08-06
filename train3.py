import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv

# Sample neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# Initialize model, loss function, and optimizer
model = SimpleNN(input_size=10, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store training metrics
losses = []
task_delays = []
queue_sizes = []
rewards = []

# Training loop with logging
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()

    # Sample input and target
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Handle NaN values in loss
    if torch.isnan(loss):
        print(f"Warning: NaN detected in loss at epoch {epoch}")
        loss = torch.nan_to_num(loss, nan=0.0)

    # Backward pass
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    # Simulated additional metrics (replace with actual calculations if available)
    task_delay = np.random.uniform(1, 10)  # Simulated task delay
    queue_size = np.random.randint(0, 50)  # Simulated queue congestion
    reward = np.random.uniform(-5, 10)  # Simulated reward

    # Store values
    losses.append(loss.item())
    task_delays.append(task_delay)
    queue_sizes.append(queue_size)
    rewards.append(reward)

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Reward = {reward:.4f}, Queue Size = {queue_size}, Task Delay = {task_delay:.4f}")

# Save logs for analysis
with open("training_logs.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Loss", "Reward", "Queue_Size", "Task_Delay"])
    for i in range(num_epochs):
        writer.writerow([i+1, losses[i], rewards[i], queue_sizes[i], task_delays[i]])

print("Training complete! Logs saved to training_logs.csv")
