import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random

# Define Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# Initialize Model
input_size, output_size = 10, 1
model = SimpleNN(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define offloading strategies
def random_offloading():
    return random.choice(["Cloud", "RSU", "Vehicle"])

def cloud_only_offloading():
    return "Cloud"

def greedy_offloading(queue_status):
    return min(queue_status, key=queue_status.get)  # Choose the least congested node

# Simulation Settings
epochs = 10
nodes = ["Cloud", "RSU", "Vehicle"]
queue_status = {"Cloud": 50, "RSU": 20, "Vehicle": 10}  # Initial congestion levels

# Store logs
logs = []

# Training loop for all methods
for epoch in range(epochs):
    for method in ["FDRL", "Random", "Cloud-Only", "Greedy"]:
        optimizer.zero_grad()

        # Sample inputs and targets
        inputs = torch.randn(5, input_size)
        targets = torch.randn(5, output_size)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Handle NaN values
        if torch.isnan(loss):
            loss = torch.nan_to_num(loss, nan=0.0)

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Simulate different offloading approaches
        if method == "Random":
            selected_node = random_offloading()
        elif method == "Cloud-Only":
            selected_node = cloud_only_offloading()
        elif method == "Greedy":
            selected_node = greedy_offloading(queue_status)
        else:  # DRL
            selected_node = random.choice(nodes)  # Simulating a trained model’s decision

        # Simulate performance metrics
        task_delay = np.random.uniform(1, 10) if selected_node != "Cloud" else np.random.uniform(5, 15)
        queue_size = queue_status[selected_node] + np.random.randint(-5, 5)
        reward = np.random.uniform(-1, 10)
        offloading_efficiency = np.random.uniform(0.5, 1.0)

        # Update queue status (simulate dynamic changes)
        queue_status[selected_node] = max(0, queue_size)

        # Log results
        logs.append([epoch + 1, method, loss.item(), reward, queue_size, task_delay, offloading_efficiency])

        print(f"Epoch {epoch+1} | {method}: Loss={loss.item():.4f}, Reward={reward:.4f}, Queue={queue_size}, Delay={task_delay:.4f}")

# Save logs to CSV
df = pd.DataFrame(logs, columns=["Epoch", "Method", "Loss", "Reward", "Queue_Size", "Task_Delay", "Offloading_Efficiency"])
df.to_csv("training_comparison_logs.csv", index=False)

print("Training complete! Logs saved to training_comparison_logs.csv")
