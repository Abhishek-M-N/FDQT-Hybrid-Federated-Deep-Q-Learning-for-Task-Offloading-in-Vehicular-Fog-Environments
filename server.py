import flwr as fl
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

# Define a simple model
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# Federated Averaging Strategy
strategy = fl.server.strategy.FedAvg()

# Start the FL server
if __name__ == "__main__":
    fl.server.start_server(
        server_address="127.0.0.1:8080",  # Use correct address
        config=fl.server.ServerConfig(num_rounds=10),  # Number of FL training rounds
        strategy=strategy,
    )
