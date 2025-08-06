import numpy as np
import random
import json
from tabulate import tabulate

# Number of rounds
ROUNDS = 10
NUM_TASKS = 100  # Total tasks per round

# Store results for each method
methods = ["FL", "DRL", "Random", "MWF"]
metrics = ["reward", "delay", "tasks_processed", "queue_length"]
results = {method: {metric: [] for metric in metrics} for method in methods}

# Function to simulate each method
def run_method(method):
    reward = np.random.uniform(50, 100)  # Random reward between 50-100
    delay = np.random.uniform(1, 10)     # Random delay between 1-10
    tasks_processed = random.randint(50, 100)  # Tasks processed
    queue_length = random.randint(0, 50)  # Remaining queue length
    return reward, delay, tasks_processed, queue_length

# Run simulation for each round
for _ in range(ROUNDS):
    for method in methods:
        r, d, tp, ql = run_method(method)
        results[method]["reward"].append(r)
        results[method]["delay"].append(d)
        results[method]["tasks_processed"].append(tp)
        results[method]["queue_length"].append(ql)

# Compute averages
averages = {method: {metric: np.mean(results[method][metric]) for metric in metrics} for method in methods}

# Print numerical results as a table
table_data = []
for method in methods:
    table_data.append([
        method,
        round(averages[method]["reward"], 2),
        round(averages[method]["delay"], 2),
        round(averages[method]["tasks_processed"], 2),
        round(averages[method]["queue_length"], 2),
    ])

headers = ["Method", "Avg Reward", "Avg Delay", "Tasks Processed", "Queue Length"]
print("\n📊 **Comparison of Methods (Numerical Results)**\n")
print(tabulate(table_data, headers=headers, tablefmt="grid"))

# Save results as JSON for plotting
with open("comparison_results.json", "w") as f:
    json.dump(averages, f, indent=4)

print("\n✅ Simulation completed! Results saved to comparison_results.json")
