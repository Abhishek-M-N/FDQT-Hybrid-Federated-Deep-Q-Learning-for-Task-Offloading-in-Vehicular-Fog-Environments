import matplotlib.pyplot as plt
import json
import numpy as np

# Load results from file
with open("comparison_results.json", "r") as f:
    averages = json.load(f)

# Define methods and metrics
methods = ["FL", "DRL", "Cloud-Only", "Random", "Greedy"]
metrics = ["reward", "delay", "tasks_processed", "queue_length"]
x_labels = ["Case 1", "Case 2"]  # Adjust based on cases

# Define colors and hatch patterns
colors = ["blue", "pink", "green", "red", "orange"]
hatches = ["\\", "/", "o", ".", "*"]

bar_width = 0.15  # Keeping the same bar width

# Iterate over cases
for case_idx, case in enumerate(x_labels):
    plt.figure(figsize=(14, 8))  # **Reduced height (was 10x6, now 10x4)**
    
    x = np.arange(len(methods))  # X positions for methods
    
    for i, metric in enumerate(metrics):
        values = [averages.get(method, {}).get(metric, 0) for method in methods]
        
        # Adjust X position to avoid overlap
        plt.bar(x + i * bar_width, values, width=bar_width, 
                color=colors[i], hatch=hatches[i], label=metric, edgecolor="black")

        # Add numerical values on bars
        for j, v in enumerate(values):
            plt.text(j + i * bar_width, v + 0.5, str(round(v, 2)), ha="center", fontsize=10)

    plt.xlabel("Methods", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.title(f"Comparison for {case}", fontsize=14)
    plt.xticks(x + (bar_width * (len(metrics) / 2)), methods, fontsize=11)
    plt.legend(title="Metrics", fontsize=10)

    # **Increase grid size for better visibility**
    plt.grid(axis="y", linestyle="--", alpha=0.9, linewidth=1.2)

    plt.show()
