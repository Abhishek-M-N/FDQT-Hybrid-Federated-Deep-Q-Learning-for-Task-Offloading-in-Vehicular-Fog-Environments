import matplotlib.pyplot as plt
import json
import numpy as np

# Load results from file
with open("comparison_results.json", "r") as f:
    averages = json.load(f)

# Define strategy names and styles
strategy_names = ["FDQT", "DRL", "Random", "MWF"]
colors = ["blue", "green", "pink", "red", "orange"]
markers = ["o", "^", "*", "s", "d"]
linestyles = ["solid", "dashdot", "dashed", "dotted", "solid"]

# Extract delay values for each strategy (ensure default value if missing)
delays = [averages.get(name, {}).get("delay", 0) for name in strategy_names]

# X-axis positions for the strategies
x_positions = np.arange(len(strategy_names))

plt.figure(figsize=(10, 5))

# Plot each strategy separately
for i, name in enumerate(strategy_names):
    plt.plot([x_positions[i], x_positions[i]], [0, delays[i]], linestyle=linestyles[i], 
             color=colors[i], linewidth=2, label=name, marker=markers[i], markersize=10)

# Set x-axis labels and format
plt.xticks(x_positions, strategy_names, fontsize=14)

# Labels and title
plt.xlabel("Strategy Name", fontsize=14)
plt.ylabel("Response Delay (s)", fontsize=14)
plt.title("Response Delay Comparison", fontsize=14)

# Place the legend
plt.legend(title="Strategy Name", fontsize=14, loc="upper left", bbox_to_anchor=(0.4, 0.5))

# Grid for better visibility
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
