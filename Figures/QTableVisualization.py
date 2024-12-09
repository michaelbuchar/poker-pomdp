# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file (first 100 rows only)
file_path = "QTableVisualization.csv"  # Replace with your actual file path
data = pd.read_csv(file_path)

# Extract the last 7 columns and limit to the first 100 rows
action_columns = data.columns[-6:]
action_data = data.iloc[:300, -6:]

# Create a plot
fig, ax = plt.subplots(figsize=(10, 6))

# Create a colormap
cmap = plt.cm.viridis

# Normalize the action values for colormap
norm = plt.Normalize(action_data.min().min(), action_data.max().max())

# Plot the heatmap
c = ax.imshow(action_data.T, aspect='auto', cmap=cmap, norm=norm)

# Set axis labels
ax.set_xticks(np.arange(action_data.shape[0]))
ax.set_xticklabels([f"Row {i}" for i in range(1, action_data.shape[0] + 1)], rotation=90)
ax.set_yticks(np.arange(len(action_columns)))
ax.set_yticklabels(action_columns)

# Add a color bar
plt.colorbar(c, ax=ax, label="Action Value")

# Title and labels
ax.set_title("Action Values Heatmap")
ax.set_ylabel("Actions")
ax.set_xticklabels([])
ax.set_xlabel("States")
# Show the plot
plt.tight_layout()
plt.show()
