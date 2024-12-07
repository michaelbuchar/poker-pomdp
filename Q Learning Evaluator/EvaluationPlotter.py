import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV files
baseline_data = pd.read_csv("BaselineResults.csv")
q_learning_data = pd.read_csv("Q_learningResults.csv")
Smoothing_data = pd.read_csv("SmoothingResults.csv")

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot Baseline data
ax.plot(baseline_data["Game"], baseline_data["Wins"], label="Baseline Wins", marker="o", linestyle="-")
ax.plot(baseline_data["Game"], -baseline_data["Losses"], label="Baseline Losses", marker="o", linestyle="--")

# Plot Q-learning data
ax.plot(q_learning_data["Game"], q_learning_data["Wins"], label="Q-Learning Nearest Neighbor Wins", marker="o", linestyle="-")
ax.plot(q_learning_data["Game"], -q_learning_data["Losses"], label="Q-Learning Nearest Neighbor Losses", marker="o", linestyle="--")

ax.plot(Smoothing_data["Game"], Smoothing_data["Wins"], label="Q-Learning Kernel Smoothing Wins", marker="o", linestyle="-")
ax.plot(Smoothing_data["Game"], -Smoothing_data["Losses"], label="Q-Learning Kernel Smoothing Losses", marker="o", linestyle="--")

# Add horizontal line at zero for reference
ax.axhline(0, color="black", linewidth=1, linestyle="--")

# Set x-axis limit
ax.set_xlim(0, 250)

# Add axis labels and title
ax.set_xlabel("Games Played")
ax.set_title("Wins and Losses Across Games (Baseline vs Q-Learning)")
ax.legend()

# Customize gridlines
ax.grid(visible=True, which='major', linestyle='-', linewidth=0.75)  # Major gridlines
ax.grid(visible=True, which='minor', linestyle=':', linewidth=0.5)   # Minor gridlines
ax.minorticks_on()

# Adjust the "Wins" label position
ax.annotate(
    "Wins",
    xy=(0, max(baseline_data["Wins"].max(), q_learning_data["Wins"].max()) * 0.3),  # Closer to the axis
    xytext=(-40, max(baseline_data["Wins"].max(), q_learning_data["Wins"].max()) * 0.3),  # Adjusted x offset
    textcoords="offset points",
    fontsize=16,
    color="green",
    fontweight = 'bold',
    rotation=90,
    ha="center"
)

# Adjust the "Losses" label position
ax.annotate(
    "Losses",
    xy=(0, -max(baseline_data["Losses"].max(), q_learning_data["Losses"].max()) * 0.3),  # Closer to the axis
    xytext=(-40, -max(baseline_data["Losses"].max(), q_learning_data["Losses"].max()) * 0.3),  # Adjusted x offset
    textcoords="offset points",
    fontsize=16,
    fontweight = 'bold',
    color="red",
    rotation=90,
    ha="center"
)

# Adjust layout to prevent cutoff
plt.tight_layout()
plt.show()
