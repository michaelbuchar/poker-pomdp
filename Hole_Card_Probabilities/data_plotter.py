import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Helper function to rank cards
rank_map = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14
}

def get_card_rank(card):
    return rank_map[card[0]]

# Load the data
data = pd.read_csv("hole_card_simulation_results.csv")

# Extract ranks for coloring
def extract_card_ranks(hole_cards):
    cards = hole_cards.split(" ")
    rank1 = get_card_rank(cards[0][1:])
    rank2 = get_card_rank(cards[1][1:])
    return rank1, rank2

ranks = data['Hole cards'].apply(extract_card_ranks)
card1_ranks = ranks.apply(lambda x: x[0])
card2_ranks = ranks.apply(lambda x: x[1])

# Duplicate data for permutations
data_permuted = data.copy()
data_permuted['Hole cards'] = data['Hole cards'].apply(lambda x: " ".join(x.split(" ")[::-1]))
ranks_permuted = data_permuted['Hole cards'].apply(extract_card_ranks)
card1_ranks_permuted = ranks_permuted.apply(lambda x: x[0])
card2_ranks_permuted = ranks_permuted.apply(lambda x: x[1])

# Combine original and permuted data
card1_ranks = pd.concat([card1_ranks, card1_ranks_permuted], ignore_index=True)
card2_ranks = pd.concat([card2_ranks, card2_ranks_permuted], ignore_index=True)
probs = pd.concat([data[['Win prob', 'Loss prob', 'Split prob']], data_permuted[['Win prob', 'Loss prob', 'Split prob']]], ignore_index=True).values
probs /= probs.sum(axis=1, keepdims=True)

# Compute average win probabilities and standard deviations for each rank pair
unique_combinations = pd.DataFrame({
    "Rank1": card1_ranks,
    "Rank2": card2_ranks,
    "WinProb": probs[:, 0]
})
grouped = unique_combinations.groupby(["Rank1", "Rank2"]).agg(
    AvgWinProb=("WinProb", "mean"),
    StdWinProb=("WinProb", "std")
).reset_index()

# Create a 3D bar chart with solid bars
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Bar positions
x = grouped['Rank1']
y = grouped['Rank2']
z = np.zeros_like(x)

dx = dy = 0.8  # Width of the bars
dz = grouped['AvgWinProb']  # Height of the bars (average win probability)

# Plot solid bars
colors = plt.cm.viridis(grouped['AvgWinProb'] / grouped['AvgWinProb'].max())
ax.bar3d(x, y, z, dx, dy, dz, color=colors, edgecolor='k', alpha=1.0)

# Labels and adjustments
ax.set_title("3D Bar Chart: Rank1, Rank2 vs Average Win Probability")
ax.set_xlabel("Rank 1")
ax.set_ylabel("Rank 2")
ax.set_zlabel("Average Win Probability")

plt.show()
