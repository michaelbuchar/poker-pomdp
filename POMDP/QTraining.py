import pandas as pd
import numpy as np
from collections import defaultdict

# Load training data
training_data = pd.read_csv('poker_training_data_large_condensed.csv')

# Define state features and state space ranges
state_features = [
    'Win Bucket', 'Suited', 'Connectedness', 'High Card', 'Flop Status',
    'Paired Board', 'Connected Board', 'Board High Card', 'Count High Cards',
    'Rank of Pair', 'Number of Paired Cards', 'Flush', 'Straight', 'Full House',
    'Pot Size', 'Player Bankroll'
]

# Action mapping
actions = training_data['Action'].unique()
action_mapping = {action: idx for idx, action in enumerate(actions)}
reverse_action_mapping = {idx: action for action, idx in action_mapping.items()}

# Extract states and actions from training data
training_states = training_data[state_features].values
training_actions = training_data['Action'].map(action_mapping).values
training_rewards = training_data['Reward'].values
next_states = training_data[['Next ' + feature for feature in state_features]].values

# Initialize Q-table as a dynamic dictionary
Q = defaultdict(lambda: np.zeros(len(actions)))

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.95
epochs = 10

# Training Q-table
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for idx, row in training_data.iterrows():
        state = tuple(map(int, row[state_features].values))  # Ensure state is a tuple of Python integers
        next_state = tuple(map(int, row[['Next ' + feature for feature in state_features]].values))  # Ensure the same for next state
        action = row['Action']
        reward = row['Reward']

        # Get action index
        action_idx = action_mapping[action]

        # Update Q-value
        max_future_Q = np.max(Q[next_state]) if next_state in Q else 0
        Q[state][action_idx] += learning_rate * (reward + discount_factor * max_future_Q - Q[state][action_idx])

        # Logging
        if idx % 10000 == 0:  # Log progress every 10000 rows
            print(f"Row {idx + 1}/{len(training_data)}: State {state}, Action {action}, Reward {reward:.2f}")

# Generate policy for training states
policy = {}
for state in Q:
    best_action_idx = np.argmax(Q[state])
    best_action = reverse_action_mapping[best_action_idx]
    policy[state] = best_action

# Save policy
with open('complete_policy.policy', 'w') as f:
    for state, action in policy.items():
        state_str = ", ".join(map(str, state))  # Convert state tuple to a comma-separated string of integers
        f.write(f"{state_str}: {action}\n")

print("Policy generated and saved to 'complete_policy.policy'.")
