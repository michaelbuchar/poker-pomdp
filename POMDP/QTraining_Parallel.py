import pandas as pd
import numpy as np
from collections import defaultdict
from multiprocessing import Pool, Manager, cpu_count
import time

# Function to initialize Q-table entries
def initialize_q(actions_count):
    """Initialize the default Q-table entry."""
    return np.zeros(actions_count)

# Function to process a chunk of data
def process_chunk(chunk, action_mapping, learning_rate, discount_factor, state_features, actions_count, worker_id):
    print(f"Worker {worker_id}: Processing {len(chunk)} rows...")
    local_Q = defaultdict(lambda: initialize_q(actions_count))
    for idx, row in chunk.iterrows():
        state = tuple(map(int, row[state_features].values))  # Ensure state is a tuple of Python integers
        next_state = tuple(map(int, row[['Next ' + feature for feature in state_features]].values))  # Ensure the same for next state
        action = row['Action']
        reward = row['Reward']

        # Get action index
        action_idx = action_mapping[action]

        # Update Q-value locally
        max_future_Q = np.max(local_Q[next_state]) if next_state in local_Q else 0
        local_Q[state][action_idx] += learning_rate * (reward + discount_factor * max_future_Q - local_Q[state][action_idx])

    print(f"Worker {worker_id}: Finished processing.")
    return dict(local_Q)

# Function to merge Q-tables
def merge_Q_tables(Q, local_Q, worker_id, actions_count):
    print(f"Merging results from Worker {worker_id}...")
    for state, values in local_Q.items():
        if state not in Q:
            Q[state] = np.zeros(actions_count)  # Initialize if not exists
        Q[state] += values  # Add values from local_Q
    print(f"Worker {worker_id}: Results merged.")

# Main execution block
if __name__ == '__main__':
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
    actions_count = len(actions)

    # Initialize Q-table as a dynamic dictionary
    manager = Manager()
    Q = manager.dict()

    # Hyperparameters
    learning_rate = 0.1
    discount_factor = 0.95
    epochs = 10
    num_workers = cpu_count()

    # Split data into chunks
    chunk_size = len(training_data) // num_workers
    print(f"Data split into {num_workers} chunks of size {chunk_size}.")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs} started...")
        start_time = time.time()

        chunks = [training_data.iloc[i:i + chunk_size] for i in range(0, len(training_data), chunk_size)]

        # Parallel processing
        with Pool(num_workers) as pool:
            results = pool.starmap(
                process_chunk,
                [(chunk, action_mapping, learning_rate, discount_factor, state_features, actions_count, worker_id)
                 for worker_id, chunk in enumerate(chunks)]
            )

        # Merge results back into the shared Q-table
        for worker_id, local_Q in enumerate(results):
            merge_Q_tables(Q, local_Q, worker_id, actions_count)

        end_time = time.time()
        print(f"Epoch {epoch + 1} completed in {end_time - start_time:.2f} seconds.")

    # Generate policy for training states
    print("\nGenerating policy from Q-table...")
    policy = {}
    for state in Q.keys():
        best_action_idx = np.argmax(Q[state])
        best_action = reverse_action_mapping[best_action_idx]
        policy[state] = best_action
    print("Policy generation complete.")

    # Save policy
    output_file = 'complete_policy.policy'
    print(f"Saving policy to {output_file}...")
    with open(output_file, 'w') as f:
        for state, action in policy.items():
            state_str = ", ".join(map(str, state))  # Convert state tuple to a comma-separated string of integers
            f.write(f"{state_str}: {action}\n")
    print(f"Policy saved to '{output_file}'.")
