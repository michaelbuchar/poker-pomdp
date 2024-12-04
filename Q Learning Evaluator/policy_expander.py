import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from itertools import product
from multiprocessing import Pool, cpu_count
import os
import time

# Define the state space ranges
state_space_ranges = {
    'Win Bucket': [1, 2, 3],
    'Suited': [0, 1],
    'Connectedness': [0, 1],
    'High Card': [0, 1],
    'Flop Status': [0, 1, 2, 3],
    'Paired Board': [0, 1],
    'Connected Board': [0, 1],
    'Board High Card': [0, 1, 2],
    'Count High Cards': [0, 1, 2, 3, 4, 5],
    'Rank of Pair': [0, 1, 2, 3],
    'Number of Paired Cards': [0, 1, 2, 3],
    'Flush': [0, 1, 2],
    'Straight': [0, 1, 2],
    'Full House': [0, 1],
    'Pot Size': list(range(6)),
    'Player Bankroll': list(range(6))
}

# Load the sparse policy from the file
sparse_policy_file = 'complete_policy.policy'
output_file = 'expanded_policy.policy'

# Read sparse policy
print("Loading sparse policy...")
start_time = time.time()
sparse_policy = {}
policy_states = []
policy_actions = []

with open(sparse_policy_file, 'r') as f:
    for line in f:
        state, action = line.strip().split(": ")
        state_tuple = tuple(map(int, state.split(", ")))
        sparse_policy[state_tuple] = action
        policy_states.append(state_tuple)
        policy_actions.append(action)

policy_states = np.array(policy_states)
print(f"Sparse policy loaded with {len(sparse_policy)} states in {time.time() - start_time:.2f} seconds.")

# Build KDTree for nearest neighbor lookup
print("Building KDTree for sparse policy...")
start_time = time.time()
kdtree = KDTree(policy_states)
print(f"KDTree built in {time.time() - start_time:.2f} seconds.")

# Generate the state space keys and values
state_keys = list(state_space_ranges.keys())
state_value_arrays = [np.array(values) for values in state_space_ranges.values()]
num_states = np.prod([len(values) for values in state_value_arrays])

# Batch configuration
batch_size = 10000
num_batches = (num_states + batch_size - 1) // batch_size

print(f"Total states to process: {num_states}")
print(f"Processing in {num_batches} batches of size {batch_size}...\n")

# Function to process a batch
def process_batch(batch_range):
    batch_start, batch_end = batch_range
    print(f"Processing batch {batch_start // batch_size + 1}/{num_batches}...")

    # Generate indices for this batch
    batch_indices = np.unravel_index(
        range(batch_start, batch_end),
        [len(v) for v in state_value_arrays]
    )

    # Map indices to state values
    batch_states = [
        tuple(state_value_arrays[i][idx] for i, idx in enumerate(batch))
        for batch in zip(*batch_indices)
    ]

    # Find the nearest neighbor for each state in the batch
    distances, neighbor_indices = kdtree.query(batch_states)

    # Write results incrementally
    with open(output_file, 'a') as f:
        for state, neighbor_idx in zip(batch_states, neighbor_indices):
            f.write(
                f"{', '.join(map(str, state))}: {policy_actions[neighbor_idx]}\n"
            )

    print(f"Batch {batch_start // batch_size + 1}/{num_batches} completed.")

# Main script
if __name__ == "__main__":
    # Clear the output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Generate batch ranges
    batch_ranges = [
        (i * batch_size, min((i + 1) * batch_size, num_states))
        for i in range(num_batches)
    ]

    # Use multiprocessing to process batches in parallel
    start_time = time.time()
    with Pool(cpu_count()) as pool:
        print(f"Processing {num_batches} batches in parallel with {cpu_count()} workers...\n")
        pool.map(process_batch, batch_ranges)

    print(f"\nExpanded policy saved to {output_file}.")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")
