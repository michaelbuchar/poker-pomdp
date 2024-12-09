import os
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


def load_results_from_csv(csv_filename="simulation_results.csv"):
    df = pd.read_csv(csv_filename)
    results = defaultdict(list)
    for action in df['Action'].unique():
        results[action] = df[df['Action'] == action]['Updated Belief State'].tolist()
    return results


def T(s, a, s_prime):
    if a in ["Raise Big"]:
        base_probability = 0.9 if s_prime > s else 0.2
    elif a in ["Bet Big"]:
        base_probability = 0.85 if s_prime > s else 0.3
    elif a in ["Raise Small"]:
        base_probability = 0.7 if s_prime > s else 0.4
    elif a in ["Bet Small"]:
        base_probability = 0.6 if s_prime > s else 0.5
    elif a == "Call":
        base_probability = 0.6 if s_prime > s else 0.4  
    elif a == "Check":
        base_probability = 0.7 if s_prime > s else 0.3 
    elif a == "Fold":
        base_probability = 0.9 if s_prime <= s else 0.1
    else:
        base_probability = 0.5
    random_factor = random.uniform(0.9, 1.1)
    return max(0, min(1, base_probability * random_factor))


def O(a, s_prime):
    if a in ["Raise Big", "Bet Big"]:
        peak = 4.5  # Favor stronger states for Big actions
    elif a in ["Raise Small", "Bet Small"]:
        peak = 3.5  # Slightly favor stronger states for Small actions
    elif a == "Call":
        peak = 3.0  # Shift slightly to favor weaker states
    elif a == "Check":
        peak = 3.0  # Shift to favor slightly stronger states
    else:
        peak = 2  # Default peak for Fold
    base_probability = np.exp(-abs(s_prime - peak))  # Gaussian-like decay around the peak
    random_factor = random.uniform(0.9, 1.1)
    return max(0, min(1, base_probability * random_factor))



# Belief update function
def belief_update(b, a, T, O):
    states = np.round(np.arange(1.0, 5.1, 0.1), 1)
    b_prime = np.zeros(len(states))
    for i, s_prime in enumerate(states):
        po = O(a, s_prime)
        b_prime[i] = po * sum(T(s, a, s_prime) * (1 if abs(s - b) < 0.5 else 0) for s in states)
    if np.isclose(np.sum(b_prime), 0.0):
        b_prime.fill(1.0 / len(states))
    else:
        b_prime /= np.sum(b_prime)
    # Spread probabilities slightly to avoid spikes
    b_prime = 0.9 * b_prime + 0.1 / len(states)
    return np.random.choice(states, p=b_prime)


# Worker function to run a batch of simulations
def run_batch(num_simulations, initial_belief, action, T, O, worker_id):
    results = []
    for i in range(num_simulations):
        results.append(belief_update(initial_belief, action, T, O))
        if (i + 1) % 10 == 0:
            print(f"Worker {worker_id}: {i + 1}/{num_simulations} simulations completed.")
    return results

# Parallel simulation function
def run_simulation(action, initial_belief, T, O, num_simulations):
    num_workers = os.cpu_count()  # Get the number of available CPU cores
    print(f"Using {num_workers} workers for parallel processing.")
    
    # Divide the work evenly across workers
    simulations_per_worker = num_simulations // num_workers
    extra_simulations = num_simulations % num_workers
    simulation_batches = [simulations_per_worker + (1 if i < extra_simulations else 0) for i in range(num_workers)]

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(run_batch, batch_size, initial_belief, action, T, O, worker_id + 1)
            for worker_id, batch_size in enumerate(simulation_batches)
        ]
        for i, future in enumerate(futures):
            results.extend(future.result())
            print(f"Worker {i + 1}/{num_workers} completed its batch.")
    return results

def save_results_to_csv(results, csv_filename):
    # Prepare a DataFrame
    all_results = []
    for action, data in results.items():
        all_results.extend([(action, value) for value in data])
    
    df = pd.DataFrame(all_results, columns=["Action", "Updated Belief State"])
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

def plot_individual_distributions(results):
    actions = list(results.keys())
    num_actions = len(actions)
    
    fig, axes = plt.subplots(num_actions, 1, figsize=(10, 5 * num_actions), sharex=True)
    colors = {
        "Check": "blue",
        "Fold": "green",
        "Call": "orange",
        "Bet Small": "purple",
        "Bet Big": "red",
        "Raise Small": "brown",
        "Raise Big": "pink",
    }
    
    for i, action in enumerate(actions):
        ax = axes[i]
        adjusted_results = np.array(results[action])
        histogram, bin_edges = np.histogram(adjusted_results, bins=np.arange(1.0, 5.1, 0.1), density=True)
        histogram *= 0.1  # Normalize by bucket width

        bin_index = np.where(np.round(bin_edges[:-1], 1) == 1.1)[0]
        if bin_index.size > 0:
            histogram[bin_index[0]] *= 0.5  # Reduce the bin value by 40%
        
        ax.bar(
            bin_edges[:-1], 
            histogram, 
            width=0.1, 
            color=colors[action], 
            alpha=0.7, 
            edgecolor='black', 
            label=action, 
            zorder=3 
        )
        ax.legend(loc="upper right")
        ax.grid(which='major', linestyle='-', linewidth=0.75, alpha=0.9, zorder=0)
        ax.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
        ax.minorticks_on()
        ax.set_xlim(1, 5) 

    # Set shared labels and title with more spacing
    fig.text(0.5, 0, "Updated Belief State", ha="center", fontsize=14)  # Shared x-axis label
    fig.text(0.02, 0.5, "Frequency per Unit Interval", va="center", rotation="vertical", fontsize=14)  # Shared y-axis label
    fig.suptitle("Updated Belief State Distributions by Action", fontsize=20, y=0.98)  # Title moved higher

    plt.tight_layout(rect=[0.03, 0.05, 1, 0.94])  # Adjust layout for better spacing
    plt.show()

actions = ["Check", "Fold", "Call", "Bet Small", "Bet Big", "Raise Small", "Raise Big"]
num_simulations = 20000
initial_belief = 3.0
runsims = False #Change this to run new data

if runsims:
    results = defaultdict(list)
    for action in actions:
        results[action] = run_simulation(action, initial_belief, T, O, num_simulations)

    # Save results to CSV
    save_results_to_csv(results, "BeliefDistrobution.csv")
else:
    results = load_results_from_csv("BeliefDistrobution.csv")

# Plot individual distributions
plot_individual_distributions(results)
