from treys import Card, Deck
import State_Generation_Methods
import Game_Methods
import Reward_Function
import Action_Function
import json
import numpy as np
from sklearn.neighbors import KernelDensity
import random
import sys
import os 
import csv

def calc_reward_next_state_and_log(effective_player_action, full_state_before, is_showdown,hole_card_probability_file, winner):
    #calculate the reward based on player action, get the new state and log to the file
    reward = Reward_Function.calculate_reward(effective_player_action, full_state_before, pot, is_showdown, winner, baseline=10)
    
    print(f"Player Reward: {reward:.2f}")
    full_state_after = State_Generation_Methods.generate_state(player_hole_cards,opponent_hole_cards, board_cards, hole_card_probability_file)
    full_state_after.update({ #bucket bankroll so that it is easier to store
        "Pot Size": round(pot / 20) if pot > 0 else 0,
        "Player Bankroll": round(player_bankroll / 20) if player_bankroll > 0 else 0,
        "Opponent Bankroll": round(opponent_bankroll / 20) if opponent_bankroll > 0 else 0,
    })
    #print("Next State:",json.dumps(full_state_before, indent=4))
    
    #State_Generation_Methods.export_training_data(csv_file, full_state_before, effective_player_action, reward, full_state_after)

def get_action_kde(policy_states, policy_actions, current_state, bandwidth=1.0):
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(policy_states)
    
    # Remove "Opponent Hand Strength" from the current state
    current_state_filtered = {key: current_state[key] for key in current_state if key != "Opponent Hand Strength"}
    
    # Convert the filtered state to an array
    current_state_array = [current_state_filtered[key] for key in current_state_filtered]
    
    # Get log density and determine the action
    log_density = kde.score_samples([current_state_array])
    action = policy_actions[np.argmax(log_density)]
    
    return action

def get_action_kde_POMDP(policy_states, policy_actions, current_state, bandwidth=1.0):
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(policy_states)
    
    # Remove "Opponent Hand Strength" from the current state
    current_state_filtered = {key: current_state[key] for key in current_state if key != "Opponent Hand Strength"}
    
    # Convert the filtered state to an array
    current_state_array = [current_state_filtered[key] for key in current_state_filtered]
    
    # Get log density and determine the action
    log_density = kde.score_samples([current_state_array])
    action = policy_actions[np.argmax(log_density)]
    
    return action


def calc_full_state_before_player():
    full_state_before = State_Generation_Methods.generate_state_player(player_hole_cards, board_cards,hole_card_probability_file,opponent_belief_state)
    full_state_before.update({"Pot Size": round(pot / 20) if pot > 0 else 0,
                       "Player Bankroll": round(player_bankroll / 20) if player_bankroll > 0 else 0,})
    print("Current State before action:",json.dumps(full_state_before, indent=4))
    return full_state_before

def calc_full_state_before_opponent():
    full_state_before_opponent = State_Generation_Methods.generate_state(opponent_hole_cards, board_cards,hole_card_probability_file)
    full_state_before_opponent.update({"Pot Size": round(pot / 20) if pot > 0 else 0,
                       "Player Bankroll": round(player_bankroll / 20) if player_bankroll > 0 else 0,})
    #print("Current State before action:",json.dumps(full_state_before_opponent, indent=4))
    return full_state_before_opponent

def check_trigger_showdown(effective_opponent_action, effective_player_action):
    if phase == "River" and effective_opponent_action == "Check" and effective_player_action == "Check":
        return True  #showdown triggered
    if phase == "River" and effective_opponent_action == "Call" and effective_player_action in ["Bet Big", "Bet Small"]:
        return True  #showdown triggered
    if phase == "River" and effective_opponent_action in ["Raise Big", "Raise Small", "Bet Big", "Bet Small"] and effective_player_action == "Call":
        return True  #showdown triggered
    if phase == "River" and effective_opponent_action == "Call" and effective_player_action in ["Raise Big", "Raise Small"]:
        return True  #showdown triggered
    return False  #no showdown
    
def run_showdown(effective_player_action, full_state_before, board_cards, player_hole_cards, opponent_hole_cards, pot, player_bankroll, opponent_bankroll):
    is_showdown = True
    # determine the winner and the new bankrolls
    player_bankroll, opponent_bankroll, pot,winner = Game_Methods.determine_showdown_winner(
        board_cards, player_hole_cards, opponent_hole_cards, pot, player_bankroll, opponent_bankroll
    )
    #calc_reward_next_state_and_log(effective_player_action, full_state_before, is_showdown, hole_card_probability_file, winner)
    pot = 0  #make sure the pot is set back to 0
    return player_bankroll, opponent_bankroll, pot

def handle_player_fold(effective_player_action,player_bankroll,opponent_bankroll, pot,full_state_before):
    if effective_player_action == "Fold":
        print(f"Player folds. Opponent wins the pot of {pot:.2f}.")
        opponent_bankroll += pot  #sransfer pot to the opponent
        pot = 0  #reset pot
        is_showdown = False
        #calc_reward_next_state_and_log(effective_player_action, full_state_before, is_showdown, hole_card_probability_file, winner=None)
        return player_bankroll, opponent_bankroll, pot, True  #end the round
        
def handle_opponent_fold(effective_opponent_action,player_bankroll,opponent_bankroll, pot,full_state_before):
    if effective_opponent_action == "Fold":
        print(f"Opponent folds. Player wins the pot of {pot:.2f}.")
        player_bankroll += pot  #sransfer pot to the player
        pot = 0  #reset pot
        is_showdown=True
        #calc_reward_next_state_and_log("Call", full_state_before, is_showdown, hole_card_probability_file, winner="Player")
        return player_bankroll, opponent_bankroll, pot, True  #end the round
    
def load_policy(policy_file):
    """
    Loads the policy from a file into separate state and action lists.

    Parameters:
        policy_file (str): Path to the policy file.

    Returns:
        tuple: (np.ndarray of policy states, list of policy actions)
    """
    policy_states = []
    policy_actions = []

    with open(policy_file, 'r') as f:
        for line in f:
            state, action = line.strip().split(": ")
            state_tuple = tuple(map(int, state.split(", ")))
            policy_states.append(state_tuple)
            policy_actions.append(action)

    return np.array(policy_states), policy_actions

def T(s, a, s_prime):
    if a in ["Raise Big", "Bet Big"]:
        base_probability = 0.8 if s_prime > s else 0.2
    elif a in ["Raise Small", "Bet Small"]:
        base_probability = 0.6 if s_prime > s else 0.4
    elif a == "Check":
        base_probability = 0.7 if s_prime <= s else 0.3
    elif a == "Fold":
        base_probability = 0.9 if s_prime <= s else 0.1
    else:
        base_probability = 1.0 / 5
    random_factor = random.uniform(0.9, 1.1)
    # Add slight randomness
    random_factor = random.uniform(0.9, 1.1)
    return max(0, min(1, base_probability * random_factor))  # Clamp to [0, 1]

def O(a, s_prime):
    """
    Observation probability for the opponent's action given their hand strength.
    Accounts for expanded action space with randomness.
    """
    if a == "Raise Big":
        base_probability = 0.9 if s_prime >= 4 else 0.1  # Very strong hands likely
    elif a == "Raise Small":
        base_probability = 0.7 if s_prime >= 3 else 0.3  # Moderate-to-strong hands likely
    elif a == "Bet Big":
        base_probability = 0.8 if s_prime >= 4 else 0.2  # Strong hands likely
    elif a == "Bet Small":
        base_probability = 0.6 if s_prime >= 3 else 0.4  # Moderate-to-strong hands likely
    elif a == "Check":
        base_probability = 0.8 if s_prime <= 2 else 0.2  # Weak hands likely
    elif a == "Fold":
        base_probability = 0.9 if s_prime <= 2 else 0.1  # Very weak hands likely
    else:
        base_probability = 1.0 / 5  # Uniform probability for undefined actions

    # Add slight randomness
    random_factor = random.uniform(0.9, 1.1)
    return max(0, min(1, base_probability * random_factor))  # Clamp to [0, 1]

def belief_update(b, a, T, O):
    """
    Updates the belief about the opponent's hand strength based on their action using transition
    and observation probabilities.

    Parameters:
    - b: Current belief about the opponent's hand strength (integer from 1 to 5).
    - a: Opponent's action (e.g., "Bet Small", "Raise Big", "Check", "Fold").
    - T: Transition probabilities function T(s, a, s') -> probability.
    - O: Observation probabilities function O(a, s') -> probability.

    Returns:
    - Updated belief about the opponent's hand strength (integer from 1 to 5).
    """
    print(f'inital Belif {b}')
    states = [1, 2, 3, 4, 5]  # Possible states for opponent hand strength
    b_prime = np.zeros(len(states))  # Initialize the updated belief vector

    # Calculate the updated belief for each state
    for i, s_prime in enumerate(states):
        # Observation probability: likelihood of observing action `a` given state `s_prime`
        po = O(a, s_prime)

        # Sum over all previous states `s`
        b_prime[i] = po * sum(T(s, a, s_prime) * (1 if s == b else 0) for s in states)

    # Normalize the belief vector
    if np.isclose(np.sum(b_prime), 0.0):
        b_prime.fill(1.0 / len(states))  # Uniform distribution if no valid update
    else:
        b_prime /= np.sum(b_prime)

    # Calculate the new belief as the most likely state
    updated_b = states[np.argmax(b_prime)]

    print(f'after Belif {updated_b}')

    return updated_b

def play_phase(phase, player_hole_cards,opponent_belief_state, opponent_bankroll, player_bankroll, pot, board_cards, hole_card_probability_file):
    #main script for playing a turn of poker
    Qpolicy_file = "QLearningPolicy.policy"
    Qpolicy_states, Qpolicy_actions = load_policy(Qpolicy_file)
    POMDPpolicy_file = "QLearningPolicy.policy"
    POMDPpolicy_states, POMDPpolicy_actions = load_policy(POMDPpolicy_file)
    """Play a single phase of the game."""
    print(f"\n--- Phase: {phase} ---")
    game_ended = False
    raise_count = 0
    ante = 5
    
    player_goes_first = random.choice([True, False])
    print(f"Player goes first: {player_goes_first}")

    if player_goes_first:
        # Gather the initial state for logging
        full_state_before = calc_full_state_before_player()

        # Player's turn
        player_action_space = ["Check", "Bet Big", "Bet Small", "Fold"]
        policy_action_POMDP = get_action_kde_POMDP(POMDPpolicy_states, POMDPpolicy_actions, full_state_before)
        player_bankroll, pot, effective_player_action, player_bet, raise_count = Action_Function.handle_action(
            entity="Player",  # Correct capitalization
            action_space=player_action_space,
            bankroll=player_bankroll,
            previous_bet=0,
            pot=pot,
            is_river=(phase == "River"),
            raise_count=raise_count,
            policy_action=policy_action_POMDP,
        )

        # Handle player folding
        if effective_player_action == "Fold":
            player_bankroll, opponent_bankroll, pot, game_ended = handle_player_fold(
                effective_player_action, player_bankroll, opponent_bankroll, pot, full_state_before
            )
            return player_bankroll, opponent_bankroll, pot, game_ended,opponent_belief_state
        elif phase == "Hole Cards": #play does ante-in
            if player_bankroll > ante:
                player_bankroll-=ante
                pot+=ante
            else:
                player_bankroll=0
                pot+=player_bankroll

        # Opponent's turn using the basline policy for smart decisions
        #full_state_before_opponent = calc_full_state_before_opponent()
        if effective_player_action not in ["Check"]:
            opponent_action_space = ["Call", "Raise Big", "Raise Small", "Fold"]
        else:
            opponent_action_space = ["Check", "Bet Big", "Bet Small", "Fold"]
        #policy_action = get_action_kde(Qpolicy_states, Qpolicy_actions, full_state_before_opponent)
        opponent_bankroll, pot, effective_opponent_action, opponent_bet, raise_count = Action_Function.handle_action(
            entity="Opponent",
            action_space=opponent_action_space,
            bankroll=opponent_bankroll,
            previous_bet=player_bet,
            pot=pot,
            is_river=(phase == "River"),
            raise_count=raise_count,
            policy_action=[],  # Opponent does not use policy
        )
        opponent_belief_state = belief_update(opponent_belief_state, effective_opponent_action, T, O)

        # Handle opponent folding
        if effective_opponent_action == "Fold":
            player_bankroll, opponent_bankroll, pot, game_ended = handle_opponent_fold(
                effective_opponent_action, player_bankroll, opponent_bankroll, pot, full_state_before
            )
            return player_bankroll, opponent_bankroll, pot, game_ended,opponent_belief_state
        elif phase == "Hole Cards": #play does ante-in
            if opponent_bankroll > ante:
                opponent_bankroll-=ante
                pot+=ante
            else:
                opponent_bankroll=0
                pot+=opponent_bankroll

        # Check for showdown
        is_showdown = check_trigger_showdown(effective_opponent_action, effective_player_action)
        if is_showdown:
            player_bankroll, opponent_bankroll, pot = run_showdown(
                effective_player_action, full_state_before, board_cards, player_hole_cards,
                opponent_hole_cards, pot, player_bankroll, opponent_bankroll
            )
            return player_bankroll, opponent_bankroll, pot, True,opponent_belief_state

        # Log the updated state and reward
        #calc_reward_next_state_and_log(effective_player_action, full_state_before, is_showdown,hole_card_probability_file, winner=None)

        # If the opponent bets or raises, the player must respond
        if effective_opponent_action in ["Bet Big", "Bet Small", "Raise Big", "Raise Small"]:
            player_action_space = ["Call", "Raise Big", "Raise Small", "Fold"]
            policy_action_POMDP = get_action_kde_POMDP(POMDPpolicy_states, POMDPpolicy_actions, full_state_before)
            player_bankroll, pot, effective_player_action, player_bet, raise_count = Action_Function.handle_action(
                entity="Player",
                action_space=player_action_space,
                bankroll=player_bankroll,
                previous_bet=opponent_bet,
                pot=pot,
                is_river=(phase == "River"),
                raise_count=raise_count,
                policy_action=policy_action_POMDP,
            )

            # Recalculate the initial state
            full_state_before = calc_full_state_before_player()

            # Handle player folding
            if effective_player_action == "Fold":
                player_bankroll, opponent_bankroll, pot, game_ended = handle_player_fold(
                    effective_player_action, player_bankroll, opponent_bankroll, pot, full_state_before
                )
                return player_bankroll, opponent_bankroll, pot, game_ended,opponent_belief_state

            # Check for showdown
            is_showdown = check_trigger_showdown(effective_opponent_action, effective_player_action)
            if is_showdown:
                player_bankroll, opponent_bankroll, pot = run_showdown(
                    effective_player_action, full_state_before, board_cards, player_hole_cards,
                    opponent_hole_cards, pot, player_bankroll, opponent_bankroll
                )
                return player_bankroll, opponent_bankroll, pot, True,opponent_belief_state
            
            #full_state_before_opponent = calc_full_state_before_opponent()
            # If the player raises, the opponent must respond
            if effective_player_action in ["Raise Big", "Raise Small"]:
                opponent_action_space = ["Call", "Fold"]
                #policy_action = get_action_kde(Qpolicy_states, Qpolicy_actions, full_state_before_opponent)
                opponent_bankroll, pot, effective_opponent_action, opponent_bet, raise_count = Action_Function.handle_action(
                    entity="Opponent",
                    action_space=opponent_action_space,
                    bankroll=opponent_bankroll,
                    previous_bet=player_bet,
                    pot=pot,
                    is_river=(phase == "River"),
                    raise_count=raise_count,
                    policy_action=[]
                )

                opponent_belief_state = belief_update(opponent_belief_state, effective_opponent_action, T, O)

                # Handle opponent folding
                if effective_opponent_action == "Fold":
                    player_bankroll, opponent_bankroll, pot, game_ended = handle_opponent_fold(
                        effective_opponent_action, player_bankroll, opponent_bankroll, pot, full_state_before
                    )
                    return player_bankroll, opponent_bankroll, pot, game_ended,opponent_belief_state

                # If this is the river, run a mandatory showdown
                if phase == "River":
                    run_showdown(
                        effective_player_action, full_state_before, board_cards, player_hole_cards,
                        opponent_hole_cards, pot, player_bankroll, opponent_bankroll
                    )
                    return player_bankroll, opponent_bankroll, pot, True,opponent_belief_state  # Game ends

        return player_bankroll, opponent_bankroll, pot, game_ended,opponent_belief_state  # Game continues

        
    else:
        #full_state_before_opponent = calc_full_state_before_opponent()
        #generate opponent action since they go first
        opponent_action_space = ["Check", "Bet Big", "Bet Small","Fold"] 
        #policy_action = get_action_kde(Qpolicy_states, Qpolicy_actions, full_state_before_opponent)
        opponent_bankroll, pot, effective_opponent_action, opponent_bet,raise_count = Action_Function.handle_action(
            entity="Opponent",
            action_space=opponent_action_space,
            bankroll=opponent_bankroll,
            previous_bet=0,
            pot=pot,
            is_river=(phase == "River"),
            raise_count=raise_count,
            policy_action=[]
        )

        opponent_belief_state = belief_update(opponent_belief_state, effective_opponent_action, T, O)
        # gather the inital state for logging
        full_state_before = calc_full_state_before_player()

        #handle a fold action
        if effective_opponent_action == "Fold":
            player_bankroll, opponent_bankroll, pot, game_ended = handle_opponent_fold(effective_opponent_action,player_bankroll,opponent_bankroll, pot,full_state_before)
            return player_bankroll, opponent_bankroll, pot, game_ended,opponent_belief_state
        elif phase == "Hole Cards": #play does ante-in
            if opponent_bankroll > ante:
                opponent_bankroll-=ante
                pot+=ante
            else:
                opponent_bankroll=0
                pot+=opponent_bankroll

        # select action spaced based on opponent action
        if effective_opponent_action not in ["Check"]:
            player_action_space = ["Call", "Raise Big", "Raise Small", "Fold"]
        else:
            player_action_space = ["Check", "Bet Big", "Bet Small", "Fold"]
        #generate player inital action
        policy_action_POMDP = get_action_kde_POMDP(POMDPpolicy_states, POMDPpolicy_actions, full_state_before)
        player_bankroll, pot, effective_player_action, player_bet,raise_count = Action_Function.handle_action(
            entity="Player",
            action_space=player_action_space,
            bankroll=player_bankroll,
            previous_bet=opponent_bet,
            pot=pot,
            is_river=(phase == "River"),
            raise_count=raise_count,
            policy_action=policy_action_POMDP
        )

        #handle player folding
        if effective_player_action == "Fold":
            player_bankroll, opponent_bankroll, pot, game_ended = handle_player_fold(effective_player_action,player_bankroll,opponent_bankroll, pot,full_state_before)
            return player_bankroll, opponent_bankroll, pot, game_ended,opponent_belief_state
        elif phase == "Hole Cards": #play does ante-in
            if player_bankroll > ante:
                player_bankroll-=ante
                pot+=ante
            else:
                player_bankroll=0
                pot+=player_bankroll

        #if this is the river, this is the first opportunity that a showdown could be triggered
        is_showdown = check_trigger_showdown(effective_opponent_action, effective_player_action)
        if is_showdown == True:
            # run the showdown to determine a winner and updated bankrolls
            player_bankroll, opponent_bankroll, pot = run_showdown(effective_player_action, full_state_before, board_cards, player_hole_cards, opponent_hole_cards, pot, player_bankroll, opponent_bankroll)
            return player_bankroll, opponent_bankroll, pot, True,opponent_belief_state

        #if this is not a showdown, log the updated state and reward from the previous player action
        #calc_reward_next_state_and_log(effective_player_action, full_state_before, is_showdown, hole_card_probability_file, winner=None)

        #full_state_before_opponent = calc_full_state_before_opponent()
        # if the player bet or raised, the opponent must respond with an action in the new action space
        #policy_action = get_action_kde(Qpolicy_states, Qpolicy_actions, full_state_before_opponent)
        if effective_player_action in ["Bet Big", "Bet Small", "Raise Big", "Raise Small"]:
            opponent_action_space = ["Call", "Raise Big", "Raise Small", "Fold"]
            opponent_bankroll, pot, effective_opponent_action, opponent_bet,raise_count = Action_Function.handle_action(
                entity="Opponent",
                action_space=opponent_action_space,
                bankroll=opponent_bankroll,
                previous_bet=player_bet,
                pot=pot,
                is_river=(phase == "River"),
                raise_count=raise_count,
                policy_action=[]
            )

            opponent_belief_state = belief_update(opponent_belief_state, effective_opponent_action, T, O)
            # recalcuate a new inital state
            full_state_before = calc_full_state_before_player()

            #handle opponent folding
            if effective_opponent_action == "Fold":
                player_bankroll, opponent_bankroll, pot, game_ended = handle_opponent_fold(effective_opponent_action,player_bankroll,opponent_bankroll, pot,full_state_before)
                return player_bankroll, opponent_bankroll, pot, game_ended,opponent_belief_state

            #if this is the river, this is the second opportunity that a showdown could be triggered
            is_showdown = check_trigger_showdown(effective_opponent_action, effective_player_action)
            if is_showdown == True:
                # run the showdown to determine a winner and updated bankrolls
                player_bankroll, opponent_bankroll, pot = run_showdown(effective_player_action, full_state_before, board_cards, player_hole_cards, opponent_hole_cards, pot, player_bankroll, opponent_bankroll)
                return player_bankroll, opponent_bankroll, pot, True,opponent_belief_state

            # if the opponent raised, the player can call or fold. Limiting to 1 raise keeps the training fast
            if effective_opponent_action in ["Raise Big", "Raise Small"]:
                player_action_space = ["Call","Fold"]
                policy_action_POMDP = get_action_kde_POMDP(POMDPpolicy_states, POMDPpolicy_actions, full_state_before)
                player_bankroll, pot, effective_player_action, player_bet,raise_count = Action_Function.handle_action(
                    entity="Player",
                    action_space=player_action_space,
                    bankroll=player_bankroll,
                    previous_bet=opponent_bet,
                    pot=pot,
                    is_river=(phase == "River"),
                    raise_count=raise_count,
                    policy_action=policy_action_POMDP)
                
                #handle player folding
                if effective_player_action == "Fold":
                    player_bankroll, opponent_bankroll, pot, game_ended = handle_player_fold(effective_player_action,player_bankroll,opponent_bankroll, pot,full_state_before)
                    return player_bankroll, opponent_bankroll, pot, game_ended,opponent_belief_state

                #if this is the river, this is the third opportunity for a showdown and it must happen based on how the game as ben set up
                if phase == "River": 
                    # run the showdown to determine a winner and updated bankrolls
                    run_showdown(effective_player_action, full_state_before,board_cards, player_hole_cards, opponent_hole_cards, pot, player_bankroll, opponent_bankroll)
                    return player_bankroll, opponent_bankroll, pot, True,opponent_belief_state  # Game ends

    # default return after each hand non-river
    return player_bankroll, opponent_bankroll, pot, game_ended, opponent_belief_state  # Game continues

import os
import csv

def update_results_csv(player_win, opponent_win, csv_filename="POMDPvsBaseline.csv"):
    """
    Update the CSV file with the cumulative results of each game, appending a new row for every game.

    Parameters:
    - player_win: 1 if the player wins the current game, 0 otherwise.
    - opponent_win: 1 if the opponent wins the current game, 0 otherwise.
    - csv_filename: The name of the CSV file to write the results to.
    """
    # Check if the file exists
    file_exists = os.path.isfile(csv_filename)

    if file_exists:
        # Read the existing data
        with open(csv_filename, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header
            rows = list(reader)  # Read all rows
            if rows:
                last_row = rows[-1]
                total_games = int(last_row[0])
                total_player_wins = int(last_row[1])
                total_opponent_wins = int(last_row[2])
            else:
                total_games = 0
                total_player_wins = 0
                total_opponent_wins = 0
    else:
        # Initialize totals if the file doesn't exist
        total_games = 0
        total_player_wins = 0
        total_opponent_wins = 0

    # Update the totals
    total_games += 1
    total_player_wins += player_win
    total_opponent_wins += opponent_win

    # Append the updated totals to the file
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the header if the file is new
        if not file_exists:
            writer.writerow(["Game", "Player Wins", "Opponent Wins"])
        # Append the updated totals
        writer.writerow([total_games, total_player_wins, total_opponent_wins])


# Main simulation
if __name__ == "__main__":
    default_bankroll = 100  # Default starting bankroll for both players
    num_games = 250  # Number of games to simulate for training

    for game in range(1, num_games + 1):
        print(f"\n--- Starting Game {game} ---")
        player_bankroll = default_bankroll
        opponent_bankroll = default_bankroll
        game_hand_count = 0
        player_wins = 0
        opponent_wins = 0

        while player_bankroll > 0.5 and opponent_bankroll > 0.5:  # Ensure at least 1 chip able to be played
            print(f"\nPlayer Bankroll: {player_bankroll:.2f}, Opponent Bankroll: {opponent_bankroll:.2f}")
            print(f"\n--- Starting Hand {game_hand_count + 1} ---")
            game_hand_count += 1
            deck = Deck()
            pot = 0
            board_cards = None  # Start with no community cards
            hole_card_probability_file = "../Hole_Card_Probabilities/hole_card_simulation_results.csv"
            opponent_belief_state = 3

            # Generate hole cards
            player_hole_cards = Game_Methods.generate_hole_cards(deck)
            print(f"Player Hole Cards: {[Card.int_to_pretty_str(card) for card in player_hole_cards]}")
            opponent_hole_cards = Game_Methods.generate_hole_cards(deck)
            print(f"Opponent Hole Cards: {[Card.int_to_pretty_str(card) for card in opponent_hole_cards]}")

            # Play through the phases of the hand
            game_ended = False
            for phase in ["Hole Cards", "Flop", "Turn", "River"]:
                player_bankroll, opponent_bankroll, pot, game_ended,opponent_belief_state = play_phase(
                    phase, player_hole_cards,opponent_belief_state, opponent_bankroll, 
                    player_bankroll, pot, board_cards, hole_card_probability_file
                )
                print(f'Main Loop Player bankroll: {player_bankroll}')
                print(f'Main Loop Opponent bankroll: {opponent_bankroll}')
                print(f'Main Loop pot: {pot}')
                if game_ended:
                    break
                board_cards = Game_Methods.transition_to_next_phase(phase, deck, board_cards)

        # Determine the winner for each game
        if player_bankroll > opponent_bankroll:
            player_wins = 1
            opponent_wins = 0
        elif opponent_bankroll > player_bankroll:
            player_wins = 0
            opponent_wins = 1

        # Update the CSV with the result of the current game
        update_results_csv(player_wins, opponent_wins)

        print(f"--- End of Game {game} ---")
        print(f"Total Hands Played: {game_hand_count}")
        print(f"Final Bankrolls - Player: {player_bankroll}, Opponent: {opponent_bankroll}")
        print("Bankrolls will be reset for the next game.\n")

