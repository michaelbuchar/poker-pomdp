def calculate_reward(effective_player_action, full_state_before, pot, is_showdown, winner, baseline):
    """
    Calculate the reward given to the player based on the action, strength of the hand,
    perceived strength of the opponent's hand, and additional contextual factors.
    """
    reward = 0.0  # Initialize reward

    # Extract features from the state
    win_bucket = full_state_before.get("Win Bucket", 1)  # Player's hand strength bucket (1-3)
    hand_strength = ["Weak", "Medium", "Strong"][win_bucket - 1]  # Player's hand strength label
    paired_board = full_state_before.get("Paired Board", 0)  # Paired board feature (0 or 1)
    connected_board = full_state_before.get("Connected Board", 0)  # Connected board feature (0 or 1)
    opponent_strength = full_state_before.get("Opponent Hand Strength", 3)  # Opponent's perceived hand strength (1-5)
    pot_size = full_state_before.get("Pot Size", 0)  # Pot size (0-5 scale)
    flop_status = full_state_before.get("Flop Status", 0)  # Flop status (0-3)
    flush = full_state_before.get("Flush", 0)  # Flush state (0: None, 1: Possible, 2: On board)
    straight = full_state_before.get("Straight", 0)  # Straight state (0: None, 1: Possible, 2: On board)
    player_bankroll = full_state_before.get("Player Bankroll", 5)  # Player's bankroll (0-5 scale)

    # Scale factor to amplify rewards for training
    scale_factor = max(pot / baseline, 5)

    if is_showdown:
        # Showdown rewards based on game outcome
        if winner == "Player":
            reward += (3 - opponent_strength / 5) * scale_factor  # Larger reward for winning against stronger opponents
        elif winner == "Opponent":
            reward -= (1 + opponent_strength / 5) * scale_factor  # Larger penalty for losing to weaker opponents
        elif winner == "Split":
            reward += 0.5 * scale_factor  # Small reward for tying at showdown
    else:
        # Non-showdown actions
        if effective_player_action in ["Bet Big", "Raise Big"]:
            if hand_strength == "Strong":
                if opponent_strength <= 2:
                    reward += 2.5 * scale_factor  # Aggressive play rewarded against weak opponents
                else:
                    reward += 1.5 * scale_factor  # Less aggressive reward against stronger opponents
            elif hand_strength == "Medium":
                if paired_board or connected_board:
                    reward += 1.5 * (3 - opponent_strength / 5) * scale_factor  # Reward higher against weaker opponents
                else:
                    reward -= opponent_strength / 5 * scale_factor  # Penalize bluffing against strong opponents
            else:
                reward -= 2.0 * opponent_strength / 5 * scale_factor  # Strong penalty for big bets with weak hands
        elif effective_player_action in ["Bet Small", "Raise Small"]:
            if hand_strength in ["Medium", "Strong"]:
                reward += (2 - opponent_strength / 5) * scale_factor  # Reward small bets, slightly less against strong opponents
            else:
                reward -= 1.0 * (opponent_strength / 5) * scale_factor  # Penalize small bets with weak hands
        elif effective_player_action == "Call":
            if hand_strength in ["Medium", "Strong"]:
                reward += 0.6 * win_bucket * (3 - opponent_strength / 5) * scale_factor  # Reward calls more against weak opponents
            else:
                reward -= 0.5 * (1 + opponent_strength / 5) * scale_factor  # Penalize weak hand calls against strong opponents
        elif effective_player_action == "Check":
            if hand_strength in ["Medium", "Strong"]:
                reward += 0.3 * (3 - opponent_strength / 5) * scale_factor  # Slightly reward checking to see more cards
            else:
                reward += 0.1 * scale_factor  # Small reward for passive play with weak hands
        elif effective_player_action in ["Fold","fold"]:
            if hand_strength == "Weak":
                reward += 0.5 * (1 + opponent_strength / 5) * scale_factor  # Reward folding against strong opponents
            elif opponent_strength >= 4:
                reward += 0.2 * scale_factor  # Slight reward for folding against strong opponents
            elif pot > 0:  # Add penalty for folding with some pot size
                reward -= 0.5 * scale_factor  # Small penalty for folding in contested pots
            else:
                reward += 0.1 * scale_factor  # Minimal reward for conservative play

    # Additional rewards/penalties based on pot size
    if pot_size >= 4:  # Large pot
        if effective_player_action in ["Bet Big", "Raise Big"] and hand_strength == "Strong":
            reward += 2.0 * scale_factor  # Reward high-stakes aggression with strong hands
        elif effective_player_action in ["Call", "Bet Small"] and hand_strength == "Weak":
            reward -= 2.0 * scale_factor  # Penalize weak hand play in large pots

    # Additional logic for flush and straight board states
    if flush == 2 or straight == 2:  # Flush or straight on the board
        if effective_player_action in ["Bet Big", "Raise Big"] and hand_strength != "Strong":
            reward -= 1.5 * scale_factor  # Penalize aggressive play without strong hands
        elif effective_player_action == "Fold" and hand_strength == "Weak":
            reward += 1.0 * scale_factor  # Reward cautious fold

    # Bankroll management rewards/penalties
    if player_bankroll <= 1:  # Low bankroll
        if effective_player_action in ["Bet Big", "Raise Big"] and hand_strength != "Strong":
            reward -= 2.0 * scale_factor  # Penalize unnecessary aggression with a low bankroll
        elif effective_player_action == "Fold" and hand_strength == "Weak":
            reward += 1.0 * scale_factor  # Reward cautious play to conserve bankroll

    return round(reward, 1)
