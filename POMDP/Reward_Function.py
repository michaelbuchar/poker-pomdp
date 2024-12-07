def calculate_reward(effective_player_action, full_state_before, pot, is_showdown, winner, baseline):
    ## Calcualte the reward given to the player based on the action and strenth of the hand
    ##effective_player_action = action that they player took
    ##full_state_before = dictionary of state before
    ##pot = pot
    ##is_showdown = if this is an endgame reward
    ##winner = if this is an endgame reward, who won the hand
    ##baseline = scale factor

    reward = 0.0 # initalize reward

    #extract hand strength and board features, COULD ADD MORE STATES TO THIS IN FUTURE COMMMIT
    win_bucket = full_state_before.get("Win Bucket", 1) # 1 2 3
    hand_strength = ["Weak", "Medium", "Strong"][win_bucket - 1] #indexing
    paired_board = full_state_before.get("Paired Board", 0) # 0 or 1
    connected_board = full_state_before.get("Connected Board", 0) # 0 or 1

    #scale factor to amplify rewards for training
    scale_factor = max(pot / baseline, 5)

    if is_showdown:
        if winner == "Player":
            reward += 2 * scale_factor  #larger reward for winning a showdown
        elif winner == "Opponent":
            reward -= 2 * scale_factor  #larger penalty for losing a showdown
        elif winner == "Split":
            reward += 0.5 * scale_factor  #small reward for tying at showdown and not loosing
    else:
        if effective_player_action in ["Bet Big", "Raise Big"]:
            if hand_strength == "Strong":
                reward += 1.5 * scale_factor  #reward aggressive bets with strong hands 
            elif hand_strength == "Medium" and (paired_board or connected_board):
                reward += 1.0 * scale_factor  #encourage bets with an ok hand
            else:
                reward -= 1.5 * scale_factor  #penalize big bets with weak hands
        elif effective_player_action in ["Bet Small", "Raise Small"]:
            if hand_strength in ["Medium", "Strong"]:
                reward += 1.0 * scale_factor  #reward small bets with good hands, but the big bet should have a greater reward
            else:
                reward -= 1.0 * scale_factor  #penailize small bets with weak hands, but less than a big bet
        elif effective_player_action == "Call":
            if hand_strength in ["Medium", "Strong"]:
                reward += 0.5 * win_bucket * scale_factor #reward a call for a good hand, but less than a small bet and even less than a big bet
            else:
                reward -= 0.5 * scale_factor  #penailize weak hand calls, THIS COULD BE BASED ON THE POT IN FUTURE COMMIT
        elif effective_player_action == "Check":
            reward += 0.3 * scale_factor if hand_strength in ["Medium", "Strong"] else 0.1 * scale_factor #reward a check slightly since you get to see if odds improve
        elif effective_player_action == "Fold":
            if hand_strength == "Weak":
                reward += 0.5 * scale_factor  #reward for good fold
            else:
                reward -= 1.0 * scale_factor  #penailize for weak fold

    return round(reward, 1)
