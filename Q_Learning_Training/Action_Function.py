import random
import math

def handle_action(entity, action_space, bankroll, previous_bet, pot, is_river, raise_count,policy_action):
    # Action function for a game of poker
    ##entity = either "Player" or "Opponent".
    ##action_space = possible actions in a list
    ##bankroll = bankroll of the entity
    ##previous_bet = previous bet amount from the entity
    ##pot = current pot size
    ##is_river = if this phase is the river (0 or 1)
    ##updated_bankroll = new entity bankroll
    ##updated_pot = new pot
    ##action_taken = what action did the entity take
    ##bet = how much the entity has bet
    ##riase_count = how many times there has been a raise this turn

    def select_action(actions, bankroll, previous_bet, pot):
        """Select an action based on the current game state."""
        # If "Check" is available, sometimes opt for betting/raising instead
        if "Check" in actions:
            if random.random() < 0.6:  # 60% of the time, choose "Check"
                return "Check"
            else:
                return random.choice([a for a in actions if a not in ["Check"]])

        if "Call" in actions:
            # Fold if the call amount is greater than the bankroll
            if bankroll <= previous_bet:
                return "Fold"
            # Call only if the bet is relatively small compared to the bankroll
            if previous_bet > bankroll * 0.3:
                return "Fold"
            if previous_bet <= bankroll * 0.1:
                return random.choice(["Call", "Raise Small"])  # Call or raise if the bet is very small
            return "Call"

        # If folding is possible and there's no better option
        if "Fold" in actions:
            return "Fold"

        # If no specific conditions apply, pick a random action
        return random.choice(actions)


    def apply_action_to_game(action, bankroll, previous_bet=0): #update bets and pots based on actions
        if action == "Bet Big":
            bet = min(math.ceil(bankroll * 0.25), bankroll)
            bankroll -= bet
            return bankroll, bet
        elif action == "Bet Small":
            bet = min(math.ceil(bankroll * 0.10), bankroll)
            bankroll -= bet
            return bankroll, bet
        elif action == "Raise Big":
            raise_amount = min(math.ceil(previous_bet + (bankroll * 0.10)), bankroll)
            bankroll -= raise_amount
            return bankroll, raise_amount
        elif action == "Raise Small":
            raise_amount = min(math.ceil(previous_bet + (bankroll * 0.05)), bankroll)
            bankroll -= raise_amount
            return bankroll, raise_amount
        elif action == "Call":
            call_amount = min(math.ceil(previous_bet), bankroll)
            bankroll -= call_amount
            return bankroll, call_amount
        elif action == "Check":
            return bankroll, 0
        return bankroll, 0  # fold 

    if is_river and raise_count >= 2: #limit actions on the river after a raise/re-raise to not drain bankroll for data generation
        action_space = ["Call", "Fold"]

    if policy_action != [] and policy_action in action_space:
        action = policy_action
        print("Hey")
    else:
        action = select_action(action_space, bankroll, previous_bet, pot) #select action and update bets and pot
    if action in ["Raise Big", 'Raise Small']:
        raise_count += 1
    bankroll, bet = apply_action_to_game(action, bankroll, previous_bet)
    pot += bet

    print(f"{entity} Action: {action}, {entity} Bankroll: {bankroll:.2f}, Bet: {bet:.2f}, Pot: {pot:.2f}")
    return bankroll, pot, action, bet,raise_count
