from treys import Card, Deck
import State_Generation_Methods
import Game_Methods
import Reward_Function
import Action_Function
import json

def calc_reward_next_state_and_log(effective_player_action, full_state_before, is_showdown, winner):
    #calculate the reward based on player action, get the new state and log to the file
    reward = Reward_Function.calculate_reward(effective_player_action, full_state_before, pot, is_showdown, winner, baseline=10)
    
    print(f"Player Reward: {reward:.2f}")

    full_state_after = State_Generation_Methods.generate_state(player_hole_cards, board_cards, hole_card_probability_file)
    full_state_after.update({ #bucket bankroll so that it is easier to store
        "Pot Size": round(pot / 20) if pot > 0 else 0,
        "Player Bankroll": round(player_bankroll / 20) if player_bankroll > 0 else 0,
        "Opponent Bankroll": round(opponent_bankroll / 20) if opponent_bankroll > 0 else 0,
    })
    #print("Next State:",json.dumps(full_state_before, indent=4))
    
    State_Generation_Methods.export_training_data(csv_file, full_state_before, effective_player_action, reward, full_state_after)


def calc_full_state_before():
    full_state_before = State_Generation_Methods.generate_state(player_hole_cards, board_cards,hole_card_probability_file)
    full_state_before.update({"Pot Size": round(pot / 20) if pot > 0 else 0,
                       "Player Bankroll": round(player_bankroll / 20) if player_bankroll > 0 else 0,})
    #print("Current State before action:",json.dumps(full_state_before, indent=4))
    return full_state_before

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
    calc_reward_next_state_and_log(effective_player_action, full_state_before, is_showdown, winner)
    pot = 0  #make sure the pot is set back to 0
    return player_bankroll, opponent_bankroll, pot

def handle_player_fold(effective_player_action,player_bankroll,opponent_bankroll, pot,full_state_before):
    if effective_player_action == "Fold":
        print(f"Player folds. Opponent wins the pot of {pot:.2f}.")
        opponent_bankroll += pot  #sransfer pot to the opponent
        pot = 0  #reset pot
        calc_reward_next_state_and_log("Fold", full_state_before, is_showdown=True, winner="Opponent")
        return player_bankroll, opponent_bankroll, pot, True  #end the round
        
def handle_opponent_fold(effective_opponent_action,player_bankroll,opponent_bankroll, pot,full_state_before):
    if effective_opponent_action == "Fold":
        print(f"Opponent folds. Player wins the pot of {pot:.2f}.")
        player_bankroll += pot  #sransfer pot to the player
        pot = 0  #reset pot
        calc_reward_next_state_and_log("Call", full_state_before, is_showdown=True, winner="Player")
        return player_bankroll, opponent_bankroll, pot, True  #end the round

def play_phase(phase, player_hole_cards, opponent_bankroll, player_bankroll, pot, board_cards, raise_count=0):
    #main script for playing a turn of poker
    """Play a single phase of the game."""
    print(f"\n--- Phase: {phase} ---")
    game_ended = False
    raise_count = 0

    #generate opponent action since they go first every round
    opponent_action_space = ["Check", "Bet Big", "Bet Small","Fold"] 
    opponent_bankroll, pot, effective_opponent_action, opponent_bet,raise_count = Action_Function.handle_action(
        entity="Opponent",
        action_space=opponent_action_space,
        bankroll=opponent_bankroll,
        previous_bet=0,
        pot=pot,
        is_river=(phase == "River"),
        raise_count=raise_count
    )
    # gather the inital state for logging
    full_state_before = calc_full_state_before()

    #handle a fold action
    if effective_opponent_action == "Fold":
        player_bankroll, opponent_bankroll, pot, game_ended = handle_opponent_fold(effective_opponent_action,player_bankroll,opponent_bankroll, pot,full_state_before)
        return player_bankroll, opponent_bankroll, pot, game_ended

    # select action spaced based on opponent action
    if effective_opponent_action not in ["Check"]:
        player_action_space = ["Call", "Raise Big", "Raise Small", "Fold"]
    else:
        player_action_space = ["Check", "Bet Big", "Bet Small", "Fold"]
    #generate player inital action
    player_bankroll, pot, effective_player_action, player_bet,raise_count = Action_Function.handle_action(
        entity="Player",
        action_space=player_action_space,
        bankroll=player_bankroll,
        previous_bet=opponent_bet,
        pot=pot,
        is_river=(phase == "River"),
        raise_count=raise_count
    )

     #handle player folding
    if effective_player_action == "Fold":
        player_bankroll, opponent_bankroll, pot, game_ended = handle_player_fold(effective_player_action,player_bankroll,opponent_bankroll, pot,full_state_before)
        return player_bankroll, opponent_bankroll, pot, game_ended

    #if this is the river, this is the first opportunity that a showdown could be triggered
    is_showdown = check_trigger_showdown(effective_opponent_action, effective_player_action)
    if is_showdown == True:
        # run the showdown to determine a winner and updated bankrolls
        player_bankroll, opponent_bankroll, pot = run_showdown(effective_player_action, full_state_before, board_cards, player_hole_cards, opponent_hole_cards, pot, player_bankroll, opponent_bankroll)
        return player_bankroll, opponent_bankroll, pot, True

    #if this is not a showdown, log the updated state and reward from the previous player action
    calc_reward_next_state_and_log(effective_player_action, full_state_before, is_showdown, winner=None)

    # if the player bet or raised, the opponent must respond with an action in the new action space
    if effective_player_action in ["Bet Big", "Bet Small", "Raise Big", "Raise Small"]:
        opponent_action_space = ["Call", "Raise Big", "Raise Small", "Fold"]
        opponent_bankroll, pot, effective_opponent_action, opponent_bet,raise_count = Action_Function.handle_action(
            entity="Opponent",
            action_space=opponent_action_space,
            bankroll=opponent_bankroll,
            previous_bet=player_bet,
            pot=pot,
            is_river=(phase == "River"),
            raise_count=raise_count
        )
        # recalcuate a new inital state
        full_state_before = calc_full_state_before()

        #handle opponent folding
        if effective_opponent_action == "Fold":
            player_bankroll, opponent_bankroll, pot, game_ended = handle_opponent_fold(effective_opponent_action,player_bankroll,opponent_bankroll, pot,full_state_before)
            return player_bankroll, opponent_bankroll, pot, game_ended

        #if this is the river, this is the second opportunity that a showdown could be triggered
        is_showdown = check_trigger_showdown(effective_opponent_action, effective_player_action)
        if is_showdown == True:
            # run the showdown to determine a winner and updated bankrolls
            player_bankroll, opponent_bankroll, pot = run_showdown(effective_player_action, full_state_before, board_cards, player_hole_cards, opponent_hole_cards, pot, player_bankroll, opponent_bankroll)
            return player_bankroll, opponent_bankroll, pot, True

        # if the opponent raised, the player can call or fold. Limiting to 1 raise keeps the training fast
        if effective_opponent_action in ["Raise Big", "Raise Small"]:
            player_action_space = ["Call","Fold"]
            player_bankroll, pot, effective_player_action, player_bet,raise_count = Action_Function.handle_action(
                entity="Player",
                action_space=player_action_space,
                bankroll=player_bankroll,
                previous_bet=opponent_bet,
                pot=pot,
                is_river=(phase == "River"),
                raise_count=raise_count)
            
            #handle player folding
            if effective_player_action == "Fold":
                player_bankroll, opponent_bankroll, pot, game_ended = handle_player_fold(effective_player_action,player_bankroll,opponent_bankroll, pot,full_state_before)
                return player_bankroll, opponent_bankroll, pot, game_ended

            #if this is the river, this is the third opportunity for a showdown and it must happen based on how the game as ben set up
            if phase == "River": 
                # run the showdown to determine a winner and updated bankrolls
                run_showdown(effective_player_action, full_state_before,board_cards, player_hole_cards, opponent_hole_cards, pot, player_bankroll, opponent_bankroll)
                return player_bankroll, opponent_bankroll, pot, True  # Game ends

    # default return after each hand non-river
    return player_bankroll, opponent_bankroll, pot, game_ended  # Game continues

if __name__ == "__main__":
    default_bankroll = 100  #sefault starting bankroll for both players
    num_games = 20000  #number of games to simulate for training

    for game in range(1, num_games + 1):
        print(f"\n--- Starting Game {game} ---")
        player_bankroll = default_bankroll
        opponent_bankroll = default_bankroll
        game_hand_count = 0

        while player_bankroll > 0.5 and opponent_bankroll > 0.5: # make sure there is at least 1 chip able to be played
            print(f"\nPlayer Bankroll: {player_bankroll:.2f}, Opponent Bankroll: {opponent_bankroll:.2f}")
            print(f"\n--- Starting Hand {game_hand_count + 1} ---")
            game_hand_count += 1
            deck = Deck()
            pot = 0
            board_cards = None  # Start with no community cards
            csv_file = "poker_training_data.csv"  # Output CSV file
            hole_card_probability_file = "../Hole_Card_Probabilities/hole_card_simulation_results.csv"

            # Generate hole cards
            player_hole_cards = Game_Methods.generate_hole_cards(deck)
            print(f"Player Hole Cards: {[Card.int_to_pretty_str(card) for card in player_hole_cards]}")
            opponent_hole_cards = Game_Methods.generate_hole_cards(deck)
            print(f"Opponent Hole Cards: {[Card.int_to_pretty_str(card) for card in opponent_hole_cards]}")

            # Play through the phases of the hand
            game_ended = False
            for phase in ["Hole Cards", "Flop", "Turn", "River"]:
                player_bankroll, opponent_bankroll, pot, game_ended = play_phase(phase, player_hole_cards, opponent_bankroll, player_bankroll, pot, board_cards)
                print(f'Main Loop Player bankroll:{player_bankroll}')
                print(f'Main Loop Opponent bankroll:{opponent_bankroll}')
                if game_ended:
                    break
                board_cards = Game_Methods.transition_to_next_phase(phase, deck, board_cards)

        print(f"--- End of Game {game} ---")
        print(f"Total Hands Played: {game_hand_count}")
        print(f"Final Bankrolls - Player: {player_bankroll}, Opponent: {opponent_bankroll}")
        print("Bankrolls will be reset for the next game.\n")
