from treys import Card, Evaluator

def generate_hole_cards(deck):
    return [deck.draw(1)[0] for _ in range(2)]


def generate_board_cards(deck, num_cards=3):
    return [deck.draw(1)[0] for _ in range(num_cards)]


def has_flop_been_dealt(board_cards):
    ## 0 = no flop
    ## 1 = flop
    ## 2 = turn 
    ## 3 = river
    if len(board_cards) < 3:
        return 0  
    elif len(board_cards) == 3:
        return 1
    elif len(board_cards) == 4:
        return 2 
    elif len(board_cards) == 5:
        return 3
    

def transition_model(deck, board_cards):
    if not board_cards: #transition to flop
        board_cards = generate_board_cards(deck, num_cards=3)
        print(f"Transition to Flop : {[Card.int_to_pretty_str(card) for card in board_cards]}")
    elif len(board_cards) == 3: # transition to turn
        board_cards += generate_board_cards(deck, num_cards=1)
        print(f"Transition to Turn : {[Card.int_to_pretty_str(card) for card in board_cards]}")
    elif len(board_cards) == 4: #transition to river
        board_cards += generate_board_cards(deck, num_cards=1)
        print(f"Transition to River : {[Card.int_to_pretty_str(card) for card in board_cards]}")
    else:
        print("Transition out of bounds")
    return board_cards

def transition_to_next_phase(phase, deck, board_cards):
    if phase != "River":
        return transition_model(deck, board_cards) #call the transition model if we need to transition
    return board_cards 

def determine_showdown_winner(board_cards, player_hole_cards, opponent_hole_cards, pot, player_bankroll, opponent_bankroll):
    ##determine the winner of the showdown and update bankrolls and the pot
    ##board cards = the community cards stored in Treys format
    ##player_hole_cards = the player hole cards stored in Treys format
    ##opponent_hole_cards = the opponent hole cards stored in Treys format
    ##pot = pot amout
    ##player_bankroll = player chips
    ##opponent_bankroll = opponent chips
    player_score = Evaluator().evaluate(board_cards, player_hole_cards)
    opponent_score = Evaluator().evaluate(board_cards, opponent_hole_cards)

    print(f"Player Hand Score: {player_score}")
    print(f"Opponent Hand Score: {opponent_score}")

    if player_score < opponent_score:  #lowe score is better
        print("Player wins the pot!")
        player_bankroll += pot
        pot = 0 #reset pot
        print(f'Player bankroll:{player_bankroll}')
        print(f'Opponent bankroll:{opponent_bankroll}')
        winner = "Player" #use in logging
    elif player_score > opponent_score:
        print("Opponent wins the pot!")
        opponent_bankroll += pot
        pot = 0 #reset pot
        print(f'Player bankroll:{player_bankroll}')
        print(f'Opponent bankroll:{opponent_bankroll}')
        winner = 'Opponent' #use in logging
    else:
        print("The round ends in a tie. Pot is split.")
        player_bankroll += pot / 2
        opponent_bankroll += pot / 2
        pot = 0 #reset pot
        print(f'Player bankroll:{player_bankroll}')
        print(f'Opponent bankroll:{opponent_bankroll}')
        winner = 'Split' #use in logging

    return player_bankroll, opponent_bankroll, pot,winner

