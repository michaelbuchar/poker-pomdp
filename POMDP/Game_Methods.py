from treys import Card, Evaluator, Deck
import numpy as np

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

def evaluate_hole_card_strength(hole_cards):
    """
    Evaluate the strength of hole cards based on pre-flop heuristics.
    The strength is calculated based on card ranks, suits, and pair status.

    Args:
        hole_cards (list): A list of two cards represented as integers.

    Returns:
        int: A strength score for the hole cards (0 to 100).
    """
    if len(hole_cards) != 2:
        raise ValueError("Hole cards must contain exactly two cards.")

    # Extract ranks and suits
    rank1 = Card.get_rank_int(hole_cards[0])
    rank2 = Card.get_rank_int(hole_cards[1])
    suit1 = Card.get_suit_int(hole_cards[0])
    suit2 = Card.get_suit_int(hole_cards[1])

    # High card values for poker (Jack = 9, Queen = 10, King = 11, Ace = 12)
    high_card_ranks = {9, 10, 11, 12}

    # Evaluate basic strength
    strength = 0

    # Pairs are strong
    if rank1 == rank2:
        if rank1 >= 10:  # High pair
            strength += 90
        elif rank1 >= 6:  # Mid pair
            strength += 70
        else:  # Low pair
            strength += 50

    # High cards (e.g., Ace, King, Queen, Jack)
    elif rank1 in high_card_ranks or rank2 in high_card_ranks:
        strength += 30

    # Suited cards add value
    if suit1 == suit2:
        strength += 20

    # Connected cards (e.g., 10 and 9)
    if abs(rank1 - rank2) == 1:
        strength += 15

    # Gap penalty for non-connected cards
    gap = abs(rank1 - rank2)
    if gap == 2:
        strength += 5
    elif gap > 2:
        strength -= 10

    # Bonus for Ace-King, Ace-Queen, Ace-Jack
    if (rank1 == 12 or rank2 == 12) and (rank1 in {10, 11} or rank2 in {10, 11}):
        strength += 25

    # Clamp the strength between 0 and 100
    strength = max(0, min(strength, 100))

    return strength

def evaluate_hand_rank(complete_hand): # not sure if needed?

    if len(complete_hand) < 5 or len(complete_hand) > 7:
        raise ValueError("A complete hand must have between 5 and 7 cards.")

    evaluator = Evaluator()
    # Split the hand into hole cards and board cards
    hole_cards = complete_hand[:2]
    board_cards = complete_hand[2:]

    # Use the Evaluator to calculate the hand rank
    hand_rank = evaluator.evaluate(board_cards, hole_cards)

    return hand_rank


def belief_update(b, P, a, o):
    """
    Updates a discrete belief based on the POMDP model.

    Parameters:
    - b: numpy array representing the belief vector
    - P: a dictionary containing the POMDP model:
        - P['S']: list of states
        - P['T']: function T(s, a, s') for transition probabilities
        - P['O']: function O(a, s', o) for observation probabilities
    - a: action taken
    - o: observation received

    Returns:
    - Updated belief vector (numpy array)
    """
    S = P['S']
    T = P['T']
    O = P['O']

    # Initialize the updated belief vector
    b_prime = np.zeros_like(b)

    # Update belief for each state s'
    for i_prime, s_prime in enumerate(S):
        # Observation likelihood
        po = O(a, s_prime, o)
        # Sum over all previous states s
        b_prime[i_prime] = po * sum(T(s, a, s_prime) * b[i] for i, s in enumerate(S))

    # Check if the sum of b_prime is approximately zero
    if np.isclose(np.sum(b_prime), 0.0):
        # If so, replace with a uniform distribution
        b_prime.fill(1.0)

    # Normalize the belief vector
    b_prime /= np.sum(b_prime)

    return b_prime





