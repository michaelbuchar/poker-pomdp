import csv
from treys import Card
import Game_Methods

def generate_state(hole_cards, board_cards,hole_card_probability_file):
    # gather all state data for storage
    win_prob = lookup_hole_card_win_probability(f"{Card.int_to_pretty_str(hole_cards[0])} {Card.int_to_pretty_str(hole_cards[1])}", hole_card_probability_file) #find win probability

    flop_status = Game_Methods.has_flop_been_dealt(board_cards or []) #determine if this turn is pre-flop

    if flop_status == 0: # if pre-flop, -1 all board and comparison states
        board_state_features = {
            "Paired Board": -1,
            "Connected Board": -1,
            "Board High Card": -1,
            "Count High Cards": -1,
        }
        player_board_features = {
            "Rank of Pair": -1,
            "Number of Paired Cards": -1,
            "Flush": -1,
            "Straight": -1,
            "Full House": -1,
        }
    else: # if we can compute the state, then call the state calculation methods
        board_state_features = get_board_state_features(board_cards)
        player_board_features = get_player_board_features(hole_cards, board_cards)

    return { #combine into single dictionary
        "Win Bucket": calc_win_state_bucket(win_prob),
        "Suited": is_paired([hole_cards[0],hole_cards[1]]),
        "Connectedness": int(abs(Card.get_rank_int(hole_cards[0]) - Card.get_rank_int(hole_cards[1])) == 1),
        "High Card": int(Card.get_rank_int(hole_cards[0]) >= 9 or Card.get_rank_int(hole_cards[1]) >= 9),
        "Flop Status": flop_status,  # 0: No flop, 1: Flop, 2: Turn, 3: River
        **board_state_features,
        **player_board_features,
    }

def lookup_hole_card_win_probability(hole_cards_query, hole_card_proability_file):
    # determine the hole card win probability from the previous simulations stored in a CSV file
    formatted_hole_cards = hole_cards_query.strip()
    reversed_hole_cards = " ".join(reversed(formatted_hole_cards.split())) # check the card order forwards and backwards for a match

    with open(hole_card_proability_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["Hole cards"].strip() == formatted_hole_cards:
                return float(row["Win prob"])
            elif row["Hole cards"].strip() == reversed_hole_cards:
                return float(row["Win prob"])
    return None # there should always be a match, but this is a catch

def calc_win_state_bucket(win_prob):
    if win_prob < 0.5:
        win_bucket = 1  # weak
    elif win_prob < 0.7:
        win_bucket = 2  # medium
    elif win_prob <= 0.99:
        win_bucket = 3  # strong
    return win_bucket

def is_paired(cards):
    #determine if there is a pair on the board
    ranks = [Card.get_rank_int(card) for card in cards]
    return int(len(ranks) != len(set(ranks)))  # Check for duplicate ranks

def is_board_connected(board_cards):
    # Determine if there are at least 3 consecutive ranks
    ranks = sorted(set(Card.get_rank_int(card) for card in board_cards))  # Sort and remove duplicates
    for i in range(len(ranks) - 2):  # Check groups of 3
        if ranks[i + 1] - ranks[i] == 1 and ranks[i + 2] - ranks[i + 1] == 1:
            return 1  # Found 3 consecutive ranks
    return 0


def get_board_high_card_level(board_cards):
    #determine the highest card on the board
    ranks = [Card.get_rank_int(card) for card in board_cards]
    highest_rank = max(ranks)
    if 0 <= highest_rank <= 3:#2-5
        return 0
    elif 4 <= highest_rank <= 8:#6-10
        return 1
    elif 9 <= highest_rank <= 12:#face card
        return 2
    else:
        print(f"Unexpected card rank: {highest_rank}")

def count_high_cards(board_cards):
    #count the number of high cards
    high_cards = [card for card in board_cards if 9 <= Card.get_rank_int(card) <= 12] #jack to ace
    return len(high_cards)


def get_board_state_features(board_cards):
    # return dictionary of state features
    return {
        "Paired Board": is_paired(board_cards),
        "Connected Board": is_board_connected(board_cards),
        "Board High Card": get_board_high_card_level(board_cards),
        "Count High Cards": count_high_cards(board_cards),
    }

def get_rank_of_pair(player_cards, board_cards):
    # determine the highest rank of the pair if there is one
    combined_cards = player_cards + board_cards
    ranks = [Card.get_rank_int(card) for card in combined_cards]
    rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}

    highest_pair_rank = max((rank for rank, count in rank_counts.items() if count >= 2), default=None) #determine rank

    if highest_pair_rank is None:
        return 0  #bucket no pair
    elif 0 <= highest_pair_rank <= 5:
        return 1  #bucket 2-7 rank pair
    elif 6 <= highest_pair_rank <= 9:
        return 2  #bucket 8 to 10 pair
    else:
        return 3  # bucket face card pair
    
def get_number_of_paired_cards(player_cards, board_cards):
    # determine how many pairs are on the board
    combined_cards = player_cards + board_cards
    ranks = [Card.get_rank_int(card) for card in combined_cards]
    rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
    return sum(1 for count in rank_counts.values() if count >= 2)

def get_flush_status(player_cards, board_cards):
    #determines if a flush is present
    combined_cards = player_cards + board_cards
    suits = [Card.get_suit_int(card) for card in combined_cards]
    suit_counts = {suit: suits.count(suit) for suit in set(suits)}

    max_suit_count = max(suit_counts.values(), default=0)
    if max_suit_count < 4:
        return 0  # No flush
    elif max_suit_count == 4:
        return 1  # Flush draw
    else:
        return 2  # Flush complete
    
def get_straight_status(player_cards, board_cards):
    #determine if a straight is present
    combined_cards = player_cards + board_cards
    ranks = sorted(set(Card.get_rank_int(card) for card in combined_cards))  #sorted ranks

    consecutive_count = 1 #initalize counts
    max_consecutive = 1
    for i in range(1, len(ranks)):
        if ranks[i] == ranks[i - 1] + 1: #start counting up
            consecutive_count += 1
            max_consecutive = max(max_consecutive, consecutive_count)
        else: #return back to 1 if the count stops
            consecutive_count = 1

    if max_consecutive < 4:
        return 0  #no straight
    elif max_consecutive == 4:
        return 1  #straight draw
    else:
        return 2  #straight present
    
def get_full_house_status(player_cards, board_cards):
    ##determine if a full house is present
    combined_cards = player_cards + board_cards
    ranks = [Card.get_rank_int(card) for card in combined_cards]
    rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}

    has_three_of_a_kind = any(count >= 3 for count in rank_counts.values())
    has_pair = sum(1 for count in rank_counts.values() if count >= 2) >= 2

    return 1 if has_three_of_a_kind and has_pair else 0

def get_player_board_features(player_cards, board_cards):
    # return list of player and board combined features
    return {
        "Rank of Pair": get_rank_of_pair(player_cards, board_cards),
        "Number of Paired Cards": get_number_of_paired_cards(player_cards, board_cards),
        "Flush": get_flush_status(player_cards, board_cards),
        "Straight": get_straight_status(player_cards, board_cards),
        "Full House": get_full_house_status(player_cards, board_cards),
    }

def export_training_data(csv_file, current_state, action, reward, next_state):
    #export training data to file
    fieldnames = (
        list(current_state.keys()) +
        ["Action", "Reward"] +
        [f"Next {key}" for key in next_state.keys()]
    )

    #put a header if empty
    try:
        with open(csv_file, "x", newline="") as file:  # "x" mode creates file if deleted
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
    except FileExistsError:
        pass 

    row = {**current_state, "Action": action, "Reward": reward}
    row.update({f"Next {key}": value for key, value in next_state.items()})

    with open(csv_file, "a", newline="") as file: #Append the data to the CSV
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(row)

