from treys import Deck, Evaluator, Card
import csv

def simulate_priors(num_simulations, output_file):
    ## Simulates X number of poker games to generate dirichlet priors
    ## num_simulation = number of simulations that will be run 
    ## output_file = output filename
    ## Output is the total number of wins, losses and splits

    wins = 0
    losses = 0
    splits = 0

    for i in range(num_simulations):
        print(f"Sim progress: {i}/{num_simulations}")
        deck = Deck() #create a new deck
        hole_cards = deck.draw(2) #draw 2 cards for the player
        opponent_hole_cards = deck.draw(2) #draw 2 cards for the opponent
        community_cards = deck.draw(5) #draw 5 cards for the community cards
        our_score = Evaluator.evaluate(community_cards, hole_cards) #evaluate hands of player and opponent
        opponent_score = Evaluator.evaluate(community_cards, opponent_hole_cards)

        if our_score < opponent_score:  #lower score is better in Treys
            wins += 1
        elif our_score > opponent_score:
            losses += 1
        else:
            splits += 1

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Result", "Count"])
        writer.writerow(["Wins", wins])
        writer.writerow(["Losses", losses])
        writer.writerow(["Split pots", splits])

    return wins, losses, splits