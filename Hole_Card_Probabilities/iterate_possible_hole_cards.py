from treys import Deck, Evaluator, Card
import csv


def iterate_possible_hole_cards(priors, num_simulations_per_hole_cards,percent_prior_in_dirichlet, output_file):
    ## Iterates through all hole card combinations, simulating X number of games and stores the wins,loses,and splits to an output file
    ## Priors = scales priors from the X% dirichlet model
    ## Num_simulation_per_hole_cards = number of simulations that were run per hole card pair
    ## percent_prior_in_dirichlet = % of the dirichlet model that will be made up of the priors
    ## output_file = output where the hole card data will be saved

    evaluator = Evaluator() #shorthand evaluator
    all_cards = [Card.new(rank + suit) for rank in "23456789TJQKA" for suit in "cdhs"] #create list of all hole card combinations from Treys

    total_combinations = len(all_cards) * (len(all_cards) - 1) // 2 #calcualte number of sims for printing
    combination_count = 0

    with open(output_file, "w", newline="",encoding="utf-8") as file: #open the output file and print every line seperatly, so that if the simulation need to be restarted the data is saved
        writer = csv.DictWriter(file, fieldnames=["Hole cards", "Wins", "Losses", "Splits", "Win prob", "Loss prob", "Split prob"]) #write headers
        writer.writeheader()

        for i, card1 in enumerate(all_cards):
            for j, card2 in enumerate(all_cards): #iterate through all combinations
                if i >= j:  # Skip duplicate combinations and identical cards
                    continue

                combination_count += 1 # logging
                print(f"Simulating for hole cards: {Card.int_to_pretty_str(card1)} {Card.int_to_pretty_str(card2)} ({combination_count}/{total_combinations})")

                wins = 0
                losses = 0
                splits = 0

                for _ in range(num_simulations_per_hole_cards):
                    deck = Deck() #create a deck
                    deck.cards = [card for card in deck.cards if card != card1 and card != card2] #remove hole cards from the deck
                    opponent_hole_cards = deck.draw(2) #deal opponent cards and community cards
                    community_cards = deck.draw(5)
                    our_score = evaluator.evaluate(community_cards, [card1, card2])
                    opponent_score = evaluator.evaluate(community_cards, opponent_hole_cards)

                    if our_score < opponent_score: #lower score is better
                        wins += 1
                    elif our_score > opponent_score:
                        losses += 1
                    else: 
                        splits += 1

                #compute dirichlet-adjusted probabilities
                dirichlet_alpha = percent_prior_in_dirichlet * num_simulations_per_hole_cards
                win_prob = (wins + priors.get("Wins", 0) ) / (num_simulations_per_hole_cards + dirichlet_alpha)
                loss_prob = (losses + priors.get("Losses", 0)) / (num_simulations_per_hole_cards + dirichlet_alpha)
                split_prob = (splits + priors.get("Split pots", 0)) / (num_simulations_per_hole_cards + dirichlet_alpha)
                
                result = { # create the results row
                    "Hole cards": f"{Card.int_to_pretty_str(card1)} {Card.int_to_pretty_str(card2)}",
                    "Wins": wins,
                    "Losses": losses,
                    "Splits": splits,
                    "Win prob": win_prob,
                    "Loss prob": loss_prob,
                    "Split prob": split_prob,
                }

                writer.writerow(result) #write the results to the file
                file.flush() #save the file after each iteration


