import simulate_priors
import create_scaled_priors
import iterate_possible_hole_cards

num_simulations = 1000000
output_file = "poker_simulation_results.csv"
dirichlet_hole_card_file="hole_card_simulation_results.csv"
simulate_priors.simulate_priors(num_simulations, output_file)

#read and scale the priors from simulate_priors.py, then run the simulation saving to the file
num_hole_card_simulations = 100000
percent_prior_in_dirichlet = 0.1
priors = create_scaled_priors.create_scaled_priors(output_file,num_simulations, num_hole_card_simulations,percent_prior_in_dirichlet)
iterate_possible_hole_cards.iterate_possible_hole_cards(priors, num_hole_card_simulations,percent_prior_in_dirichlet, dirichlet_hole_card_file)
