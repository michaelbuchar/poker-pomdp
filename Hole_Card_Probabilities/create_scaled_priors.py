import csv

def create_scaled_priors(input_file, num_simulations, num_hole_card_simulations,percent_prior_in_dirichlet):
    ##Reads results from the CSV file and scales them to be X% of the number of simulations.
    ##input_file = input_filename from the simulate_priors.py output
    ##num_simulations = number of simulations that were run on the random poker priors simulation
    ##num_hole_card_simulations = number of simulations that will be run for each hole card probability determination
    ##percent_prior_in_dirichlet = what percent of the dirichlet will be made up of the priors

    total_scaled = 0

    with open(input_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        scaled_values = []

        for row in reader: #get each row, scale the priors to X% of num_simulations
            result, count = row
            scaled_value = int(int(count) * percent_prior_in_dirichlet * num_simulations)
            scaled_values.append((result, scaled_value))
            total_scaled += scaled_value #should add up to num_simulations, but made this modular if it was cut off during simulation

    # Make sure the values add up to num_simulations if the simulation was cut off 
    target_sum = num_hole_card_simulations*percent_prior_in_dirichlet
    normalized_priors = {}
    scale_factor = target_sum / total_scaled

    # scale each of the values to X%
    for result, scaled_value in scaled_values:
        normalized_value = round(scaled_value * scale_factor)
        normalized_priors[result] = normalized_value

    return normalized_priors
