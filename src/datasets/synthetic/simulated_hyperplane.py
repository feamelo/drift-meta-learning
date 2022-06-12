# This code generates the simulated hyperplane dataset used in many stream learning papers.

import numpy as np
import pandas as pd


def drift(start, number_to_generate, magnitude_of_change, prob_direction_change):
    directions = [magnitude_of_change] * (number_to_generate - 1)
    for i in range(number_to_generate - 1):
        if not np.random.uniform() >= prob_direction_change:
            directions[i] = directions[i-1] * -1
        else:
            directions[i] = directions[i-1]

    builder = [start]
    for direction in directions:
        start = start + direction
        builder.append(start)
    return builder


def hyperplane(num_to_generate, num_attributes, num_with_drift, mag_of_change, noise, prob_direction_change):
    # num_to_generate = 100 #qtde de linhas
    # num_attributes = 10 #qtde de atibutos
    # num_with_drift = 5 #qtd de atributos com drift
    # mag_of_change = 1.0
    # noise = .05
    # prob_direction_change = .1

    # Create list with "n" randomly uniform numbers
    n = num_attributes - num_with_drift
    static_weights = np.random.uniform(size=n)

    # Make a dataframe with "num_to_generate" equal rows containing "static_weights" data
    static_weights = pd.DataFrame([static_weights] * num_to_generate)

    dynamic_weights = pd.DataFrame([drift(start, num_to_generate, mag_of_change, prob_direction_change)
                                    for start in np.random.uniform(size=num_with_drift)]).T

    weights = pd.concat([static_weights.reset_index(
        drop=True), dynamic_weights], axis=1)
    weights.columns = range(weights.columns.size)

    data = pd.DataFrame(np.random.uniform(
        size=[num_to_generate, num_attributes]))

    result = data * weights
    result['zero'] = weights.apply(lambda x: 0.5 * np.sum(x), axis=1)
    classes = result.apply(lambda x: sum(
        x[0:num_attributes]) <= x['zero'], axis=1)

    # Do everything and then add noise
    data['target'] = classes.apply(
        lambda x: x if np.random.uniform() <= noise else not x)

    return data, weights


#     # This will generate 15 different datasets of 10000 observations
#     set.seed(1)
#     for (i in 1: 15) {
# hyperplane.data = hyperplane(10000, 10, 5, 1.0, .05, .1,)
# write.csv(hyperplane.data$data, paste("hyperplane", i, ".csv", sep=""), row.names=FALSE)
# write.csv(hyperplane.data$weights, paste("weights", i, ".csv", sep=""), row.names=FALSE)
# }
