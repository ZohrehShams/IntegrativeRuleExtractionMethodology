"""
Uses extracted rules to classify new instances from test data and store results in file
"""
import random

import numpy as np
import pandas as pd

from rules.rule import Rule
from rules.term import Neuron

# TODO this code is highly parallelizable!

def predict(rules, X):
    """
    Args:
        rules: rules used to classify instances in X
        X: input data as numpy array

    Returns: Numpy array of predictions.
    """
    y = np.array([])

    for instance in X:
        # Map of Neuron objects to values from input data
        neuron_to_value_map = {Neuron(layer=0, index=i): instance[i] for i in range(len(instance))}
        # neuron_to_value_map = {Neuron(layer=0, index=i+1000): instance[i] for i in range(len(instance))}

        # Each output class given a score based on how many rules x satisfies
        class_ruleset_scores = {}
        for class_ruleset in rules:
            score = class_ruleset.evaluate_rule_by_majority_voting(neuron_to_value_map)
            class_ruleset_scores[class_ruleset] = score

        # Output class with max score decides the classification of instance. If tie, choose randomly
        if len(set(class_ruleset_scores.values())) == 1:
            max_class = random.choice(list(rules)).conclusion
        else:
            max_class = max(class_ruleset_scores, key=class_ruleset_scores.get).conclusion

        # Output class encoding is index out output neuron
        y = np.append(y, max_class.encoding)


    return y