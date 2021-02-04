"""
Uses extracted rules to classify a new instance and the clauses that were responsible for this classification
"""
import random

import numpy as np
import pandas as pd

from rules.rule import Rule
from rules.term import Neuron
from src import DATA_FP

def  predict_explain(rules, instance):
    """
    Args:
        rules: rules used to classify instances in X
        intance: an instance of input data as numpy array

    Returns: a prediction and its explanation in the form of rules that got triggered for this prediction.
    """

    neuron_to_value_map = {Neuron(layer=0, index=i): instance[i] for i in range(len(instance))}

    class_ruleset_scores = {}
    class_ruleset_explanation = {}
    explanation_clauses = []
    for class_ruleset in rules:
        score, explanation_clauses = class_ruleset.evaluate_rule_by_majority_voting_with_explanation(neuron_to_value_map)
        class_ruleset_scores[class_ruleset] = score
        class_ruleset_explanation[class_ruleset] = explanation_clauses

    if len(set(class_ruleset_scores.values())) == 1:
        class_ruleset_random = random.choice(list(rules))
        max_class = class_ruleset_random.conclusion
    else:
        max_class_ruleset = max(class_ruleset_scores, key=class_ruleset_scores.get)
        max_class = max_class_ruleset.conclusion
        explanation_clauses = class_ruleset_explanation[max_class_ruleset]

    name = max_class.name

    return name, explanation_clauses



def print_explanation(prediction, explanation):
    explanation_str = '\n'
    for clause in explanation:
        explanation_str += "If " + print_clause(clause) + " Then " + str(prediction) + '\n'
    return explanation_str

def print_clause(clause):
    terms_str = [print_term(term) for term in clause.terms]
    return ' AND '.join(terms_str)

def print_term(term):
    n = round(term.threshold, 2)
    return '(' + print_neuron(term.neuron) + ' ' + str(term.operator) + ' ' + str(n) + ')'

def print_neuron(neuron):
    layer = neuron.layer
    if layer == 0:
        data_df = pd.read_csv(DATA_FP)
        features_name = list(data_df.columns)

        # deleting the label column name
        del features_name[-1]

        return features_name[neuron.index]
    else:
        return 'h_' + str(neuron.layer) + ',' + str(neuron.index)

