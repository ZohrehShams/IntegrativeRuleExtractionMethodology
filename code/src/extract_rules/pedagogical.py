import numpy as np
import pandas as pd

from logic_manipulator.merge import merge
from rules.C5 import C5


def extract_rules(model):
    """
    Extract rules in a pedagogical manner using C5 on the outputs and inputs of the network
    Assigns model rules to extracted rules
    """
    # Inputs to neural network. C5 requires DataFrame inputs
    X = model.get_layer_activations(layer_index=0)

    # y = output classifications of neural network. C5 requires y to be a pd.Series
    nn_model_predictions = np.argmax(model.model.predict(X), axis=1)
    y = pd.Series(nn_model_predictions)

    assert len(X) == len(y), 'Unequal number of data instances and predictions'

    # Use C5 to extract rules using only input and output values of the network
    # C5 returns disjunctive rules with conjunctive terms
    # R
    rules = C5(x=X, y=y,
               rule_conclusion_map=model.output_classes,
               prior_rule_confidence=1)

    # Merge rules so that they are in Disjunctive Normal Form
    # Now there should be only 1 rule per rule conclusion
    # Ruleset is encapsulated/represented by a DNF rule
    # DNF_rules is a set of rules
    DNF_rules = merge(rules)
    assert len(DNF_rules) == len(model.output_classes), 'Should only exist 1 DNF rule per class'

    return DNF_rules
