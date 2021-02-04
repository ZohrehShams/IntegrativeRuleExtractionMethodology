from typing import Set

from rules.term import Term, TermOperator
from logic_manipulator.helpers import terms_set_to_neuron_dict

def remove_redundant_terms(terms: Set[Term]) -> Set[Term]:
    """
    Remove redundant terms from a clause, returning only the necessary terms
    """
    neuron_conditions = terms_set_to_neuron_dict(terms)  # {Neuron: {TermOperator: [Float]}}
    necessary_terms = set()

    # Find most general neuron thresholds, range as general as possible, for '>' keep min, for '<=' keep max
    for neuron in neuron_conditions.keys():
        for TermOp in TermOperator:
            if neuron_conditions[neuron][TermOp]:  # if non-empty list
                necessary_terms.add(
                    Term(neuron, TermOp, TermOp.most_general_value(neuron_conditions[neuron][TermOp])))

    return necessary_terms