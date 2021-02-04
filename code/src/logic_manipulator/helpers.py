from typing import Set

from rules.term import Term, TermOperator

def terms_set_to_neuron_dict(terms: Set[Term]):
    # Convert set of conditions into dictionary
    neuron_conditions = {}

    for term in terms:
        if not term.get_neuron() in neuron_conditions:  # unseen Neuron
            neuron_conditions[term.get_neuron()] = {TermOp: [] for TermOp in TermOperator}
        neuron_conditions[(term.get_neuron())][term.get_operator()].append(term.get_threshold())

    # Return {Neuron: {TermOperator: [Float]}}
    return neuron_conditions