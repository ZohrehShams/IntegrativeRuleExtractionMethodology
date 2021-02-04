import math
from typing import Set
import pandas as pd

from rules.term import Term, Neuron
from rules.helpers import parse_variable_str_to_dict
from rules.rule import Rule

# Interface to R running embedded in a Python process
from rpy2.robjects.packages import importr
from rpy2 import robjects
from rpy2.robjects import pandas2ri

# activate Pandas conversion between R objects and Python objects
pandas2ri.activate()

# C50 R package is interface to C5.0 classification model
# C50 = importr('C50', lib_loc='C:/Users/sumaiyah/Documents/sumaiyah/R/win-library/3.5')
C50 = importr('C50')
C5_0 = robjects.r('C5.0')


def _truncate(x, decimals):
    """
    Truncates a given number x to contain at most `decimals`.

    :param float x: Number which we want to truncate.
    :param int decimals: Maximum number of decimals that the result will
        contain.

    :returns float: Truncated result.
    """
    power = 10**decimals
    return math.trunc(power * x) / power


def _parse_C5_rule_str(rule_str, rule_conclusion_map, prior_rule_confidence) -> Set[Rule]:
    rules_set: Set[Rule] = set()

    rule_str_lines = rule_str.split('\n')
    line_index = 2

    metadata_variables = parse_variable_str_to_dict(rule_str_lines[line_index])
    n_rules = metadata_variables['rules']

    for _ in range(0, n_rules):
        line_index += 1

        rule_data_variables = parse_variable_str_to_dict(rule_str_lines[line_index])
        n_rule_terms = rule_data_variables['conds']
        rule_conclusion = rule_conclusion_map[(rule_data_variables['class'])]

        # C5 rule confidence=(number of training cases correctly classified + 1)/(total training cases covered  + 2)
        rule_confidence = (rule_data_variables['ok'] + 1) / (rule_data_variables['cover'] + 2)
        # Weight rule confidence by confidence of previous rule
        rule_confidence = rule_confidence * prior_rule_confidence

        rule_terms: Set[Term] = set()
        for _ in range(0, n_rule_terms):
            line_index += 1

            term_variables = parse_variable_str_to_dict(rule_str_lines[line_index])
            term_neuron_str = term_variables['att'].split('_')
            term_neuron = Neuron(layer=int(term_neuron_str[1]), index=int(term_neuron_str[2]))

            term_operator = '<=' if term_variables['result'] == '<' else '>'  # In C5, < -> <=, > -> >
            term_operand = term_variables['cut']

            rule_terms.add(Term(neuron=term_neuron, operator=term_operator, threshold=term_operand))

        rules_set.add(Rule.from_term_set(premise=rule_terms, conclusion=rule_conclusion, confidence=rule_confidence))

    return rules_set

def C5(x: pd.DataFrame, y: pd.Series, rule_conclusion_map, prior_rule_confidence) -> Set[Rule]:
    y = robjects.vectors.FactorVector(y)

    C5_model = C50.C5_0(x=x, y=y, rules=True, control=C50.C5_0Control(winnow=True, minCases=15))
    # C5_model = C50.C5_0(x=x, y=y, rules=True, control=C50.C5_0Control(winnow=True))


    C5_rules_str = C5_model.rx2('rules')[0]

    C5_rules = _parse_C5_rule_str(C5_rules_str, rule_conclusion_map, prior_rule_confidence)

    return C5_rules
