"""
Compute comprehensibility of ruleset generated

- Number of rules per class = number of conjunctive clauses in a classes DNF
- Number of terms per rule: Min, Max, Average
"""
from collections import namedtuple, OrderedDict

ClassRuleSetInfo = namedtuple('ClassRuleSetInfo', 'output_class n_rules min_n_terms_per_rule max_n_terms_per_rule '
                                                  'av_n_terms_per_rule')


def comprehensibility(rules):
    all_ruleset_info = []

    for class_ruleset in rules:
        class_name = class_ruleset.get_conclusion().name

        # Number of rules in that class
        n_rules_in_class = len(class_ruleset.get_premise())

        #  Get min max average number of terms in a clause
        min_n_terms = float('inf')
        max_n_terms = 0
        total_n_terms = 0
        for clause in class_ruleset.get_premise():
            # Number of terms in the clause
            n_clause_terms = len(clause.get_terms())

            if n_clause_terms < min_n_terms:
                min_n_terms = n_clause_terms
            if n_clause_terms > max_n_terms:
                max_n_terms = n_clause_terms

            total_n_terms += n_clause_terms

        av_n_terms_per_rule = total_n_terms / n_rules_in_class

        class_ruleset_info = [class_name,
                              n_rules_in_class,
                              min_n_terms,
                              max_n_terms,
                              av_n_terms_per_rule]

        all_ruleset_info.append(class_ruleset_info)

    output_classes, n_rules, min_n_terms, max_n_terms, av_n_terms_per_rule = zip(*all_ruleset_info)
    return OrderedDict(output_classes=output_classes,
                       n_rules_per_class=n_rules,
                       min_n_terms=min_n_terms,
                       max_n_terms=max_n_terms,
                       av_n_terms_per_rule=av_n_terms_per_rule)
