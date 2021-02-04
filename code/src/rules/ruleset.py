"""
Represent a ruleset made up of rules
"""

from typing import Set, Dict

from rules.term import Term
from rules.clause import ConjunctiveClause
from rules.rule import Rule

class Ruleset:
    """
    Represents a set of disjunctive rules
    """

    def __init__(self, rules: Set[Rule] = None):
        if rules is None:
            rules = set()

        self.rules = rules

    def add_rules(self, rules: Set[Rule]):
        self.rules = self.rules.union(rules)

    def get_rule_premises_by_conclusion(self, conclusion) -> Set[ConjunctiveClause]:
        """
        Return a set of conjunctive clauses that all imply a given conclusion
        """
        premises = set()
        for rule in self.rules:
            if conclusion == rule.get_conclusion():
                premises = premises.union(rule.get_premise())

        return premises

    def get_terms_with_conf_from_rule_premises(self) -> Dict[Term, float]:
        """
        Return all the terms present in the bodies of all the rules in the ruleset with their max confidence
        """
        term_confidences = {}

        for rule in self.rules:
            for clause in rule.get_premise():
                clause_confidence = clause.get_confidence()
                for term in clause.get_terms():
                    if term in term_confidences:
                        term_confidences[term] = max(term_confidences[term], clause_confidence)
                    else:
                        term_confidences[term] = clause_confidence

        return term_confidences

    def get_terms_from_rule_premises(self) -> Set[Term]:
        """
        Return all the terms present in the bodies of all the rules in the ruleset
        """
        terms = set()
        for rule in self.rules:
            for clause in rule.get_premise():
                terms = terms.union(clause.get_terms())
        return terms

    def __str__(self):
        ruleset_str = '\n'
        for rule in self.rules:
            ruleset_str += str(rule) + '\n'

        return ruleset_str


    def get_rule_by_conclusion(self, conclusion) -> Rule:
        for rule in self.rules:
            if conclusion == rule.get_conclusion():
                return rule


    def get_ruleset_conclusions(self):
        conclusions=set()
        for rule in self.rules:
            conclusions.add(rule.get_conclusion())
        return conclusions




    def combine_external_clause(self, conjunctiveClause, conclusion):
        premises = self.get_rule_premises_by_conclusion(conclusion)
        premises.add(conjunctiveClause)

        rule = self.get_rule_by_conclusion(conclusion)
        newRule = Rule(premises, conclusion)

        if rule != None:
            self.rules.remove(rule)

        self.rules.add(newRule)

        return self.rules


    def combine_ruleset(self, other):
        conclusions_self = self.get_ruleset_conclusions()
        conclusions_other = other.get_ruleset_conclusions()
        combined_rules = set()

        diff = conclusions_self.symmetric_difference(conclusions_other)
        intersect = conclusions_self.intersection(conclusions_other)

        for rule in self.rules.union(other.rules):
            if rule.get_conclusion() in diff:
                combined_rules.add(rule)

        for rule in self.rules:
            if rule.get_conclusion() in intersect:
                premise = other.get_rule_premises_by_conclusion(rule.get_conclusion())
                combined_premise = premise.union(rule.get_premise())
                combined_rule = Rule(combined_premise, rule.get_conclusion())
                combined_rules.add(combined_rule)
        return combined_rules

