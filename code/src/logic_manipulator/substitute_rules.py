import itertools

from rules.clause import ConjunctiveClause
from rules.rule import Rule
from rules.ruleset import Ruleset


def substitute(total_rule: Rule, intermediate_rules: Ruleset) -> Rule:
    """
    Substitute the intermediate rules from the previous layer into the total rule
    """
    new_premise_clauses = set()

    print('  Rule Premise Length: ', len(total_rule.get_premise()))
    premise_count = 1

    # for each clause in the total rule
    for old_premise_clause in total_rule.get_premise():
        print('    premise: %d' % premise_count)

        # list of sets of conjunctive clauses that are all conjunctive
        conj_new_premise_clauses = []
        for old_premise_term in old_premise_clause.get_terms():
            clauses_to_append = intermediate_rules.get_rule_premises_by_conclusion(old_premise_term)
            if clauses_to_append:
                conj_new_premise_clauses.append(clauses_to_append)

        # Print progress bar of all clause combinations need to be iterated over
        n_clause_combs = 1
        for clause_set in conj_new_premise_clauses:
            n_clause_combs = n_clause_combs * len(clause_set)
        if n_clause_combs > 10000:
            for _ in range(0, n_clause_combs // 10000):
                print('.', end='', flush=True)
            print()

        # When combined into a cartesian product, get all possible conjunctive clauses for merged rule
        # Itertools implementation does not build up intermediate results in memory
        conj_new_premise_clauses_combinations = itertools.product(*tuple(conj_new_premise_clauses))

        # given tuples of ConjunctiveClauses that are all now conjunctions, union terms into a single clause
        clause_comb_count = 0
        for premise_clause_tuple in conj_new_premise_clauses_combinations:
            new_clause = ConjunctiveClause()
            for premise_clause in premise_clause_tuple:
                new_clause = new_clause.union(premise_clause)
            new_premise_clauses.add(new_clause)

            clause_comb_count += 1
            if clause_comb_count % 10000 == 0:
                print('.', end='', flush=True)
        premise_count += 1

    return Rule(premise=new_premise_clauses, conclusion=total_rule.get_conclusion())
