from rules.rule import Rule

# Note: this is just a test, not actually using this code yet

def remove_overlapping_clauses(rules):
    # Test: to remove overlapping clauses between rules
    rule_premises = [rule.get_premise() for rule in rules.values()]
    overlapping_clauses = set.intersection(*rule_premises)

    for output_class in rules.keys():
        class_rule = rules[output_class]

        rules[output_class] = Rule(premise=(class_rule.get_premise()-overlapping_clauses),
                                   conclusion=class_rule.get_conclusion())