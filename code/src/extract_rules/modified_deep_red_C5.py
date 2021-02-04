from rules.C5 import C5
from rules.ruleset import Ruleset
from rules.rule import Rule

from logic_manipulator.substitute_rules import substitute

def extract_rules(model):
    # Should be 1 DNF rule per class
    DNF_rules = set()

    for output_class in model.output_classes:
        output_layer = model.n_layers - 1

        # Total rule - Only keep 1 total rule in memory at a time
        total_rule = Rule.initial_rule(output_layer=output_layer,
                                       output_class=output_class,
                                       threshold=0.5)

        for hidden_layer in reversed(range(0, output_layer)):
            print('Extracting layer %d rules:' % hidden_layer)
            # Layerwise rules only store all rules for current layer
            intermediate_rules = Ruleset()

            predictors = model.get_layer_activations(layer_index=hidden_layer)

            term_confidences = total_rule.get_terms_with_conf_from_rule_premises()
            terms = term_confidences.keys()

            # how many terms iterating over
            for _ in terms:
                print('.', end='', flush=True)
            print()

            for term in terms:
                print('.', end='', flush=True)

                # y1', y2', ...ym' = t(h(x1)), t(h(x2)), ..., t(h(xm))
                target = term.apply(model.get_layer_activations_of_neuron(layer_index=hidden_layer + 1,
                                                                          neuron_index=term.get_neuron_index()))

                prior_rule_confidence = term_confidences[term]
                rule_conclusion_map = {True: term, False: term.negate()}
                intermediate_rules.add_rules(C5(x=predictors, y=target,
                                                rule_conclusion_map=rule_conclusion_map,
                                                prior_rule_confidence=prior_rule_confidence))

            print('\nSubstituting layer %d rules' % hidden_layer, end=' ', flush=True)
            total_rule = substitute(total_rule=total_rule, intermediate_rules=intermediate_rules)
            print('done')

        DNF_rules.add(total_rule)

    return DNF_rules


