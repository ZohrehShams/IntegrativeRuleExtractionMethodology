from sklearn.tree import _tree, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from typing import Set
import numpy as np

from rules.term import Term, Neuron
from rules.clause import ConjunctiveClause
from rules.rule import Rule
from rules.ruleset import Ruleset
from rules.rule import OutputClass
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from evaluate_rules.predict import predict
from evaluate_rules.accuracy import accuracy
from logic_manipulator.delete_redundant_terms import remove_redundant_terms

import pickle
from sklearn.datasets import load_iris, load_breast_cancer




def tree_to_code(tree, feature_names_to_id):
    tree_ = tree.tree_
    feature_names = list(feature_names_to_id.keys())
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature]

    pathto = dict()
    # Mapping tree branches (clause) to the class of that branch
    clause_con_dict = dict()

    def recurse(node, depth, parent):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            s = Term(Neuron(0, feature_names_to_id[name]), '<=', threshold)
            if node == 0:
                pathto[node] = {s}
            else:
                pathto[node] = pathto[parent].union({s})

            recurse(tree_.children_left[node], depth + 1, node)

            s = Term(Neuron(0, feature_names_to_id[name]), '>', threshold)
            if node == 0:
                pathto[node] = {s}
            else:
                pathto[node] = pathto[parent].union({s})

            recurse(tree_.children_right[node], depth + 1, node)
        else:
            clause = ConjunctiveClause(pathto[parent])
            conclusion = np.argmax(tree_.value[node])
            clause_con_dict[clause] = conclusion

    recurse(0, 1, 0)
    return clause_con_dict




def decisionTree_Ruleset(decision_tree,feature_names_to_id_map, output_classes) -> Set[Rule]:
    rule_conclusion_map = tree_to_code(decision_tree, feature_names_to_id_map)

    rules_set: Set[Rule] = set()

    for clause in rule_conclusion_map.keys():
        conclusion = rule_conclusion_map[clause]
        for item in output_classes:
            if item.encoding == conclusion:
                name = item.name
        rules_set = Ruleset(rules_set).combine_external_clause(clause, OutputClass(name, conclusion))

    return rules_set





def randomForest_Ruleset(random_forest, feature_names_to_id_map, output_classes) -> Set[Rule]:
    rules_set: Set[Rule] = set()
    for estimator in random_forest:
        rule_conclusion_map = tree_to_code(estimator, feature_names_to_id_map)
        for clause in rule_conclusion_map.keys():
            conclusion = rule_conclusion_map[clause]
            for item in output_classes:
                if item.encoding == conclusion:
                    name = item.name
            rules_set = Ruleset(rules_set).combine_external_clause(clause, OutputClass(name, conclusion))

    return rules_set


