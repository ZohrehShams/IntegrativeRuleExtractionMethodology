import random
import pandas as pd
from collections import Counter
from rules.term import TermOperator
import copy
from src import *

def overlapping_features(rules, include_operand=False):
    # Return the number of overlapping features considered in output class rulesets
    # TODO: If include operand: consider feature as a threshold on an input feature
    # TODO:this would require comparing 2 thresholds if they have the same sign but the value of threshold can differ

    all_features = []
    for class_rule in rules:
        class_features = set()
        for clause in class_rule.get_premise():
            for term in clause.get_terms():
                class_features.add(term.get_neuron())
        all_features.append(class_features)

    # Intersection over features used in each rule
    return len(set.intersection(*all_features))


# n top features recurring in the rules
def features_recurrence(rules, data_fp, n):
    data_df = pd.read_csv(data_fp)
    features_name = list(data_df.columns)

    features_list = []
    for rule in rules:
        for clause in rule.get_premise():
            for term in clause.get_terms():
                neuron = term.get_neuron()
                feature_name = features_name[neuron.index]
                features_list.append(feature_name)

    cnt = Counter(features_list)
    d = {k: v for k, v in sorted(dict(cnt).items(), key=lambda item: item[1], reverse=True)}
    for i in list(d)[0:n]:
        print(i)
    return d

# n top features recurring in the rules across the folds
def features_recurrence_across_folds(rules_list, data_fp, n):
    data_df = pd.read_csv(data_fp)
    features_name = list(data_df.columns)

    features_list = []
    for rules in rules_list:
        for rule in rules:
            for clause in rule.get_premise():
                for term in clause.get_terms():
                    neuron = term.get_neuron()
                    feature_name = features_name[neuron.index]
                    features_list.append(feature_name)

    cnt = Counter(features_list)
    d = {k: v for k, v in sorted(dict(cnt).items(), key=lambda item: item[1], reverse=True)}
    for i in list(d)[0:n]:
        print(i)
    return d


# recurrence of specific favourite features in the rules
def fav_features_recurrence(rules, data_fp, fav_features):
    data_df = pd.read_csv(data_fp)
    features_name = list(data_df.columns)

    features_list = []
    for rule in rules:
        for clause in rule.get_premise():
            for term in clause.get_terms():
                neuron = term.get_neuron()
                feature_name = features_name[neuron.index]
                if feature_name in fav_features:
                    features_list.append(feature_name)

    cnt = Counter(features_list)
    d = {k: v for k, v in sorted(dict(cnt).items(), key=lambda item: item[1], reverse=True)}
    print(d)


# recurrence of specific favourite features in the rules across the folds
def fav_features_recurrence_across_folds(rules_list, data_fp, fav_features):
    data_df = pd.read_csv(data_fp)
    features_name = list(data_df.columns)

    features_list = []
    for rules in rules_list:
        for rule in rules:
            for clause in rule.get_premise():
                for term in clause.get_terms():
                    neuron = term.get_neuron()
                    feature_name = features_name[neuron.index]
                    if feature_name in fav_features:
                        features_list.append(feature_name)

    cnt = Counter(features_list)
    d = {k: v for k, v in sorted(dict(cnt).items(), key=lambda item: item[1], reverse=True)}
    print(d)


# n random features appearing in the rules
def random_features_in_rules(rules, data_fp, n):
    data_df = pd.read_csv(data_fp)
    features_name = list(data_df.columns)

    features_list = []
    for rule in rules:
        for clause in rule.get_premise():
            for term in clause.get_terms():
                neuron = term.get_neuron()
                feature_name = features_name[neuron.index]
                if feature_name not in features_list:
                    features_list.append(feature_name)
    sorted_list = sorted(features_list)
    random.seed(0)
    random_fav_features = random.choices(sorted_list, k=n)
    return random_fav_features

# n random features appearing in the rules
def random_features_in_rules_across_folds(rules_list, data_fp, n):
    data_df = pd.read_csv(data_fp)
    features_name = list(data_df.columns)

    features_list = []
    for rules in rules_list:
        for rule in rules:
            for clause in rule.get_premise():
                for term in clause.get_terms():
                    neuron = term.get_neuron()
                    feature_name = features_name[neuron.index]
                    if feature_name not in features_list:
                        features_list.append(feature_name)

    sorted_list = sorted(features_list)
    random.seed(0)
    random_fav_features = random.choices(sorted_list, k=n)
    return random_fav_features


# n top features recurring in the rules for each class
def features_recurrence_per_class(rules, data_fp, n):
    data_df = pd.read_csv(data_fp)
    features_name = list(data_df.columns)

    features_dict = {}
    for rule in rules:
        class_features = []
        for clause in rule.get_premise():
            for term in clause.get_terms():
                neuron = term.get_neuron()
                feature_name = features_name[neuron.index]
                class_features.append(feature_name)
        features_dict[rule.get_conclusion().name] = class_features

    for item in features_dict:
        cnt = Counter(features_dict[item])
        d = {k: v for k, v in sorted(dict(cnt).items(), key=lambda item: item[1], reverse=True)}
        features_dict[item] = list(d)[0:n]
        
    return features_dict


# n top features recurring in the rules for each class across folds
def features_recurrence_per_class_across_folds(rules_list, data_fp, n):
    data_df = pd.read_csv(data_fp)
    features_name = list(data_df.columns)

    features_dict = {}
    for rules in rules_list:
        for rule in rules:
            class_features = []
            for clause in rule.get_premise():
                for term in clause.get_terms():
                    neuron = term.get_neuron()
                    feature_name = features_name[neuron.index]
                    class_features.append(feature_name)
            if rule.get_conclusion().name in features_dict.keys():
                features_dict[rule.get_conclusion().name] +=  class_features
            else:
                features_dict[rule.get_conclusion().name] = class_features

    for item in features_dict:
        cnt = Counter(features_dict[item])
        d = {k: v for k, v in sorted(dict(cnt).items(), key=lambda item: item[1], reverse=True)}
        features_dict[item] = list(d)[0:n]

    return features_dict
            
# frequency of operator for each feature appearing in the ruleset for each class
# {'ilc': {'ge_a': [8, 6], 'ge_b': [1, 3]},
#  'idc': {'ge_a': [2, 1], 'ge_c': [3, 3]}}
# In the above example feature 'ge_a' appears with greater and less than operator
# 8 and 6 times respectively for class ilc.
def features_operator_frequency_recurrence_per_class(rules, data_fp):
    data_df = pd.read_csv(data_fp)
    features_name = list(data_df.columns)
    features_operator_frequency_dict = {}

    for rule in rules:
        class_feature_operator_frequency_dict = {}
        class_terms = []
        for clause in rule.get_premise():
            for term in clause.get_terms():
                class_terms.append(term)

        for term in class_terms:
            neuron = term.get_neuron()
            feature_name = features_name[neuron.index]
            if feature_name not in class_feature_operator_frequency_dict:
                op_list = [0, 0]
                if term.operator == TermOperator.GreaterThan:
                    op_list[0] += 1
                else:
                    op_list[1] += 1
                class_feature_operator_frequency_dict[feature_name] = op_list
            else:
                op_list = class_feature_operator_frequency_dict[feature_name]
                if term.operator == TermOperator.GreaterThan:
                    op_list[0] += 1
                else:
                    op_list[1] += 1

        features_operator_frequency_dict[rule.get_conclusion().name] = class_feature_operator_frequency_dict
    return features_operator_frequency_dict


# frequency of operator for each feature appearing in the ruleset for each class across five folds
# {'ilc': {'ge_a': [8, 6], 'ge_b': [1, 3]},
#  'idc': {'ge_a': [2, 1], 'ge_c': [3, 3]}}
# In the above example feature 'ge_a' appears with greater and less than operator
# 8 and 6 times respectively for class ilc.
def features_operator_frequency_recurrence_per_class_across_folds(rules_list, data_fp):
    data_df = pd.read_csv(data_fp)
    features_name = list(data_df.columns)
    features_operator_frequency_dict = {}

    for rules in rules_list:
        for rule in rules:
            class_feature_operator_frequency_dict = {}
            class_terms = []
            for clause in rule.get_premise():
                for term in clause.get_terms():
                    class_terms.append(term)

            for term in class_terms:
                neuron = term.get_neuron()
                feature_name = features_name[neuron.index]
                if feature_name not in class_feature_operator_frequency_dict:
                    op_list = [0, 0]
                    if term.operator == TermOperator.GreaterThan:
                        op_list[0] += 1
                    else:
                        op_list[1] += 1
                    class_feature_operator_frequency_dict[feature_name] = op_list
                else:
                    op_list = class_feature_operator_frequency_dict[feature_name]
                    if term.operator == TermOperator.GreaterThan:
                        op_list[0] += 1
                    else:
                        op_list[1] += 1

            # features_operator_frequency_dict[rule.get_conclusion().name] = class_feature_operator_frequency_dict
            if rule.get_conclusion().name in features_operator_frequency_dict.keys():
                existing_class_feature_operator_frequency_dict = features_operator_frequency_dict[rule.get_conclusion().name]
                for feature in class_feature_operator_frequency_dict.keys():
                    if feature in existing_class_feature_operator_frequency_dict.keys():
                        x = class_feature_operator_frequency_dict[feature]
                        y = existing_class_feature_operator_frequency_dict[feature]
                        existing_class_feature_operator_frequency_dict[feature] = [x + y for x, y in zip(x, y)]
                    else:
                        existing_class_feature_operator_frequency_dict[feature] = class_feature_operator_frequency_dict[feature]
            else:
                features_operator_frequency_dict[rule.get_conclusion().name] = class_feature_operator_frequency_dict

    return features_operator_frequency_dict


# frequency of operator for n top recurring feature of each class
def top_features_operator_frequency_recurrence_per_class(rules, data_fp, n):
    recurrence = features_recurrence_per_class(rules, data_fp, n)
    operator_frequency = features_operator_frequency_recurrence_per_class(rules, data_fp)
    operator_frequency_new = copy.deepcopy(operator_frequency)

    for key in operator_frequency:
        temp_list = recurrence[key]
        for k in operator_frequency[key]:
            if k not in temp_list:
                del operator_frequency_new[key][k]
    return operator_frequency_new


# frequency of operator for n top recurring feature of each class across five folds
def top_features_operator_frequency_recurrence_per_class_across_folds(rules_list, data_fp, n):
    recurrence = features_recurrence_per_class_across_folds(rules_list, data_fp, n)
    operator_frequency = features_operator_frequency_recurrence_per_class_across_folds(rules_list, data_fp)
    operator_frequency_new = copy.deepcopy(operator_frequency)

    for key in operator_frequency:
        temp_list = recurrence[key]
        for k in operator_frequency[key]:
            if k not in temp_list:
                del operator_frequency_new[key][k]
    return operator_frequency_new


# recurrence of features in an explanation (set of individual rules used for a prediction)
def features_recurrence_in_explanation(explanation):
    data_df = pd.read_csv(DATA_FP)
    features_name = list(data_df.columns)

    features_list = []
    for clause in explanation:
        for term in clause.get_terms():
            neuron = term.get_neuron()
            feature_name = features_name[neuron.index]
            features_list.append(feature_name)

    cnt = Counter(features_list)
    
    return cnt