import pickle

from model.generation.helpers.split_data import load_split_indices, load_data
from rules.clause import ConjunctiveClause
from rules.rule import OutputClass, Rule
from rules.term import Neuron
from src import n_fold_rules_RF_COMB_fp, N_FOLDS, N_FOLD_CV_SPLIT_INDICIES_FP, DATASET_INFO, DATA_FP

k = 4


def rank_rule_scores(dnf_rule: Rule, X_train, y_train, use_rl: bool):
    """

    Args:
        dnf_rule: dnf rule for a class
        X_train: train data
        y_train: test data
        use_rl: if true takes the length of rules into account too

    Returns:
        Set two scores for each rule, accuracy and rank, where rank is based on
        the following formula: (cc - ic)/(cc + ic) + (cc)/(ic + k) + (cc)/(rl)

    """
    # Each run of rule extraction return a DNF rule for each output class
    rule_output = dnf_rule.get_conclusion()

    # Each clause in the dnf rule is considered a rule for this output class
    for clause in dnf_rule.get_premise():
        cc = ic = 0
        rl = len(clause.get_terms())

        # Iterate over all items in the training data
        for i in range(0, len(X_train)):
            # Map of Neuron objects to values from input data. This is the form of data a rule expects
            neuron_to_value_map = {Neuron(layer=0, index=j): X_train[i][j]
                                   for j in range(len(X_train[i]))}

            if clause.evaluate(data=neuron_to_value_map):
                if rule_output.encoding == y_train[i]:
                    cc += 1
                else:
                    ic += 1


        # Compute rule rank_score
        if cc + ic == 0:
            accuracy_score = rank_score = 0
        else:
            accuracy_score = cc / (cc+ic)
            rank_score = ((cc - ic) / (cc + ic)) + cc / (ic + k)

        if use_rl:
            rank_score += cc / rl

        # print('cc: %d, ic: %d, rl: %d  rankscroe: %f' % (cc, ic, rl, rank_score))

        # Save rank score
        clause.set_accuracy_score(accuracy_score)
        clause.set_rank_score(rank_score)
        

def rank_rule_scores_fav(dnf_rule: Rule, features_name, favourite_features):
    """

    Args:
        dnf_rule: dnf rule for a class
        features_name: names of all features
        favourite_features: favourite features we want to promote

    Returns:
        Set favourite score for each rule, where score is based on
        the following formula: (cc - ic)/(cc + ic) + (cc)/(ic + k) + (cc)/(rl) + (x)/(i+2)

    """
    x = 4
    addition = 0
    i = 0

    for clause in dnf_rule.get_premise():
        base_score = clause.get_rank_score()
        for term in clause.get_terms():
            neuron = term.get_neuron()
            feature_name = features_name[neuron.index]
            if feature_name in favourite_features:
                addition += x/(i+2)
        clause.set_fav_score(base_score+addition)




