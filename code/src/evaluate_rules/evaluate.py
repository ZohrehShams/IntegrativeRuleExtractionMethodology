from collections import OrderedDict

import pandas as pd
import tensorflow as tf

from evaluate_rules.overlapping_features import overlapping_features
from evaluate_rules.predict import predict
from evaluate_rules.accuracy import accuracy
from evaluate_rules.fidelity import fidelity
from evaluate_rules.comprehensibility import comprehensibility
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score
import os

from src import RULE_EXTRACTOR


def evaluate(rules, label_file_path):
    """
    Evaluate ruleset generated from DNNs
    """
    labels_df = pd.read_csv(label_file_path)

    predicted_labels = labels_df['rule_%s_labels' % RULE_EXTRACTOR.mode]
    true_labels = labels_df['true_labels']
    nn_labels = labels_df['nn_labels']


    # Compute Accuracy
    acc = accuracy(predicted_labels, true_labels)

    # Compute Fidelity
    fid = fidelity(predicted_labels, nn_labels)

    aucr = 0
    number_of_labels = true_labels.nunique()
    if (number_of_labels == 2):
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
        aucr = auc(fpr, tpr)
        print("Rules Auc %f" % aucr)
    else:
        for i in range(number_of_labels):
            fpr, tpr, thresholds = roc_curve(true_labels == i, predicted_labels == i)
            aucr += auc(fpr, tpr)
        aucr /= number_of_labels



    # Compute Comprehensibility
    comprehensibility_results = comprehensibility(rules)

    n_overlapping_features = overlapping_features(rules)

    results = OrderedDict(acc=acc, aucr = aucr, fid=fid, n_overlapping_features=n_overlapping_features)
    results.update(comprehensibility_results)

    return results



def evaluate_tree_rules(rules, label_file_path):
    """
    Evaluate ruleset generated from DT or RF
    """

    labels_df = pd.read_csv(label_file_path)

    predicted_labels = labels_df['rule_labels']
    true_labels = labels_df['true_labels']

    # Compute Accuracy
    acc = accuracy(predicted_labels, true_labels)
    print("tree Rules accuracy %f" % acc)

    auct = 0
    number_of_labels = true_labels.nunique()
    if (number_of_labels == 2):
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
        auct = auc(fpr, tpr)
        print("Rules Auc %f" % auct)
    else:
        for i in range(number_of_labels):
            fpr, tpr, thresholds = roc_curve(true_labels == i, predicted_labels == i)
            auct += auc(fpr, tpr)
        auct /= number_of_labels

    # Compute Comprehensibility
    comprehensibility_results = comprehensibility(rules)

    n_overlapping_features = overlapping_features(rules)

    results = OrderedDict(acc=acc, auct = auct, n_overlapping_features=n_overlapping_features)
    results.update(comprehensibility_results)

    return results

    
