"""
Find best neural network initialisation

1. Load train test split
2. Build 5 Neural Networks with different initialisations using besy hyper parameters
3. Perform rule extraction on these 5 networks
4. network with smallest ruleset, save that initialisation
"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf
# from keras.models import load_model

import dnn_re
from evaluate_rules.evaluate import evaluate
from evaluate_rules.predict import predict
from model.generation.helpers import split_data
from model.generation.helpers.build_and_train_model import build_and_train_model
from src import TEMP_DIR, RULE_EXTRACTOR, NN_INIT_SPLIT_INDICES_FP, NN_INIT_RE_RESULTS_FP, LABEL_FP, \
    BEST_NN_INIT_FP


def run(X, y, hyperparameters):
    train_index, test_index = split_data.load_split_indices(file_path=NN_INIT_SPLIT_INDICES_FP)

    # Split data
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    # Save information about nn initialisation
    if not os.path.exists(NN_INIT_RE_RESULTS_FP):
        pd.DataFrame(data=[],
                     columns=['run']).to_csv(
            NN_INIT_RE_RESULTS_FP, index=False)

    # Path to trained neural network
    model_file_path = TEMP_DIR + 'model.h5'

    # Smallest ruleset i.e. total number of rules
    smallest_ruleset_size = np.float('inf')
    smallest_ruleset_acc = 0
    best_init_index = 0

    for i in range(0, 5):
        print('Testing initialisation %d' % i)

        # Build and train nn put it in temp/
        build_and_train_model(X_train, y_train, X_test, y_test, **hyperparameters, model_file_path=model_file_path)

        # Extract rules
        nn_accuracy, nn_AUC, rules, re_time, re_memory= dnn_re.run(X, y, train_index, test_index, model_file_path)

        # Save labels to labels.csv:
        # label - True data labels
        label_data = {'id': test_index,
                      'true_labels': y_test}
        # label - Neural network data labels. Use NN to predict X_test
        nn_model = tf.keras.models.load_model(model_file_path)
        nn_predictions = np.argmax(nn_model.predict(X_test), axis=1)
        label_data['nn_labels'] = nn_predictions
        # label - Rule extraction labels
        rule_predictions = predict(rules, X_test)
        label_data['rule_%s_labels' % RULE_EXTRACTOR.mode] = rule_predictions
        pd.DataFrame(data=label_data).to_csv(LABEL_FP, index=False)

        # Save rule extraction time and memory usage
        results_df = pd.read_csv(NN_INIT_RE_RESULTS_FP)
        results_df.loc[i, 'run'] = i
        results_df.loc[i, 're_time (sec)'] = re_time

        re_results = evaluate(rules, LABEL_FP)
        results_df.loc[i, 'nn_acc'] = nn_accuracy
        results_df.loc[i, 're_acc'] = re_results['acc']
        results_df.loc[i, 're_fid'] = re_results['fid']
        results_df.loc[i, 'rules_num'] = sum(re_results['n_rules_per_class'])

        results_df = results_df.round(3)

        results_df = results_df[["run", "nn_acc", "re_acc", "re_fid", "re_time (sec)", "rules_num"]]

        results_df.to_csv(NN_INIT_RE_RESULTS_FP, index=False)

        # If this initialisation extrcts a smaller ruleset - save it
        ruleset_size = sum(re_results['n_rules_per_class'])
        if (ruleset_size < smallest_ruleset_size) \
                or (ruleset_size == smallest_ruleset_size and re_results['acc'] > smallest_ruleset_acc):
            smallest_ruleset_size = ruleset_size
            smallest_ruleset_acc = re_results['acc']
            best_init_index = i

            # Save initilisation as best_initialisation.h5
            tf.keras.models.load_model(TEMP_DIR + 'initialisation.h5').save(BEST_NN_INIT_FP)

    print('Found neural network with the best initialisation. (%d)' % best_init_index)
    print('==================================================================================================')
