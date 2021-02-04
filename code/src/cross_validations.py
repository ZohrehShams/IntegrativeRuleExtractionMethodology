import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import load_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.utils import class_weight
from prettytable import PrettyTable


from src import *
import dnn_re, tree_re
from evaluate_rules.evaluate import evaluate, evaluate_tree_rules
from evaluate_rules.overlapping_features import features_recurrence, features_recurrence_per_class, \
    features_operator_frequency_recurrence_per_class, top_features_operator_frequency_recurrence_per_class, \
    random_features_in_rules
from evaluate_rules.predict import predict
from model.generation import generate_data
from model.generation.helpers.init_dataset_dir import clear_file
from model.generation.helpers.split_data import load_split_indices, load_data, feature_names
from rules.tree import decisionTree_Ruleset, randomForest_Ruleset
from rules.ruleset import Ruleset
from rule_ranking.rank_rules import rank_rule_scores, rank_rule_scores_fav
from rule_ranking.eliminate_rules import eliminate_rules, eliminate_rules_fav_score
from model.generation.helpers.build_and_train_model import load_model



# Extract and evaluate rules from DNN
def cross_validate_rem_d(extract_rules_flag=False, evaluate_rules_flag=False):
    # Extract rules from model from each fold
    if extract_rules_flag:
        for fold in range(0, N_FOLDS):
            # Path to extracted rules from that fold
            extracted_rules_file_path = n_fold_rules_fp(fold)

            # Path to neural network model for this fold
            model_file_path = n_fold_model_fp(fold)

            X_train = np.load(N_FOLD_CV_SPLIT_X_train_data_FP(fold))
            X_test = np.load(N_FOLD_CV_SPLIT_X_test_data_FP(fold))
            y_train = np.load(N_FOLD_CV_SPLIT_y_train_data_FP(fold))
            y_test = np.load(N_FOLD_CV_SPLIT_y_test_data_FP(fold))

            # Extract rules
            nn_accuracy, nn_auc, rules, re_time, re_memory= dnn_re.run(X_train, y_train, X_test, y_test,
                                                                            model_file_path)

            # Save rules extracted
            print('Saving fold %d/%d rules extracted...' % (fold, N_FOLDS), end='', flush=True)
            with open(extracted_rules_file_path, 'wb') as rules_file:
                pickle.dump(rules, rules_file)
            print('done')

            # Save rule extraction time and memory usage
            print('Saving fold %d/%d results...' % (fold, N_FOLDS), end='', flush=True)
            # Initialise empty results file
            if fold == 0:
                pd.DataFrame(data=[], columns=['fold']).to_csv(N_FOLD_RESULTS_FP, index=False)

            results_df = pd.read_csv(N_FOLD_RESULTS_FP)
            row_index = fold
            results_df.loc[row_index, 'fold'] = fold
            results_df.loc[row_index, 'nn_acc'] = nn_accuracy
            results_df.loc[row_index, 'nn_auc'] = nn_auc
            results_df.loc[row_index, 're_time (sec)'] = re_time
            results_df.loc[row_index, 're_memory (MB)'] = re_memory
            results_df.to_csv(N_FOLD_RESULTS_FP, index=False)
            print('done')

    # Compute cross-validated results
    if evaluate_rules_flag:
        for fold in range(0, N_FOLDS):
            # Get train and test data folds
            train_index, test_index = load_split_indices(N_FOLD_CV_SPLIT_INDICIES_FP, fold_index=fold)

            X_test = np.load(N_FOLD_CV_SPLIT_X_test_data_FP(fold))
            y_test = np.load(N_FOLD_CV_SPLIT_y_test_data_FP(fold))

            # Path to neural network model for this fold
            model_file_path = n_fold_model_fp(fold)

            # Load extracted rules from disk
            print('Loading extracted rules from disk for fold %d/%d...' % (fold, N_FOLDS), end='', flush=True)
            with open(n_fold_rules_fp(fold), 'rb') as rules_file:
                rules = pickle.load(rules_file)
            print('done')

            # Save labels to labels.csv:
            # label - True data labels
            label_data = {'id': test_index,
                          'true_labels': y_test}
            # label - Neural network data labels. Use NN to predict X_test
            nn_model = load_model(model_file_path)

            nn_predictions = np.argmax(nn_model.predict(X_test), axis=1)
            label_data['nn_labels'] = nn_predictions
            # label_data['nn_labels'] = nn_model.predict(X_test)


            # label - Rule extraction labels
            rule_predictions = predict(rules, X_test)
            label_data['rule_%s_labels' % RULE_EXTRACTOR.mode] = rule_predictions
            pd.DataFrame(data=label_data).to_csv(LABEL_FP, index=False)

            # Evaluate rules
            print('Evaulating rules extracted from fold %d/%d...' % (fold, N_FOLDS), end='', flush=True)
            re_results = evaluate(rules, LABEL_FP)
            print('done')

            # Save rule extraction evaulation results
            row_index = fold
            results_df = pd.read_csv(N_FOLD_RESULTS_FP)
            results_df.loc[row_index, 're_acc'] = re_results['acc']
            results_df.loc[row_index, 're_auc'] = re_results['aucr']
            results_df.loc[row_index, 're_fid'] = re_results['fid']
            results_df.loc[row_index, 'rules_num'] = sum(re_results['n_rules_per_class'])
            avg_rule_length = np.array(re_results['av_n_terms_per_rule'])
            avg_rule_length *= np.array(re_results['n_rules_per_class'])
            avg_rule_length = sum(avg_rule_length)
            avg_rule_length /= sum(re_results['n_rules_per_class'])
            results_df.loc[row_index, 'rules_av_len'] = avg_rule_length

            if fold == N_FOLDS - 1:
                results_df.iloc[:, 1:] = results_df.round(3)
                results_df.loc[N_FOLDS, "fold"] = "average"
                results_df.iloc[N_FOLDS, 1:] = results_df.mean().round(3)

            results_df = results_df[
                ["fold", "nn_acc", "nn_auc", "re_acc", "re_auc", "re_fid", "re_time (sec)", "re_memory (MB)", "rules_num",
                 "rules_av_len"]]

            results_df.to_csv(N_FOLD_RESULTS_FP, index=False)


# rank the rules, eliminates x% of the lowest rank ones (based on coverage, accuracy and length),
# and cross validates the remaining ones
def cross_validate_rem_d_ranking_elimination(rank_rules_flag=False, rule_elimination=False, percentage=0,
                                   evaluate_rules_flag=False):
    if rank_rules_flag:
        for fold in range(0, N_FOLDS):
            X_train = np.load(N_FOLD_CV_SPLIT_X_train_data_FP(fold))
            y_train = np.load(N_FOLD_CV_SPLIT_y_train_data_FP(fold))

            extracted_rules_file_path = n_fold_rules_fp(fold)

            with open(extracted_rules_file_path, 'rb') as rules_file:
                rules = pickle.load(rules_file)

            for rule in rules:
                rank_rule_scores(rule, X_train, y_train, use_rl=True)

            clear_file(extracted_rules_file_path)
            print('Saving fold %d/%d rules after scoring...' % (fold, N_FOLDS), end='', flush=True)
            with open(extracted_rules_file_path, 'wb') as rules_file:
                pickle.dump(rules, rules_file)

    if rule_elimination:
        for fold in range(0, N_FOLDS):
            extracted_rules_file_path = n_fold_rules_fp(fold)
            remaining_rules = eliminate_rules(extracted_rules_file_path, percentage)

            # Save remaining rules
            print('Saving fold %d/%d remaining rules ...' % (fold, N_FOLDS), end='', flush=True)
            with open(n_fold_rules_fp_remaining(N_FOLD_RULES_REMAINING_DP, fold)(percentage), 'wb') as rules_file:
                pickle.dump(remaining_rules, rules_file)
            print('done')

    if evaluate_rules_flag:
        for fold in range(0, N_FOLDS):
            train_index, test_index = load_split_indices(N_FOLD_CV_SPLIT_INDICIES_FP, fold_index=fold)

            X_test = np.load(N_FOLD_CV_SPLIT_X_test_data_FP(fold))
            y_test = np.load(N_FOLD_CV_SPLIT_y_test_data_FP(fold))

            model_file_path = n_fold_model_fp(fold)

            print('Loading extracted rules from disk for fold %d/%d...' % (fold, N_FOLDS), end='', flush=True)
            with open(n_fold_rules_fp_remaining(N_FOLD_RULES_REMAINING_DP, fold)(percentage), 'rb') as rules_file:
                rules = pickle.load(rules_file)
            print('done')

            # Initialise empty results file
            if fold == 0:
                pd.DataFrame(data=[], columns=['fold']).to_csv(N_FOLD_RESULTS_FP_REMAINING(percentage), index=False)

            results_df = pd.read_csv(N_FOLD_RESULTS_FP_REMAINING(percentage))
            row_index = fold
            results_df.loc[row_index, 'fold'] = fold

            label_data = {'id': test_index,
                          'true_labels': y_test}
            nn_model = load_model(model_file_path)
            nn_predictions = np.argmax(nn_model.predict(X_test), axis=1)
            label_data['nn_labels'] = nn_predictions
            rule_predictions = predict(rules, X_test)
            label_data['rule_%s_labels' % RULE_EXTRACTOR.mode] = rule_predictions
            pd.DataFrame(data=label_data).to_csv(LABEL_FP, index=False)

            # Evaluate rules
            print('Evaulating rules remained from fold %d/%d...' % (fold, N_FOLDS), end='', flush=True)
            re_results = evaluate(rules, LABEL_FP)
            print('done')

            # Save rule extraction evaulation results
            row_index = fold
            results_df.loc[row_index, 're_acc'] = re_results['acc']
            results_df.loc[row_index, 're_auc'] = re_results['aucr']
            results_df.loc[row_index, 're_fid'] = re_results['fid']
            results_df.loc[row_index, 'rules_num'] = sum(re_results['n_rules_per_class'])
            avg_rule_length = np.array(re_results['av_n_terms_per_rule'])
            avg_rule_length *= np.array(re_results['n_rules_per_class'])
            avg_rule_length = sum(avg_rule_length)
            avg_rule_length /= sum(re_results['n_rules_per_class'])
            results_df.loc[row_index, 'rules_av_len'] = avg_rule_length

            if fold == N_FOLDS - 1:
                results_df.iloc[:, 1:] = results_df.round(3)
                results_df.loc[N_FOLDS, "fold"] = "average"
                results_df.iloc[N_FOLDS, 1:] = results_df.mean().round(3)


            results_df = results_df[["fold", "re_acc", "re_auc", "rules_num", "rules_av_len"]]

            results_df.to_csv(N_FOLD_RESULTS_FP_REMAINING(percentage), index=False)


# rank the rules, eliminates x% of the lowest rank ones (based on coverage, accuracy, length and inclusion
# of favourite features), and cross validates the remaining ones
def cross_validate_rem_d_fav_ranking_elimination(favourite_features=[], rank_rules_fav_flag=False,
                                                 rule_elimination=False, percentage=0):
    if rank_rules_fav_flag:
        for fold in range(0, N_FOLDS):
            extracted_rules_file_path = n_fold_rules_fp(fold)

            with open(extracted_rules_file_path, 'rb') as rules_file:
                rules = pickle.load(rules_file)

            data_df = pd.read_csv(DATA_FP)
            features_name = list(data_df.columns)

            for rule in rules:
                rank_rule_scores_fav(rule, features_name, favourite_features)

            clear_file(extracted_rules_file_path)
            print('Saving fold %d/%d rules after scoring...' % (fold, N_FOLDS), end='', flush=True)
            with open(extracted_rules_file_path, 'wb') as rules_file:
                pickle.dump(rules, rules_file)

    if rule_elimination:
        for fold in range(0, N_FOLDS):
            extracted_rules_file_path = n_fold_rules_fp(fold)
            remaining_rules = eliminate_rules_fav_score(extracted_rules_file_path, percentage)

            # Save remaining rules
            print('Saving fold %d/%d remaining rules ...' % (fold, N_FOLDS), end='', flush=True)
            with open(n_fold_rules_fp_remaining(N_FOLD_RULES_REMAINING_DP, fold)(percentage), 'wb') as rules_file:
                pickle.dump(remaining_rules, rules_file)
            print('done')



# Extract and evaluate rules from decision tree (DT=True) or random forest (DT=False)
def cross_validated_rem_t(X, y, extract_evaluate_rules_flag=False, DT=False):
    if extract_evaluate_rules_flag:
        for fold in range(0, N_FOLDS):
            train_index, test_index = load_split_indices(N_FOLD_CV_SPLIT_INDICIES_FP, fold_index=fold)
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]


            main_df = pd.read_csv(DATA_FP)
            main_df = main_df.drop([DATASET_INFO.target_col], axis=1)

            feat_list = list(main_df.columns)
            feature_names_to_id_map = dict(zip(feat_list, range(len(feat_list))))
            # for key in feature_names_to_id_map:
            #     feature_names_to_id_map[key] += 1000

            max_depth = None
            n_estimators = 20

            if DT:
                extracted_rules_file_path = n_fold_rules_DT_fp(fold)
                accuracy, auc, rules = tree_re.run_dt(X_train, y_train, X_test, y_test, feature_names_to_id_map, DATASET_INFO.output_classes, max_depth)
                N_FOLD_RESULTS_tree_FP = N_FOLD_RESULTS_DT_FP
            else:
                extracted_rules_file_path = n_fold_rules_RF_fp(fold)
                accuracy, auc, rules = tree_re.run_rf(X_train, y_train, X_test, y_test, feature_names_to_id_map, DATASET_INFO.output_classes, n_estimators, max_depth)
                N_FOLD_RESULTS_tree_FP = N_FOLD_RESULTS_RF_FP


            # Save rules extracted
            print('Saving fold %d/%d rules extracted...' % (fold, N_FOLDS), end='', flush=True)
            with open(extracted_rules_file_path, 'wb') as rules_file:
                pickle.dump(rules, rules_file)
            print('done')

            # Save labels to labels.csv:
            # label - True data labels
            label_data = {'id': test_index,
                          'true_labels': y_test}

            # label - Rule extraction labels
            rule_predictions = predict(rules, X_test)
            label_data['rule_labels'] = rule_predictions
            pd.DataFrame(data=label_data).to_csv(LABEL_FP, index=False)

            print('Evaulating rules extracted from fold %d/%d...' % (fold, N_FOLDS), end='', flush=True)
            re_results = evaluate_tree_rules(rules, LABEL_FP)
            print('done')

            # Initialise empty results file
            if fold == 0:
                pd.DataFrame(data=[], columns=['fold']).to_csv(N_FOLD_RESULTS_tree_FP, index=False)

            results_df = pd.read_csv(N_FOLD_RESULTS_tree_FP)
            row_index = fold
            results_df.loc[row_index, 'fold'] = fold
            results_df.loc[row_index, 're_acc'] = re_results['acc']
            results_df.loc[row_index, 're_auc'] = re_results['auct']
            results_df.loc[row_index, 'rules_num'] = sum(re_results['n_rules_per_class'])
            avg_rule_length = np.array(re_results['av_n_terms_per_rule'])
            avg_rule_length *= np.array(re_results['n_rules_per_class'])
            avg_rule_length = sum(avg_rule_length)
            avg_rule_length /= sum(re_results['n_rules_per_class'])
            results_df.loc[row_index, 'rules_av_len'] = avg_rule_length


            if fold == N_FOLDS - 1:
                results_df.iloc[:, 1:] = results_df.round(3)
                results_df.loc[N_FOLDS, "fold"] = "average"
                results_df.iloc[N_FOLDS, 1:] = results_df.mean().round(3)


            results_df = results_df[["fold",  "re_acc", "re_auc", "rules_num", "rules_av_len"]]

            results_df.to_csv(N_FOLD_RESULTS_tree_FP, index=False)


# Combines the rules from DNN and decision tree (DT=True) or random forest (DT=False) and cross validates
# the new combined ruleset
def cross_validate_combined_rem_d_rem_t(combine_rules_flag=False, DT=True, evaluate_rules_flag=False):
    if combine_rules_flag:
        for fold in range(0, N_FOLDS):
            with open(n_fold_rules_fp(fold), 'rb') as rules_file:
                rules_dnn = pickle.load(rules_file)

            if DT:
                combined_rules_file_path = n_fold_rules_DT_COMB_fp(fold)
                with open(n_fold_rules_DT_fp(fold), 'rb') as rules_file:
                    rules_tree = pickle.load(rules_file)
            else:
                combined_rules_file_path = n_fold_rules_RF_COMB_fp(fold)
                with open(n_fold_rules_RF_fp(fold), 'rb') as rules_file:
                    rules_tree = pickle.load(rules_file)

            combined_rules = Ruleset(rules_dnn).combine_ruleset(Ruleset(rules_tree))

            # Save rules combined
            print('Saving fold %d/%d rules combined...' % (fold, N_FOLDS), end='', flush=True)
            with open(combined_rules_file_path, 'wb') as rules_file:
                pickle.dump(combined_rules, rules_file)
            print('done')


    if evaluate_rules_flag:
        for fold in range(0, N_FOLDS):
            train_index, test_index = load_split_indices(N_FOLD_CV_SPLIT_INDICIES_FP, fold_index=fold)
            X_test = np.load(N_FOLD_CV_SPLIT_X_test_data_FP(fold))
            y_test = np.load(N_FOLD_CV_SPLIT_y_test_data_FP(fold))

            combined_rules_file_path = n_fold_rules_DT_COMB_fp(fold)

            print('Loading extracted rules from disk for fold %d/%d...' % (fold, N_FOLDS), end='', flush=True)
            with open(combined_rules_file_path, 'rb') as rules_file:
                rules = pickle.load(rules_file)
            print('done')

            # Save labels to labels.csv:
            # label - True data labels
            label_data = {'id': test_index,
                          'true_labels': y_test}

            # label - Rule extraction labels
            rule_predictions = predict(rules, X_test)
            label_data['rule_labels'] = rule_predictions
            pd.DataFrame(data=label_data).to_csv(LABEL_FP, index=False)

            # Evaluate rules
            print('Evaulating rules extracted from fold %d/%d...' % (fold, N_FOLDS), end='', flush=True)
            re_results = evaluate_tree_rules(rules, LABEL_FP)
            print('done')


            # Initialise empty results file
            if fold == 0:
                pd.DataFrame(data=[], columns=['fold']).to_csv(N_FOLD_RESULTS_DT_COMB_FP, index=False)

            results_df = pd.read_csv(N_FOLD_RESULTS_DT_COMB_FP)
            row_index = fold
            results_df.loc[row_index, 'fold'] = fold
            results_df.loc[row_index, 're_acc'] = re_results['acc']
            results_df.loc[row_index, 're_auc'] = re_results['auct']
            results_df.loc[row_index, 'rules_num'] = sum(re_results['n_rules_per_class'])
            avg_rule_length = np.array(re_results['av_n_terms_per_rule'])
            avg_rule_length *= np.array(re_results['n_rules_per_class'])
            avg_rule_length = sum(avg_rule_length)
            avg_rule_length /= sum(re_results['n_rules_per_class'])
            results_df.loc[row_index, 'rules_av_len'] = avg_rule_length

            if fold == N_FOLDS - 1:
                results_df.iloc[:, 1:] = results_df.round(3)
                results_df.loc[N_FOLDS, "fold"] = "average"
                results_df.iloc[N_FOLDS, 1:] = results_df.mean().round(3)

            results_df = results_df[["fold", "re_acc", "re_auc", "rules_num", "rules_av_len"]]

            results_df.to_csv(N_FOLD_RESULTS_DT_COMB_FP, index=False)

# cross validates decision tree
def cross_validate_dt(X, y, max_depth, flag=False):
    if flag:
        table = PrettyTable()
        table.field_names = ["Fold", "DT Accuracy", 'DT AUC']
        averages = np.array([0.0] * (len(table.field_names) - 1))

        for fold in range(0, N_FOLDS):
            train_index, test_index = load_split_indices(N_FOLD_CV_SPLIT_INDICIES_FP, fold_index=fold)
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            cw = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)))
            decision_tree = DecisionTreeClassifier(random_state=1, class_weight=cw, max_depth=max_depth)
            decision_tree = decision_tree.fit(X_train, y_train)
            predicted = decision_tree.predict(X_test)

            dt_accuracy = accuracy_score(y_test, predicted)
            fpr, tpr, thresholds = roc_curve(y_test, predicted)
            dt_auc= auc(fpr, tpr)

            new_row = [
                round(dt_accuracy, 3),
                round(dt_auc, 3)]
            table.add_row([fold] + new_row)
            averages += np.array(new_row) / N_FOLDS

        table.add_row(
            ["avg"] +
            list(map(lambda x: round(x, 3), averages))
        )
        print(table)

# cross validates random forest
def cross_validate_rf(X, y, estimator, max_depth, max_features, flag=False):
    if flag:
        table = PrettyTable()
        table.field_names = ["Fold", "RF Accuracy", "RF AUC"]
        averages = np.array([0.0] * (len(table.field_names) - 1))

        for fold in range(0, N_FOLDS):
            train_index, test_index = load_split_indices(N_FOLD_CV_SPLIT_INDICIES_FP, fold_index=fold)
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            cw = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)))
            random_forest = RandomForestClassifier(n_estimators=estimator, max_depth=max_depth,
                                                   max_features=max_features, random_state=1, class_weight=cw)
            random_forest = random_forest.fit(X_train, y_train)
            predicted = random_forest.predict(X_test)

            rf_accuracy = accuracy_score(y_test, predicted)
            fpr, tpr, thresholds = roc_curve(y_test, predicted)
            rf_auc= auc(fpr, tpr)

            new_row = [
                round(rf_accuracy, 3),
                round(rf_auc, 3)]
            table.add_row([fold] + new_row)
            averages += np.array(new_row) / N_FOLDS

        table.add_row(
            ["avg"] +
            list(map(lambda x: round(x, 3), averages))
        )
        print(table)