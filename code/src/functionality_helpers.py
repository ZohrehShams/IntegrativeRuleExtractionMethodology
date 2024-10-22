import numpy as np
import pickle
import dnn_re
from evaluate_rules.predict_explain import predict_explain, print_explanation
from evaluate_rules.overlapping_features import features_recurrence_in_explanation
from src import *
from evaluate_rules.predict_explain import predict_explain, print_explanation
from evaluate_rules.overlapping_features import *
from rule_ranking.rank_rules import rank_rule_scores, rank_rule_scores_fav
from rule_ranking.eliminate_rules import eliminate_rules, eliminate_rules_fav_score
from model.generation.helpers.init_dataset_dir import clean_up, clear_file


# Extract ruleset from the entire dataset (no fold split) and saves them
def validate_rem_d(extract_rules_flag=False):
    if extract_rules_flag:
        X = np.load(N_FOLD_CV_SPLIT_X_data_FP)
        y = np.load(N_FOLD_CV_SPLIT_y_data_FP)

        # Extract rules
        nn_accuracy, nn_auc, rules, re_time, re_memory= dnn_re.run_whole_dataset(X, y, model_fp)

        for rule in rules:
            print(len(rule.premise))

        # Save rules extracted
        print('Saving rules extracted...', end='', flush=True)
        with open(rules_fp, 'wb') as rules_file:
            pickle.dump(rules, rules_file)
        print('done')

        # Save rule extraction time and memory usage
        print('Saving results...', end='', flush=True)

# Prints explanation for an instance generated by random sampling;
# also prints the frequency of features in the explanation
def explain_prediction_entire_data(flag=False):
    if flag:
        np.random.seed(110)
        instance = np.random.uniform(0, 1, 1004)

        with open(rules_fp, 'rb') as rules_file:
            rules = pickle.load(rules_file)

        prediction, explanation = predict_explain(rules, instance)
        print(print_explanation(prediction, explanation))
        print(features_recurrence_in_explanation(explanation))

def explain_prediction(flag=False):
    if flag:
        np.random.seed(114)
        instance = np.random.uniform(0, 1, 1004)
        fold = np.random.randint(5)

        with open(n_fold_rules_fp(fold), 'rb') as rules_file:
            rules = pickle.load(rules_file)

        prediction, explanation = predict_explain(rules, instance)
        print(print_explanation(prediction, explanation))
        print(features_recurrence_in_explanation(explanation))


# Prints the top 10 recurring features in the entire ruleset,
# as well as in the ruleset for each class,
# along with the frequency of operator for each of the top features
def compute_top_recurring_features(flag=False):
    if flag:
        with open(rules_fp, 'rb') as rules_file:
            rules = pickle.load(rules_file)

        print(features_recurrence(rules, DATA_FP, 10))
        print(features_recurrence_per_class(rules, DATA_FP, 10))
        print(top_features_operator_frequency_recurrence_per_class(rules, DATA_FP, 10))


# Prints the top 50 recurring features across the folds,
# as well as in the ruleset for each class,
# along with the frequency of operator for each of the top features
def compute_top_recurring_features_across_folds(flag=False):
    if flag:
        list_of_rules=[]
        for fold in range(0, N_FOLDS):
            with open(n_fold_rules_fp(fold), 'rb') as rules_file:
                rules = pickle.load(rules_file)
                list_of_rules.append(rules)

        print("features recurrence across folds:")
        features_recurrence_across_folds(list_of_rules, DATA_FP, 50)
        print('\n')
        print("features recurrence per class across folds %s" %(features_recurrence_per_class_across_folds(list_of_rules, DATA_FP, 50)))
        print('\n')
        print("top features operator frequency recurrence per class across folds %s" %(top_features_operator_frequency_recurrence_per_class_across_folds(list_of_rules, DATA_FP, 50)))


# Shows the frequency of the favourite features in the ruleset
def compute_favourite_features_frequency(rule_path, fav_features, flag=False):
    if flag:
        with open(rule_path, 'rb') as rules_file:
            rules = pickle.load(rules_file)
        fav_freq = fav_features_recurrence(rules, DATA_FP, fav_features)
        return fav_freq



# Shows the frequency of the favourite features in the ruleset
def compute_favourite_features_frequency_across_folds(percentage, fav_features, flag=False):
    if flag:
        list_of_rules = []
        for fold in range(0, N_FOLDS):
            with open(n_fold_rules_fp_remaining(N_FOLD_RULES_REMAINING_DP, fold)(percentage), 'rb') as rules_file:
                rules = pickle.load(rules_file)
                list_of_rules.append(rules)
        fav_freq = fav_features_recurrence_across_folds(list_of_rules, DATA_FP, fav_features)
        return fav_freq


#  Pick n features at random from the rulset extarcted from the entire dataset
def pick_random_features(n, flag=False):
    if flag:
        with open(rules_fp, 'rb') as rules_file:
            rules = pickle.load(rules_file)
        favourite_features = random_features_in_rules(rules, DATA_FP, n)
        return favourite_features


#  Pick n features at random from  the entire dataset
def pick_random_features_across_folds(n, flag=False):
    if flag:
        list_of_rules = []
        data_df = pd.read_csv(DATA_FP)
        features_name = list(data_df.columns)

        for fold in range(0, N_FOLDS):
            with open(n_fold_rules_fp(fold), 'rb') as rules_file:
                rules = pickle.load(rules_file)
                list_of_rules.append(rules)
        favourite_features = random_features_in_rules_across_folds(list_of_rules, DATA_FP, n)
        return favourite_features



# Ranks the rules extracted from the entire dataset with the option of factoring in favourite features
# in the ranking. Based on the raking, lowest rank rules can be eliminated. n is the percentage of rules
# that will be eliminated. n = 0.5 eliminates 50% of the rules.
def validate_rem_d_ranking_elimination(rank_rules_flag=False, rule_elimination=False, percentage=0):
    X = np.load(N_FOLD_CV_SPLIT_X_data_FP)
    y = np.load(N_FOLD_CV_SPLIT_y_data_FP)

    if rank_rules_flag:
        extracted_rules_file_path = rules_fp

        with open(extracted_rules_file_path, 'rb') as rules_file:
            rules = pickle.load(rules_file)

        for rule in rules:
            rank_rule_scores(rule, X, y, use_rl=True)

        clear_file(extracted_rules_file_path)
        print('Saving rules after scoring...', end='', flush=True)
        with open(extracted_rules_file_path, 'wb') as rules_file:
            pickle.dump(rules, rules_file)

    if rule_elimination:
        extracted_rules_file_path = rules_fp
        remaining_rules = eliminate_rules(extracted_rules_file_path, percentage)

        # Save remaining rules
        print('Saving remaining rules ...', end='', flush=True)
        with open(rules_fp_remaining(percentage), 'wb') as rules_file:
            pickle.dump(remaining_rules, rules_file)
        print('done')


def validate_rem_d_fav_ranking_elimination(favourite_features=[], rank_rules_fav_flag=False, rule_elimination=False,
                                           percentage=0):
    if rank_rules_fav_flag:
        extracted_rules_file_path = rules_fp

        with open(extracted_rules_file_path, 'rb') as rules_file:
            rules = pickle.load(rules_file)

        data_df = pd.read_csv(DATA_FP)
        features_name = list(data_df.columns)

        for rule in rules:
            rank_rule_scores_fav(rule, features_name, favourite_features)

        clear_file(extracted_rules_file_path)
        print('Saving rules after scoring...', end='', flush=True)
        with open(extracted_rules_file_path, 'wb') as rules_file:
            pickle.dump(rules, rules_file)

    if rule_elimination:
        extracted_rules_file_path = rules_fp
        remaining_rules = eliminate_rules_fav_score(extracted_rules_file_path, percentage)

        # Save remaining rules
        print('Saving remaining rules ...', end='', flush=True)
        with open(rules_fp_remaining(percentage), 'wb') as rules_file:
            pickle.dump(remaining_rules, rules_file)
        print('done')