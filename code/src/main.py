from src import *
from functionality_helpers import *
from cross_validations import *


X, y = load_data(DATASET_INFO, DATA_FP)

# Splits the data into fold, find best architecture for the network and trains neural networks
# for each fold using the architecture
generate_data.run(X=X, y=y,
                  split_data_flag=False,
                  grid_search_flag=False,
                  find_best_initialisation_flag=False,
                  generate_fold_data_flag=False)



# Extract and evaluate rules from neural network
cross_validate_rem_d(extract_rules_flag=False, evaluate_rules_flag=False)

# Generates explanation for a prediction made by ruleset extracted from one of the folds on random
explain_prediction(flag=False)

# Lists ten top recurring feature for the ruleset extracted across five folds and for the ruleset extracted
# across five folds for each class separately as well as the frequency of the operator (> , <=) they appear with.
compute_top_recurring_features_across_folds(flag=False)

# Extract and evaluate rules from decision tree (DT=True) or random forest (DT=False)
cross_validated_rem_t(X, y, extract_evaluate_rules_flag=False, DT=True)

# Combines and evaluate the rules extracted from neural network and decision tree (DT=True) or random forest (DT=False)
cross_validate_combined_rem_d_rem_t(combine_rules_flag=False, DT=True, evaluate_rules_flag=False)

# Picks 4 features at random from the combination of rulesets extracted across folds
favourite_features = pick_random_features_across_folds(4, flag=False)
# Ranks the rules extracted from each fold. Based on the raking, 30% of lowest rank rules can be eliminated.
cross_validate_rem_d_ranking_elimination(rank_rules_flag=False, rule_elimination=False, percentage=0.3,
                                   evaluate_rules_flag=False)
# Shows the frequency of the favourite features in the combination of rulesets extracted across folds after
# eliminating 30% of the lowest rank rules.
compute_favourite_features_frequency_across_folds(percentage=0.3, fav_features=favourite_features, flag=False)
# Ranks the rules extracted from each fold considering the favourite features. Based on the raking, 30% of
# lowest rank rules can be eliminated.
cross_validate_rem_d_fav_ranking_elimination(favourite_features=favourite_features, rank_rules_fav_flag=False,
                                                 rule_elimination=False, percentage=0.3)
# Shows the frequency of the favourite features in the combination of rulesets extracted across folds after
# eliminating 30% of the lowest rank rules, where the ranking considers favourite features.
compute_favourite_features_frequency_across_folds(percentage=0.3, fav_features=favourite_features, flag=False)

# Measures the performance of decision tree across five fold
cross_validate_dt(X, y, max_depth=10, flag=False)
# Measures the performance of random forest  across five fold
cross_validate_rf(X, y, estimator=50, max_depth=10, max_features=0.5, flag=False)

clean_up()



