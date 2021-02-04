# REM
Source code for paper “REM: An Integrative Rule Extraction Methodology for Explainable Data Analysis in Healthcare”


# Overview
Deep learning models are receiving increasing attention in clinical decision-making, however the lack of explainability impedes their deployment in day-to-day clinical practice. We propose REM, an explainable methodology for extracting rules from deep neural networks and combining them with other data-driven and knowledge-driven rules. This allows integrating machine learning and reasoning for investigating applied and basic biological research questions. We evaluate the utility of REM on the predictive tasks of classifying histological and immunohistochemical breast cancer subtypes from genotype and phenotype data.
We demonstrate that REM efficiently extracts accurate, comprehensible and, biologically relevant rulesets from deep neural networks that can be readily integrated with rulesets obtained from tree-based approaches. REM provides explanation facilities for predictions and enables the clinicians to validate and calibrate the extracted rulesets with their domain knowledge. With these functionalities, REM caters for a novel and direct human-in-the-loop approach in clinical decision making.


# Instruction for reproducing the results

Results reported in Table 1: 
- They are average of results reported in Table 5 and 6.  

Results reported in Table 2:
- Results reported in the second row are average of results in Table 7.
- Results reported in the third row are average of results in Table 8.

Results reported in Table 4:
- Set the dataset_name to “MB-1004-GE-2Hist” in __init__.py and run main.py by setting flags in the following to True:
    - cross_validate_dt(X, y, max_depth=10, flag=False)
    - cross_validate_rf(X, y, estimator=50, max_depth=10, max_features=0.5, flag=False)

Results reported in Table 5:
- Set the dataset_name to “MB-1004-GE-2Hist” in __init__.py and run main.py by setting the two flags appearing in the following to True: cross_validate_rem_d(extract_rules_flag=False, evaluate_rules_flag=False)

Results reported in Table 6:
- Set the dataset_name to “MB-GE-ER” in __init__.py and run main.py by setting the two flags appearing in the following to True: cross_validate_rem_d(extract_rules_flag=False, evaluate_rules_flag=False)

Results reported in Table 7: 
- Set the dataset_name to “MB-ClinP-ER” in __init__.py and run main.py by setting the two flags appearing in the following to True: cross_validate_rem_t(X, y, extract_evaluate_rules_flag=False, DT=False)

Results reported in Table 8:
- Set the dataset_name to “MB-GE-ClinP-ER” in __init__.py and run main.py by setting all flags in the following to True: cross_validate_combined_rem_d_rem_t(combine_rules_flag=False, DT=False, evaluate_rules_flag=False)

Results reported in Table 9:
- Set the dataset_name to “MB-1004-GE-2Hist” in __init__.py  and un main.py by setting the flag appearing in compute_top_reccuring_features_across_folds(flag=False) to True

Results reported in Figure 4: 
- Set the dataset_name to “MB-GE-ER” in __init__.py and run main.py by setting all the flags in the following to true:
    - favourite_features = pick_random_features_across_folds(4, flag=Flase)
    - cross_validate_rem_d_ranking_elimination(rank_rules_flag=False, rule_elimination=False, percentage=0.3, evaluate_rules_flag=False)
    - compute_favourite_features_frequency_across_folds(percentage=0.3, fav_features=favourite_features, flag=False)
    - cross_validate_rem_d_fav_ranking_elimination(favourite_features=favourite_features, rank_rules_fav_flag=False, rule_elimination=False, percentage=0.3)
    - compute_favourite_features_frequency_across_folds(percentage=0.3, fav_features=favourite_features, flag=False)
- Use src/rule_ranking/rule_ranking_vis.py scripts to generate the bar plot in Figure 4.

Results reported in Figure 5: 
- Repeat the procedure described for Figure 4, with varying values for x set in rank_rule_scores_fav in src/rule_ranking/rank_rules.py, by rerunning the following with all flags set to True:
    - cross_validate_rem_d_fav_ranking_elimination(favourite_features=favourite_features, rank_rules_fav_flag=False, rule_elimination=False, percentage=0.3)
    - compute_favourite_features_frequency_across_folds(percentage=0.3, fav_features=favourite_features, flag=False)


