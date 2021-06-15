from collections import namedtuple

from configurations import get_configuration
from extract_rules.modified_deep_red_C5 import extract_rules as MOD_DeepRED_C5


DATASET_INFO = get_configuration(dataset_name='breast_cancer_uci')
N_FOLDS = 5

# Algorithm used for Rule Extraction
RuleExMode = namedtuple('RuleExMode', 'mode run')
RULE_EXTRACTOR = RuleExMode(mode='MOD_DeepRED_C5', run=MOD_DeepRED_C5)


# Parameters for case study I and II
# BATCH_SIZE = 16
# EPOCHS = 50
# LAYER_1 = 128
# LAYER_2 = 16


# Parameters for demo
BATCH_SIZE = 32
EPOCHS = 100
LAYER_1 = 64
LAYER_2 = 16

# --------------------------------------------- File paths -----------------------------------------------------------
# NB: FP/fp = file path, DP/dp = directory path

DATASET_DP = '../data/%s/' % DATASET_INFO.name
DATA_FP = DATASET_DP + 'data.csv'

# <dataset_name>/cross_validation/<n>_folds/
CV_DP = DATASET_DP + 'cross_validation/'
N_FOLD_CV_DP = CV_DP + '%d_folds/' % N_FOLDS
N_FOLD_CV_SPLIT_INDICIES_FP = N_FOLD_CV_DP + 'data_split_indices.txt'
N_FOLD_CV_SPLIT_X_train_data_FP = lambda fold: N_FOLD_CV_DP + 'fold_%d_X_train.npy' % fold
N_FOLD_CV_SPLIT_y_train_data_FP = lambda fold: N_FOLD_CV_DP + 'fold_%d_y_train.npy' % fold
N_FOLD_CV_SPLIT_X_test_data_FP = lambda fold: N_FOLD_CV_DP + 'fold_%d_X_test.npy' % fold
N_FOLD_CV_SPLIT_y_test_data_FP = lambda fold: N_FOLD_CV_DP + 'fold_%d_y_test.npy' % fold
N_FOLD_CV_SPLIT_X_data_FP = N_FOLD_CV_DP + 'X.npy'
N_FOLD_CV_SPLIT_y_data_FP = N_FOLD_CV_DP + 'y.npy'

# <dataset_name>/cross_validation/<n>_folds/rule_extraction/<rule_ex_mode>/rules_extracted/
N_FOLD_RULE_EX_MODE_DP = N_FOLD_CV_DP + 'rule_extraction/' + RULE_EXTRACTOR.mode + '/'
N_FOLD_RESULTS_FP = N_FOLD_RULE_EX_MODE_DP + 'results.csv'
N_FOLD_RULES_DP = N_FOLD_RULE_EX_MODE_DP + 'rules_extracted/'
n_fold_rules_fp = lambda fold: N_FOLD_RULES_DP + 'fold_%d.rules' % fold
rules_fp = N_FOLD_RULES_DP + 'fold.rules'
N_FOLD_RESULTS_FP_whole = N_FOLD_RULE_EX_MODE_DP + 'results_whole.csv'

# <dataset_name>/cross_validation/<n>_folds/trained_models/
N_FOLD_MODELS_DP = N_FOLD_CV_DP + 'trained_models/'
n_fold_model_fp = lambda fold: N_FOLD_MODELS_DP + 'fold_%d_model.h5' % fold
model_fp = N_FOLD_MODELS_DP + 'model.h5'

# <dataset_name>/neural_network_initialisation/
NN_INIT_DP = DATASET_DP + 'neural_network_initialisation/'
NN_INIT_GRID_RESULTS_FP = NN_INIT_DP + 'grid_search_results.txt'
NN_INIT_SPLIT_INDICES_FP = NN_INIT_DP + 'data_split_indices.txt'
NN_INIT_RE_RESULTS_FP = NN_INIT_DP + 're_results.csv'
BEST_NN_INIT_FP = NN_INIT_DP + 'best_initialisation.h5'

# Store temporary files during program execution
TEMP_DIR = 'src/temp/'
LABEL_FP = TEMP_DIR + 'labels.csv'


N_FOLD_RULE_EX_DT_DP = N_FOLD_CV_DP + 'rule_extraction/MOD_DecisionTree/'
N_FOLD_RESULTS_DT_FP = N_FOLD_RULE_EX_DT_DP + 'results.csv'
N_FOLD_RULES_DT_DP = N_FOLD_RULE_EX_DT_DP + 'rules_extracted/'
n_fold_rules_DT_fp = lambda fold: N_FOLD_RULES_DT_DP + 'fold_%d.rules' % fold


N_FOLD_RULE_EX_RF_DP = N_FOLD_CV_DP + 'rule_extraction/MOD_RandomForest/'
N_FOLD_RESULTS_RF_FP = N_FOLD_RULE_EX_RF_DP + 'results.csv'
N_FOLD_RULES_RF_DP = N_FOLD_RULE_EX_RF_DP + 'rules_extracted/'
n_fold_rules_RF_fp = lambda fold: N_FOLD_RULES_RF_DP + 'fold_%d.rules' % fold


N_FOLD_RULE_EX_DT_COMB_DP = N_FOLD_CV_DP + 'rule_extraction/MOD_DT_Combined/'
N_FOLD_RESULTS_DT_COMB_FP = N_FOLD_RULE_EX_DT_COMB_DP + 'results.csv'
N_FOLD_RULES_DT_COMB_DP = N_FOLD_RULE_EX_DT_COMB_DP + 'rules_extracted/'
n_fold_rules_DT_COMB_fp = lambda fold: N_FOLD_RULES_DT_COMB_DP + 'fold_%d.rules' % fold

N_FOLD_RULE_EX_RF_COMB_DP = N_FOLD_CV_DP + 'rule_extraction/MOD_RF_Combined/'
N_FOLD_RESULTS_RF_COMB_FP = N_FOLD_RULE_EX_RF_COMB_DP + 'results.csv'
N_FOLD_RULES_RF_COMB_DP = N_FOLD_RULE_EX_RF_COMB_DP + 'rules_extracted/'
n_fold_rules_RF_COMB_fp = lambda fold: N_FOLD_RULES_RF_COMB_DP + 'fold_%d.rules' % fold


N_FOLD_RULES_REMAINING_DP = lambda reduction_percentage: N_FOLD_RULE_EX_MODE_DP + 'rules_remaining_after_%s_reduction/' % reduction_percentage
n_fold_rules_fp_remaining = lambda path, fold: lambda reduction_percentage: path(reduction_percentage) + 'fold_%d_remaining.rules' % fold
N_FOLD_RESULTS_FP_REMAINING = lambda reduction_percentage: N_FOLD_RULE_EX_MODE_DP + 'results_%s_reduction.csv' % reduction_percentage
rules_fp_remaining = lambda reduction_percentage: N_FOLD_RULES_REMAINING_DP(reduction_percentage) + "fold.rules"


