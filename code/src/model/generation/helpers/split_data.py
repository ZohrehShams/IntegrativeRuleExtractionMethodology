"""
Split data and save indices of the split for reproducibility
"""

import pandas as pd
from sklearn.model_selection import ShuffleSplit  # Returns indices unlike train_test_split
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from model.generation.helpers.init_dataset_dir import create_directory, clear_file
from src import N_FOLD_CV_DP, N_FOLD_RULE_EX_MODE_DP, N_FOLD_RULES_DP, N_FOLD_CV_SPLIT_INDICIES_FP, \
    NN_INIT_SPLIT_INDICES_FP, N_FOLD_MODELS_DP


def save_split_indices(train_index, test_index, file_path):
    """
    Args:
        train_index: List of train indicies of data
        test_index: List of test indices of data
        file_path: File to save split indices

    Write list of indices to split_indices.txt

    File of the form with a train and test line for each fold
    train 0 1 0 2 ...
    test 3 4 6 ...
    train 1 5 2 ...
    test 6 8 9 ...
    ...

    """
    with open(file_path, 'a') as file:
        file.write('train ' + ' '.join([str(index) for index in train_index]) + '\n')
        file.write('test ' + ' '.join([str(index) for index in test_index]) + '\n')


def load_split_indices(file_path, fold_index=0):
    """
    Args:
        file_path: path to split indices file
        fold_index: index of the fold whose train and test indices you want

    Returns:
        train_index: list of integer indices for train data
        test_index: list of integer indices for test data

    File of the form
    train 0 1 0 2 ...
    test 3 4 6 ...
    train 1 5 2 ...
    test 6 8 9 ...
    ...
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        assert len(lines) >= (2 * fold_index) + 2, 'Error: not enough information in fold indices file %d < % d' \
                                                   % (len(lines), (2 * fold_index) + 2)

        train_index = lines[(fold_index * 2)].split(' ')[1:]
        test_index = lines[(fold_index * 2) + 1].split(' ')[1:]

        # Convert string indices to ints
        train_index = [int(i) for i in train_index]
        test_index = [int(i) for i in test_index]

    return train_index, test_index


def stratified_k_fold(X, y, n_folds):
    """

    Args:
        X: input features
        y: target
        n_folds: how many folds to split data into

    Split data into folds and saves indices in to data_split_indices.txt
    """
    # Make directory for
    create_directory(dir_path=N_FOLD_CV_DP)  # cross_validation/<n>_folds/
    create_directory(dir_path=N_FOLD_RULE_EX_MODE_DP)  # <n>_folds/rule_extraction/<ruleemode>/
    create_directory(dir_path=N_FOLD_RULES_DP)  # <n>_folds/rule_extraction/<ruleemode>/rules_extracted
    create_directory(dir_path=N_FOLD_MODELS_DP) # <n>_folds/trained_models

    # Initialise split indices file
    clear_file(N_FOLD_CV_SPLIT_INDICIES_FP)

    # Split data
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=12345)

    # Save indices
    for train_index, test_index in skf.split(X, y):
        save_split_indices(train_index=train_index,
                           test_index=test_index,
                           file_path=N_FOLD_CV_SPLIT_INDICIES_FP)

    print('Split data into %d folds.' % n_folds)


def train_test_split(X, y, test_size=0.2):
    """

    Args:
        X: input features
        y: target
        test_size: percentage of the data used for testing

    Returns:

    Single train test split of the data used for initilising the neural network
    """

    # Initialise split indices file
    clear_file(NN_INIT_SPLIT_INDICES_FP)

    # Split data
    rs = ShuffleSplit(n_splits=2, test_size=test_size, random_state=42)

    for train_index, test_index in rs.split(X):
        save_split_indices(train_index=train_index,
                           test_index=test_index,
                           file_path=NN_INIT_SPLIT_INDICES_FP)

        # Only want 1 split
        break

    print('Split data into train/test split for initialisation.')


def load_data(dataset_info, data_path):
    """

    Args:
        dataset_info: meta data about dataset e.g. name, target col
        data_path: path to data.csv

    Returns:
        X: data input features
        y: data target
    """
    data = pd.read_csv(data_path)

    X = data.drop([dataset_info.target_col], axis=1).values
    y = data[dataset_info.target_col].values

    return X, y


def feature_names(data_path):
    data = pd.read_csv(data_path)
    return list(data.columns)