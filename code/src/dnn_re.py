from collections import namedtuple

import memory_profiler
import numpy as np
import pandas as pd
import tensorflow as tf
# from keras.models import load_model
from model.generation.helpers.build_and_train_model import load_model
import pickle

import tracemalloc

# from evaluate_rules.evaluate import evaluate
from evaluate_rules.predict import predict
from model.model import Model
from src import TEMP_DIR, DATASET_INFO, RULE_EXTRACTOR
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical as to_categorical_sk
from sklearn.metrics import confusion_matrix, auc, roc_curve


# Data is made up of X (input), y (target)
DataValues = namedtuple('DataValues', 'X y')

# run the dnn for training data
def run(X_train, y_train, X_test, y_test, model_file_path):
    import time
    _, nn_accuracy, nn_AUC = load_model(model_file_path).evaluate(X_test, to_categorical_sk(y_test))

    train_data = DataValues(X=X_train, y=y_train)
    test_data = DataValues(X=X_test, y=y_test)

    # Initialise NN Model object
    NN_model = Model(model_path=model_file_path,
                     output_classes=DATASET_INFO.output_classes,
                     train_data=train_data,
                     test_data=test_data,
                     activations_path=TEMP_DIR + 'activations/')

    # Rule Extraction
    start_time= time.time()
    tracemalloc.start()
    rules = RULE_EXTRACTOR.run(NN_model)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    # Use rules for prediction
    NN_model.set_rules(rules)

    # Rule extraction time and memory usage
    time = end_time - start_time
    # converting KiB to MB
    memory = current* (1024 / 1000000)

    return nn_accuracy, nn_AUC, rules, time, memory
    

# run the dnn for the whole data
def run_whole_dataset(X, y, model_file_path):
    import time

    _, nn_accuracy, nn_AUC = load_model(model_file_path).evaluate(X, to_categorical_sk(y))


    train_data = DataValues(X=X, y=y)
    # This is never used
    test_data = DataValues(X=X, y=y)

    # Initialise NN Model object
    NN_model = Model(model_path=model_file_path,
                     output_classes=DATASET_INFO.output_classes,
                     train_data=train_data,
                     test_data=test_data,
                     activations_path=TEMP_DIR + 'activations/')

    # Rule Extraction
    start_time= time.time()
    tracemalloc.start()
    rules = RULE_EXTRACTOR.run(NN_model)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time= time.time()

    # Use rules for prediction
    NN_model.set_rules(rules)

    # Rule extraction time and memory usage
    time = end_time - start_time
    # converting KiB to MB
    memory = current * (1024 / 1000000)

    return nn_accuracy, nn_AUC, rules, time, memory
    

