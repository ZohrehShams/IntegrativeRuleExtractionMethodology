"""
Build neural network models given number of nodes in each hidden layer
"""
import os
import tensorflow as tf
import numpy as np
import random as python_random
import sklearn

os.environ['PYTHONHASHSEED'] = str(1)
tf.random.set_seed(42)
np.random.seed(42)
python_random.seed(42)

from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score
from src import *

class LogitAUC(tf.keras.metrics.AUC):
    """
    Custom AUC metric that operates in logit activations (i.e. does not
    require them to be positive and will pass a softmax through them before
    computing the AUC)
    """
    def __init__(self, *args, **kwargs):
        super(LogitAUC, self).__init__(*args, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.where(
            tf.equal(tf.reduce_max(y_true, axis=-1, keepdims=True), y_true),
            1,
            0
        )
        y_pred = tf.where(
            tf.equal(tf.reduce_max(y_pred, axis=-1, keepdims=True), y_pred),
            1,
            0
        )

        super(LogitAUC, self).update_state(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
        )






def load_model(path):
    """
    Wrapper around tf.keras.models.load_model that includes all custom layers
    and metrics we are including in our model when serializing.

    :param str path: The path of the model checkpoint we want to load.
    :returns tf.keras.Model: Model object corresponding to loaded checkpoint.
    """
    return tf.keras.models.load_model(
        path,
        custom_objects={
            "LogitAUC": LogitAUC
        },
    )



def create_model(layer_1, layer_2):
    input_layer = tf.keras.layers.Input(shape=(DATASET_INFO.n_inputs,))

    # Hidden layer 1
    hidden_layer_1 = tf.keras.layers.Dense(layer_1, activation='tanh')(input_layer)

    # Hidden layer 2
    hidden_layer_2 = tf.keras.layers.Dense(layer_2, activation='tanh')(hidden_layer_1)

    output_layer = tf.keras.layers.Dense(DATASET_INFO.n_outputs, activation='softmax')(hidden_layer_2)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', LogitAUC(name='auc')]
                  )

    return model


def build_and_train_model\
                (X_train, y_train, X_test, y_test, batch_size, epochs, layer_1, layer_2, model_file_path,
                          with_best_initilisation_flag=False):
    """

    Args:
        X_train:
        y_train:
        X_test:
        y_test:
        batch_size:

        epochs:
        layer_1:
        layer_2:
        model_file_path: path to store trained nn model
        with_best_initilisation_flag: if true, use initialisation saved as best_initialisation.h5

    Returns:
        model_accuracy: accuracy of nn model
        nn_predictions: predictions made by nn used for rule extraction
    """

    # To get 2 node output make y categorical
    y_train_cat, y_test_cat = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)

    # Weight classes due to imbalanced dataset
    class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)))

    if with_best_initilisation_flag:
        # Use best saved initialisation found earlier
        print('Training neural network with best initialisation')
        best_initialisation_file_path = BEST_NN_INIT_FP
        # model = tf.keras.models.load_model(best_initialisation_file_path)
        model = load_model(best_initialisation_file_path)
    else:
        # Build and initialise new model
        # print('Training neural network with new random initialisation')
        model = create_model(layer_1, layer_2)
        model.save(TEMP_DIR + 'initialisation.h5')

    # Train Model
    model.fit(X_train,
              y_train_cat,
              class_weight=class_weights,
              epochs=epochs,
              batch_size=batch_size,
              verbose=0)

    # Evaluate Loss and Accuracy of the Model
    _, nn_accuracy, nn_AUC = model.evaluate(X_test, y_test_cat)

    # Save Trained Model
    model.save(model_file_path)

    return nn_accuracy, nn_AUC



def build_and_train_model_whole_dataset(X, y, batch_size, epochs, layer_1, layer_2, model_file_path,):

    # To get 2 node output make y categorical
    y_cat = tf.keras.utils.to_categorical(y)

    # Weight classes due to imbalanced dataset
    class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(y), y)))

    print('Training neural network with new random initialisation')
    model = create_model(layer_1, layer_2)
    model.save(TEMP_DIR + 'initialisation.h5')

    model.fit(X,
              y_cat,
              class_weight=class_weights,
              epochs=epochs,
              batch_size=batch_size,
              verbose=0)

    _, nn_accuracy, nn_AUC = model.evaluate(X, y_cat)

    # Save Trained Model
    model.save(model_file_path)

    return nn_accuracy



