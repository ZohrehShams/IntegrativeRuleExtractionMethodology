from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from sklearn.metrics import auc, roc_curve
from src import NN_INIT_GRID_RESULTS_FP
from model.generation.helpers.build_and_train_model import create_model




class CustomMetricKerasClassifier(KerasClassifier):
    """
    Helper class to wrap a Keras model and turn it into a sklearn classifier
    whose scoring function can be given by any metric in the Keras model.
    """

    def __init__(self, build_fn=None, metric_name='accuracy', **sk_params):
        """
        metric_name represents the name of a valid metric in the given model.
        """

        super(CustomMetricKerasClassifier, self).__init__(
            build_fn=build_fn,
            **sk_params
        )
        self.metric_name = metric_name

    def get_params(self, **params):  # pylint: disable=unused-argument
        """
        Gets parameters for this estimator.
        """
        res = super(CustomMetricKerasClassifier, self).get_params(**params)
        res.update({'metric_name': self.metric_name})
        return res

    def score(self, x, y, **kwargs):
        """
        Returns the requested metric on the given test data and labels.
        """
        y = np.searchsorted(self.classes_, y)
        kwargs = self.filter_sk_params(
            tf.keras.models.Sequential.evaluate,
            kwargs
        )
        outputs = self.model.evaluate(x, y, **kwargs)
        if not isinstance(outputs, list):
            outputs = [outputs]

        for name, output in zip(self.model.metrics_names, outputs):
            if name == self.metric_name:
                return output
        raise ValueError(
            'The model is not configured to compute metric with name '
            f'{self.metric_name}. All available metrics are '
            f'{self.model.metrics_names}.'
        )



def grid_search(X, y):
    """

    Args:
        X: input features
        y: target

    Returns:
        batch_size: best batch size
        epochs: best number of epochs
        layer_1: best number of neurons for layer 1 (first hidden layer)
        layer_2: best number of neurons for layer 2

    Perform a 5-folded grid search over the neural network hyper-parameters
    """
    batch_size = [16, 32, 64, 128]
    epochs = [50, 100, 150, 200]
    layer_1 = [128, 64, 32, 16]
    layer_2 = [64, 32, 16, 8]


    param_grid = dict(batch_size=batch_size,
                      epochs=epochs,
                      layer_1=layer_1,
                      layer_2=layer_2)


    model = CustomMetricKerasClassifier(
        build_fn=create_model,
        # Given class imbalance, we will score our fits based on AUC rather
        # than plain accuracy.
        metric_name='auc',
        verbose=0,
    )

    class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(y), y)))

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X, tf.keras.utils.to_categorical(y), class_weight=class_weights)

    # Write best results to file
    with open(NN_INIT_GRID_RESULTS_FP, 'w') as file:
        file.write("Best: %f using %s \n" % (grid_result.best_score_, grid_result.best_params_))

        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            file.write("%f (%f) with: %r\n" % (mean, stdev, param))

    print('Grid Search for hyper parameters complete.')
    print("Best: %f using %s \n" % (grid_result.best_score_, grid_result.best_params_))

    return grid_result.best_params_['batch_size'], grid_result.best_params_['epochs'], \
           grid_result.best_params_['layer_1'], grid_result.best_params_['layer_2']