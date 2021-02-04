"""
Represent trained Neural Network model
"""


import pandas as pd
# import keras.models as keras
import tensorflow.keras.models as keras
from model.generation.helpers.build_and_train_model import load_model

class Model:
    """
    Represent trained neural network model
    """

    def __init__(self, model_path, output_classes, train_data, test_data, activations_path,):
        model_path = model_path
        # self.model: keras.Model = keras.load_model(model_path, custom_objects={"LogitAUC": LogitAUC})
        self.model: keras.Model = load_model(model_path)
        self.activations_path = activations_path

        # self.col_names = col_names
        self.output_classes = output_classes

        self.rules = set()  # DNF rule for each output class
        self.n_layers = len(self.model.layers)

        self.train_data = train_data
        self.test_data = test_data

        self.__compute_layerwise_activations()

    def __compute_layerwise_activations(self):
        """
        Store sampled activations for each layer in CSV files
        """
        # todo make this method work for func and non func keras models
        # Input features of training data
        data_x = self.train_data.X

        # Sample network at each layer
        for layer_index in range(0, self.n_layers):
            out_shape = self.model.layers[layer_index].output_shape
            if isinstance(out_shape, list):
                if len(out_shape) == 1:
                    [out_shape] = out_shape

            partial_model = keras.Model(inputs=self.model.inputs, outputs=self.model.layers[layer_index].output)

            # e.g. h_1_0, h_1_1, ...
            # neuron_labels = ['h_' + str(layer_index) + '_' + str(i)
            #                  for i in range(0, self.model.layers[layer_index].output_shape[1])]
            neuron_labels = ['h_' + str(layer_index) + '_' + str(i) for i in range(out_shape[-1])]

            activation_values = pd.DataFrame(data=partial_model.predict(data_x), columns=neuron_labels)
            activation_values.to_csv(self.activations_path + str(layer_index) + '.csv', index=False)

        print('Computed layerwise activations.')

    def get_layer_activations(self, layer_index: int):
        """
        Return activation values given layer index
        """
        filename = self.activations_path + str(layer_index) + '.csv'
        return pd.read_csv(filename)

    def get_layer_activations_of_neuron(self, layer_index: int, neuron_index: int):
        """
        Return activation values given layer index, only return the column for a given neuron index
        """
        filename = self.activations_path + str(layer_index) + '.csv'
        return pd.read_csv(filename)['h_' + str(layer_index) + '_' + str(neuron_index)]

    def set_rules(self, rules):
        self.rules = rules

    def print_rules(self):
        for rule in self.rules:
            print(rule)
