"""
Evaluate accuracy of rules
"""
import numpy as np


def accuracy(predicted_labels: np.array, true_labels: np.array):
    assert (len(predicted_labels) == len(true_labels)), "Error: number of labels inconsistent !"

    accuracy = sum(predicted_labels == true_labels) / len(predicted_labels)

    return accuracy
