"""
Evaluate fidelity of rules generated i.e. how well do they mimic the performance of the Neural Network
"""

import pandas as pd

def fidelity(predicted_labels, network_labels):
    assert (len(predicted_labels) == len(network_labels)), "Error: number of labels inconsistent !"

    fid = sum(predicted_labels == network_labels) / len(predicted_labels)

    return fid
