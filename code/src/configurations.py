# Configurations for datasets
# TODO do this using proper configs later
from collections import namedtuple

from rules.rule import OutputClass
DatasetMetaData = namedtuple('DatasetMetaData', 'name target_col output_classes n_inputs n_outputs')


def get_configuration(dataset_name):
    """
    Return target_col_name and output class encodings for dataset accordingly
    i.e. Class names with their corresponding encoding (output neuron index)
    """
    target_col_name = output_classes = dataset_info = None

    if dataset_name == 'MB-GE-ER':
        output_classes = (OutputClass(name='-', encoding=0),
                          OutputClass(name='+', encoding=1))
        dataset_info = DatasetMetaData(name=dataset_name, target_col='ER_Expr', output_classes=output_classes,
                                       n_inputs=1000, n_outputs=2)
        
    elif dataset_name == 'MB-ClinP-ER':
        output_classes = (OutputClass(name='-', encoding=0),
                          OutputClass(name='+', encoding=1))
        dataset_info = DatasetMetaData(name=dataset_name, target_col='ER_Expr', output_classes=output_classes,
                                       n_inputs=13, n_outputs=2)


    elif dataset_name == 'MB-GE-ClinP-ER':
        output_classes = (OutputClass(name='-', encoding=0),
                          OutputClass(name='+', encoding=1))
        dataset_info = DatasetMetaData(name=dataset_name, target_col='ER_Expr', output_classes=output_classes,
                                       n_inputs=1013, n_outputs=2)

    elif dataset_name == 'MB-1004-GE-2Hist':
        output_classes = (OutputClass(name='IDC', encoding=0),
                          OutputClass(name='ILC', encoding=1))
        dataset_info = DatasetMetaData(name=dataset_name, target_col='Histological_Type', output_classes=output_classes,
                                       n_inputs=1004, n_outputs=2)

    else:
        print('WARNING: invalid dataset name given!')

    return dataset_info


