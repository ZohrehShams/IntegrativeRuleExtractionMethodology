"""
Initialise the empty dataset directory

file structure

<dataset-name>
    data.csv
    cross_validation/
        <n>_folds/
            rule_extraction/
                <rule ex mode>/
                    results.csv
                    rules_extracted/
                        fold_<n>.rules
            trained_models/
                fold_<n>_model.h5
            data_split_indices.txt
    neural_network_initialisation/
        re_results.csv
        grid_search_results.txt
        data_split_indices.txt
        best_initialisation.h5
"""
import os


def create_directory(dir_path):
    """

    Args:
        dir_path: path to the new directory
    """
    # Create directory given path if it doesnt exist
    try:
        os.makedirs(dir_path)
        print("Directory ", dir_path, " Created ")
    except FileExistsError:
        print("Directory ", dir_path, " already exists")


def run(dataset_name, path_to_data_folder):
    """

    Args:
        dataset_name: e.g. 'MB-GE-ER' or 'MNIST'
        path_to_data_folder: path to main data/ folder for project

    Creates empty dataset directory as specified above
    """
    # Base directory
    base_path = path_to_data_folder + dataset_name + '/'
    create_directory(base_path)

    # <dataset_name>/neural_network_initialisation/
    create_directory(dir_path=base_path + 'neural_network_initialisation')

    # <dataset_name>/cross_validation/
    create_directory(dir_path=base_path + 'cross_validation')


def clean_up():
    from src import TEMP_DIR
    def clear_dir(dir_path):
        for file in os.listdir(dir_path):
            if os.path.isdir(dir_path + file):
                clear_dir(dir_path + file + '/')
            else:
                os.remove(dir_path + file)

    # Remove temporary files at the end
    print('Cleaning up temporary files...', end='', flush=True)
    clear_dir(TEMP_DIR)
    print('done')


def clear_file(file_path):
    """

    Args:
        file_path:

    Clear contents of a file given file path

    """
    if os.path.exists(file_path):
        open(file_path, 'w').close()
        print('Cleared contents of file %s' % file_path)