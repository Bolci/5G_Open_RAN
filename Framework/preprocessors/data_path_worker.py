import os


def create_paths_for_data_type(path: str):
    """
    Create a list of full paths for each folder in the given directory.

    Args:
        path (str): The directory path to list folders from.

    Returns:
        list: A list of full paths to each folder in the given directory.
    """
    folders = os.listdir(path)
    return [os.path.join(path, x) for x in folders]

def get_all_paths(path: dict) -> dict:
    """
    Generate paths for training, validation, and test datasets based on the provided directory structure.

    Args:
        path (dict): A dictionary containing the general data path and specific folder names for training, validation, and test datasets.

    Returns:
        dict: A dictionary with keys 'Train', 'Valid', and 'Test', each containing a list of full paths to the respective dataset folders.
    """

def prepare_single_path_per_data_type(folders_names, general_path, data_type, keep_folders = False):

    folder_paths_final = {}

    if not isinstance(folders_names, list):
        folders_names = [folders_names]

    folder_paths_mixed = []

    for single_folder_path in folders_names:
        single_folder_path_full = os.path.join(general_path, single_folder_path)
        test_folders_path = create_paths_for_data_type(single_folder_path_full)

        if keep_folders:
            folder_paths_final[f'{data_type}_{single_folder_path}'] = test_folders_path
        else:
            folder_paths_mixed += test_folders_path

    if not keep_folders:
        folder_paths_final[data_type] = folder_paths_mixed

    return folder_paths_final


def get_all_paths(path: dict,
                  keep_test_scenarios: bool = True) -> dict:

    datasets = {}

    if 'Train_folders' in path.keys():
        train_folders_path = prepare_single_path_per_data_type(folders_names = path['Train_folders'],
                                                               general_path = path['Data_path'],
                                                               data_type = 'Train',
                                                               )
        datasets |= train_folders_path


    if 'Valid_folders' in path.keys():
        valid_folders_path = prepare_single_path_per_data_type(folders_names=path['Valid_folders'],
                                                               general_path=path['Data_path'],
                                                               data_type='Valid',
                                                               )
        datasets |= valid_folders_path

    else:
        datasets |= {'Valid': []}



    if 'Test_folders' in path.keys():
        test_folders_paths = prepare_single_path_per_data_type(folders_names=path['Test_folders'],
                                                               general_path=path['Data_path'],
                                                               data_type='Test',
                                                               keep_folders = keep_test_scenarios)
        datasets |= test_folders_paths

    return datasets
