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
    general_path = path['Data_path']

    train_folders_path = []
    valid_folders_path = []
    test_folders_path = []

    if 'Train_folders' in path.keys():
        train_path = os.path.join(general_path, path['Train_folders'])
        train_folders_path = create_paths_for_data_type(train_path)

    if 'Valid_folders' in path.keys():
        valid_path = os.path.join(general_path, path['Valid_folders'])
        valid_folders_path = create_paths_for_data_type(valid_path)

    if 'Test_folders' in path.keys():
        test_path = os.path.join(general_path, path['Test_folders'])
        test_folders_path = create_paths_for_data_type(test_path)

    datasets = {'Train': train_folders_path, 'Valid': valid_folders_path, 'Test': test_folders_path}

    return datasets
