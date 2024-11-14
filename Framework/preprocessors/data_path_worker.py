import os


def create_paths_for_data_type(path: str):
    folders = os.listdir(path)
    return [os.path.join(path, x) for x in folders]


def prepare_single_path_per_data_type(data_type, path):
    general_path = path['Data_path']
    folder_paths_final = []

    folders = path[data_type]
    if not isinstance(folders, list):
        folders = [folders]

    for single_folder_path in folders:
        single_folder_path_full = os.path.join(general_path, single_folder_path)
        test_folders_path = create_paths_for_data_type(single_folder_path_full)

        folder_paths_final += test_folders_path

    return folder_paths_final


def get_all_paths(path: dict) -> dict:
    train_folders_path = []
    valid_folders_path = []
    test_folders_paths = []

    if 'Train_folders' in path.keys():
        train_folders_path = prepare_single_path_per_data_type('Train_folders', path)

    if 'Valid_folders' in path.keys():
        valid_folders_path = prepare_single_path_per_data_type('Valid_folders', path)

    if 'Test_folders' in path.keys():
        test_folders_paths = prepare_single_path_per_data_type('Test_folders', path)

    datasets = {'Train': train_folders_path, 'Valid': valid_folders_path, 'Test': test_folders_paths}

    return datasets
