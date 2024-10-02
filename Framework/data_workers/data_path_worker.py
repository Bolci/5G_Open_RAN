import os


def create_paths_for_data_type(path: str):
    folders = os.listdir(path)
    return [os.path.join(path, x) for x in folders]


def get_all_paths(path: dict) -> dict:
    general_path = path['Data_path']
    train_path = os.path.join(general_path, path['Train_folders'])
    valid_path = os.path.join(general_path, path['Valid_folders'])
    test_path = os.path.join(general_path, path['Test_folders'])

    train_folders_path = create_paths_for_data_type(train_path)
    valid_folders_path = create_paths_for_data_type(valid_path)
    test_folders_path = create_paths_for_data_type(test_path)

    return {'Train': train_folders_path,
            'Valid': valid_folders_path,
            'Test': test_folders_path}