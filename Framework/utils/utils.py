import scipy.io as sio
import json

def load_json_as_dict(path):
    """
    Loads a JSON file and returns its contents as a dictionary.

    Args:
        path (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file as a dictionary.
    """
    with open(path) as json_file:
        data = json.load(json_file)

    return data

def load_mat_file(path: str):
    """
    Loads a MATLAB .mat file and returns its contents.

    Args:
        path (str): The path to the .mat file.

    Returns:
        dict: The contents of the .mat file.
    """
    with open(path, 'rb') as f:
        mat_data = sio.loadmat(f)
    return mat_data

def save_txt(path: str, data):
    """
    Saves data to a text file in JSON format.

    Args:
        path (str): The path to the text file.
        data (dict): The data to save.
    """
    with open(path, 'w') as fp:
        json.dump(data, fp)

def load_txt(path: str):
    """
    Loads data from a text file in JSON format.

    Args:
        path (str): The path to the text file.

    Returns:
        dict: The contents of the text file as a dictionary.
    """
    with open(path, 'r') as f:
        a = json.loads(f.read())
    return a