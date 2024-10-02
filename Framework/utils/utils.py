import json
import scipy.io as sio


def load_json_as_dict(path):
    with open(path) as json_file:
        data = json.load(json_file)

    return data

def load_mat_file(path: str):
    with open(path, 'rb') as f:
        mat_data = sio.loadmat(f)
    return mat_data