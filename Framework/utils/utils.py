import scipy.io as sio
import json


def load_json_as_dict(path):
    with open(path) as json_file:
        data = json.load(json_file)

    return data

def load_mat_file(path: str):
    with open(path, 'rb') as f:
        mat_data = sio.loadmat(f)
    return mat_data

def save_txt(path: str, data):
    with open(path, 'w') as fp:
        json.dump(data, fp)

def load_txt(path: str):
    with open(path, 'r') as f:
        a = json.loads(f.read())
    return a