import scipy.io as sio

def load_file(path: str):
    with open(path, 'rb') as f:
        mat_data = sio.loadmat(f)
    return mat_data


def parse_math(mat_file: dict):
    names = mat_file['outputData'][0].dtype.names
    return {k:v for k,v in zip(names, mat_file['outputData'][0][0])}
    
    