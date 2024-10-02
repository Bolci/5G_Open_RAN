import json


def load_json_as_dict(path):
    with open(path) as json_file:
        data = json.load(json_file)

    return data