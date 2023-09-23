import json

JSON_CACHE = dict()


def read_cached_json(file_path):
    if file_path in JSON_CACHE:
        return JSON_CACHE[file_path]
    with open(file_path, 'r') as f:
        data = json.load(f)
        JSON_CACHE[file_path] = data
        return data
