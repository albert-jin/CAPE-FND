import json

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_results(results, file_path):
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)