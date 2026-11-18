import joblib


def save_bundle(path, bundle):
    joblib.dump(bundle, path)


def load_bundle(file_or_path):
    return joblib.load(file_or_path)

