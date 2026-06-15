import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


RANDOM_SEED = 42
TFIDF_PARAMS = {
    'max_features': 25000,
    'min_df': 2,
    'max_df': 0.85,
    'ngram_range': (1, 1),
}


def set_random_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_labels(train_df, val_df, test_df):
    le = LabelEncoder()
    train_df['label_encoded'] = le.fit_transform(train_df['label'])
    val_df['label_encoded'] = le.transform(val_df['label'])
    test_df['label_encoded'] = le.transform(test_df['label'])

    class_mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print(f"Classes encod\u00e9es: {class_mapping}")

    return le


def build_tfidf_features(train_df, val_df, test_df, tfidf_params=None):
    params = TFIDF_PARAMS if tfidf_params is None else tfidf_params
    tfidf = TfidfVectorizer(**params)

    X_train = tfidf.fit_transform(train_df['text']).toarray().astype(np.float32)
    X_val = tfidf.transform(val_df['text']).toarray().astype(np.float32)
    X_test = tfidf.transform(test_df['text']).toarray().astype(np.float32)

    y_train = train_df['label_encoded'].values
    y_val = val_df['label_encoded'].values
    y_test = test_df['label_encoded'].values

    n_features = len(tfidf.get_feature_names_out())
    max_features = params.get('max_features')
    max_features_text = 'no limit' if max_features is None else str(max_features)

    print(f"Parametres TF-IDF: max_features={max_features_text}, min_df={params.get('min_df')}, max_df={params.get('max_df')}")
    print(f"Forme X_train: {X_train.shape}")
    print(f"Nombre de features TF-IDF retenues: {n_features}")

    return tfidf, X_train, X_val, X_test, y_train, y_val, y_test


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_pickle(obj, path):
    ensure_parent_dir(path)
    try:
        import joblib

        joblib.dump(obj, path)
    except Exception:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def save_json(obj, path):
    ensure_parent_dir(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)
