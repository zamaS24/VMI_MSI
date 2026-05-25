import os
import re
from pathlib import Path
import pandas as pd


def extract_label_from_name(filename):
    matches = re.findall(r'\(([^)]*)\)', filename)
    if len(matches) >= 4:
        if matches[3] == '1':
            return 'homme'
        elif matches[3] == '2':
            return 'femme'
    return None


def load_split(base_dir, split_name):
    split_dir = Path(base_dir) / split_name
    data = []
    for root, _, files in os.walk(split_dir):
        for f in files:
            if f.endswith('.txt'):
                label = extract_label_from_name(f)
                if label:
                    data.append({
                        'path': os.path.join(root, f),
                        'label': label
                    })
    return pd.DataFrame(data)


def read_text(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def load_datasets(base_dir):
    train_df = load_split(base_dir, 'train')
    val_df = load_split(base_dir, 'val')
    test_df = load_split(base_dir, 'test')

    for df in [train_df, val_df, test_df]:
        df['text'] = df['path'].apply(read_text)

    return train_df, val_df, test_df
