import argparse
import json
import pickle
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path('data') / 'datasetSujet3' / 'content' / 'dataset'
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / 'outputs' / 'tfidf_mlp_model.pt'
DEFAULT_VECTORIZER_PATH = Path(__file__).resolve().parent / 'outputs' / 'tfidf_vectorizer.pkl'
DEFAULT_LOCAL_OUTPUT = PROJECT_ROOT / 'vis' / 'tfidf_lrp_local.csv'
DEFAULT_GLOBAL_OUTPUT = PROJECT_ROOT / 'vis' / 'tfidf_lrp_global.csv'
DEFAULT_HOMME_PLOT = PROJECT_ROOT / 'vis' / 'tfidf_lrp_top_homme_terms.png'
DEFAULT_FEMME_PLOT = PROJECT_ROOT / 'vis' / 'tfidf_lrp_top_femme_terms.png'


def load_runtime_dependencies():
    global np, pd, torch, nn, plt
    global load_datasets, MLPNet

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn

    try:
        from .data_loader import load_datasets
        from .model import MLPNet
    except ImportError:
        from data_loader import load_datasets
        from model import MLPNet


def parse_args():
    parser = argparse.ArgumentParser(
        description='Explain the fixed TF-IDF + MLP classifier with LRP epsilon rule.'
    )
    parser.add_argument(
        '--data_dir',
        '--data-dir',
        dest='data_dir',
        type=Path,
        default=DEFAULT_DATA_DIR,
        help='Dataset directory containing train, val, and test folders.',
    )
    parser.add_argument(
        '--model_path',
        '--model-path',
        dest='model_path',
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help='Path to the saved TF-IDF MLP checkpoint.',
    )
    parser.add_argument(
        '--vectorizer_path',
        '--vectorizer-path',
        dest='vectorizer_path',
        type=Path,
        default=DEFAULT_VECTORIZER_PATH,
        help='Path to the saved fitted TF-IDF vectorizer pickle.',
    )
    parser.add_argument(
        '--n_examples',
        '--n-examples',
        dest='n_examples',
        type=int,
        default=50,
        help='Number of test examples to explain locally.',
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=1e-6,
        help='LRP epsilon stabilizer.',
    )
    parser.add_argument(
        '--explain_class',
        '--explain-class',
        dest='explain_class',
        choices=('predicted', 'true'),
        default='predicted',
        help='Class to explain for local examples.',
    )
    parser.add_argument(
        '--top_k',
        '--top-k',
        dest='top_k',
        type=int,
        default=20,
        help='Number of positive/negative terms to keep in outputs and plots.',
    )
    parser.add_argument(
        '--device',
        default='auto',
        help='Torch device. Use auto, cpu, cuda, or cuda:0.',
    )
    parser.add_argument(
        '--local_output',
        '--local-output',
        dest='local_output',
        type=Path,
        default=DEFAULT_LOCAL_OUTPUT,
        help='Path for local LRP CSV output.',
    )
    parser.add_argument(
        '--global_output',
        '--global-output',
        dest='global_output',
        type=Path,
        default=DEFAULT_GLOBAL_OUTPUT,
        help='Path for global LRP CSV output.',
    )
    parser.add_argument(
        '--homme_plot',
        '--homme-plot',
        dest='homme_plot',
        type=Path,
        default=DEFAULT_HOMME_PLOT,
        help='Path for the top homme terms plot.',
    )
    parser.add_argument(
        '--femme_plot',
        '--femme-plot',
        dest='femme_plot',
        type=Path,
        default=DEFAULT_FEMME_PLOT,
        help='Path for the top femme terms plot.',
    )
    return parser.parse_args()


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def resolve_device(device_name):
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)


def torch_load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_vectorizer(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_model(model_path, input_dim, device):
    checkpoint = torch_load_checkpoint(model_path, device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        output_dim = checkpoint.get('output_dim', 2)
        dropout_input = checkpoint.get('dropout_input', 0.35)
        dropout_hidden = checkpoint.get('dropout_hidden', 0.45)
        classes = checkpoint.get('classes', ['femme', 'homme'])
    else:
        state_dict = checkpoint
        output_dim = 2
        dropout_input = 0.35
        dropout_hidden = 0.45
        classes = ['femme', 'homme']

    model = MLPNet(
        input_dim=input_dim,
        output_dim=output_dim,
        dropout_input=dropout_input,
        dropout_hidden=dropout_hidden,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, list(classes)


def predict(model, X, device, batch_size=64):
    preds = []
    confidences = []

    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = torch.from_numpy(X[start:start + batch_size]).to(device)
            logits = model(batch)
            probabilities = torch.softmax(logits, dim=1)
            batch_confidences, batch_preds = probabilities.max(dim=1)
            preds.extend(batch_preds.cpu().numpy())
            confidences.extend(batch_confidences.cpu().numpy())

    return np.asarray(preds), np.asarray(confidences)


def forward_with_activations(model, input_tensor):
    activations = [input_tensor]
    output = input_tensor

    for layer in model.network:
        output = layer(output)
        activations.append(output)

    return output, activations


def stabilized_denominator(z, epsilon):
    signs = torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    return z + epsilon * signs


def lrp_linear(layer, activation, relevance, epsilon):
    weight = layer.weight
    z = activation.matmul(weight.t())
    z_stable = stabilized_denominator(z, epsilon)
    redistribution = relevance / z_stable
    return activation * redistribution.matmul(weight)


def lrp_epsilon(model, input_vector, target_class, device, epsilon=1e-6):
    input_tensor = torch.from_numpy(input_vector.astype(np.float32)).to(device).unsqueeze(0)

    with torch.no_grad():
        logits, activations = forward_with_activations(model, input_tensor)

        relevance = torch.zeros_like(logits)
        relevance[:, target_class] = logits[:, target_class]

        for layer_index in range(len(model.network) - 1, -1, -1):
            layer = model.network[layer_index]

            if isinstance(layer, nn.Linear):
                relevance = lrp_linear(
                    layer,
                    activations[layer_index],
                    relevance,
                    epsilon,
                )
            elif isinstance(layer, nn.ReLU):
                relevance = relevance
            else:
                raise TypeError(f'Unsupported layer for LRP: {type(layer).__name__}')

    return relevance.squeeze(0).cpu().numpy()


def relevance_records(input_vector, relevance, vocabulary):
    nonzero_indices = np.flatnonzero(input_vector)
    records = []

    for feature_index in nonzero_indices:
        records.append({
            'feature_index': int(feature_index),
            'term': str(vocabulary[feature_index]),
            'tfidf': float(input_vector[feature_index]),
            'relevance': float(relevance[feature_index]),
        })

    return records


def top_positive_negative(records, top_k):
    positive = [record for record in records if record['relevance'] > 0]
    negative = [record for record in records if record['relevance'] < 0]

    positive.sort(key=lambda record: record['relevance'], reverse=True)
    negative.sort(key=lambda record: record['relevance'])

    return positive[:top_k], negative[:top_k]


def dumps_records(records):
    return json.dumps(records, ensure_ascii=False)


def build_local_explanations(
    model,
    test_df,
    X_test,
    y_true,
    preds,
    confidences,
    classes,
    vocabulary,
    device,
    n_examples,
    epsilon,
    explain_class,
    top_k,
):
    columns = [
        'file_path',
        'true_label',
        'predicted_label',
        'confidence',
        'explained_class',
        'top_positive_terms',
        'top_negative_terms',
        'raw_relevance_scores',
    ]
    rows = []
    n_selected = min(n_examples, X_test.shape[0])

    for row_index in range(n_selected):
        target_class = preds[row_index]
        if explain_class == 'true':
            target_class = y_true[row_index]

        relevance = lrp_epsilon(
            model,
            X_test[row_index],
            int(target_class),
            device,
            epsilon=epsilon,
        )
        records = relevance_records(X_test[row_index], relevance, vocabulary)
        positive, negative = top_positive_negative(records, top_k)

        rows.append({
            'file_path': test_df.iloc[row_index]['path'],
            'true_label': classes[int(y_true[row_index])],
            'predicted_label': classes[int(preds[row_index])],
            'confidence': float(confidences[row_index]),
            'explained_class': classes[int(target_class)],
            'top_positive_terms': dumps_records(positive),
            'top_negative_terms': dumps_records(negative),
            'raw_relevance_scores': dumps_records(records),
        })

    return pd.DataFrame(rows, columns=columns)


def aggregate_global_explanations(
    model,
    X_test,
    y_true,
    preds,
    classes,
    vocabulary,
    device,
    epsilon,
    top_k,
):
    columns = [
        'class_label',
        'rank',
        'feature_index',
        'term',
        'aggregate_relevance',
        'mean_relevance',
        'n_correct_texts',
    ]
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}
    aggregate_rows = []

    for class_name in ('homme', 'femme'):
        if class_name not in class_to_index:
            continue

        class_index = class_to_index[class_name]
        correct_indices = np.where((y_true == class_index) & (preds == class_index))[0]
        relevance_sum = np.zeros(len(vocabulary), dtype=np.float64)

        for row_index in correct_indices:
            relevance = lrp_epsilon(
                model,
                X_test[row_index],
                class_index,
                device,
                epsilon=epsilon,
            )
            relevance_sum += relevance

        if len(correct_indices) == 0:
            continue

        mean_relevance = relevance_sum / len(correct_indices)
        positive_indices = np.flatnonzero(mean_relevance > 0)
        sorted_indices = positive_indices[
            np.argsort(mean_relevance[positive_indices])[::-1]
        ][:top_k]

        for rank, feature_index in enumerate(sorted_indices, start=1):
            aggregate_rows.append({
                'class_label': class_name,
                'rank': rank,
                'feature_index': int(feature_index),
                'term': str(vocabulary[feature_index]),
                'aggregate_relevance': float(relevance_sum[feature_index]),
                'mean_relevance': float(mean_relevance[feature_index]),
                'n_correct_texts': int(len(correct_indices)),
            })

    return pd.DataFrame(aggregate_rows, columns=columns)


def save_top_terms_plot(global_df, class_label, output_path, top_k):
    subset = global_df[global_df['class_label'] == class_label].head(top_k)

    plt.figure(figsize=(10, 6))
    if subset.empty:
        plt.title(f'No correctly classified {class_label} examples')
        plt.axis('off')
    else:
        terms = subset['term'].iloc[::-1]
        scores = subset['mean_relevance'].iloc[::-1]
        plt.barh(terms, scores)
        plt.xlabel('Mean LRP relevance')
        plt.title(f'Top TF-IDF terms supporting {class_label}')
        plt.tight_layout()

    ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_outputs(local_df, global_df, args):
    ensure_parent_dir(args.local_output)
    local_df.to_csv(args.local_output, index=False, encoding='utf-8')

    ensure_parent_dir(args.global_output)
    global_df.to_csv(args.global_output, index=False, encoding='utf-8')

    save_top_terms_plot(global_df, 'homme', args.homme_plot, args.top_k)
    save_top_terms_plot(global_df, 'femme', args.femme_plot, args.top_k)


def main():
    args = parse_args()
    load_runtime_dependencies()

    device = resolve_device(args.device)
    vectorizer = load_vectorizer(args.vectorizer_path)
    vocabulary = vectorizer.get_feature_names_out()
    model, classes = load_model(args.model_path, len(vocabulary), device)
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}

    _, _, test_df = load_datasets(args.data_dir)
    missing_labels = sorted(set(test_df['label']) - set(class_to_index))
    if missing_labels:
        raise ValueError(f'Labels not present in checkpoint classes: {missing_labels}')

    X_test = vectorizer.transform(test_df['text']).toarray().astype(np.float32)
    y_true = test_df['label'].map(class_to_index).to_numpy(dtype=np.int64)
    preds, confidences = predict(model, X_test, device)

    local_df = build_local_explanations(
        model,
        test_df,
        X_test,
        y_true,
        preds,
        confidences,
        classes,
        vocabulary,
        device,
        args.n_examples,
        args.epsilon,
        args.explain_class,
        args.top_k,
    )
    global_df = aggregate_global_explanations(
        model,
        X_test,
        y_true,
        preds,
        classes,
        vocabulary,
        device,
        args.epsilon,
        args.top_k,
    )

    save_outputs(local_df, global_df, args)

    print(f'Saved local explanations: {args.local_output}')
    print(f'Saved global explanations: {args.global_output}')
    print(f'Saved homme plot: {args.homme_plot}')
    print(f'Saved femme plot: {args.femme_plot}')


if __name__ == '__main__':
    main()
