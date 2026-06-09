import argparse
import json
import pickle
import re
from pathlib import Path


TFIDF_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path('data') / 'datasetSujet3' / 'content' / 'dataset'
DEFAULT_MODEL_PATH = TFIDF_ROOT / 'outputs' / 'tfidf_mlp_model.pt'
DEFAULT_VECTORIZER_PATH = TFIDF_ROOT / 'outputs' / 'tfidf_vectorizer.pkl'
DEFAULT_VIS_DIR = TFIDF_ROOT / 'vis'
DEFAULT_LRP_GLOBAL_PATH = DEFAULT_VIS_DIR / 'tfidf_lrp_global.csv'
DEFAULT_IG_GLOBAL_PATH = DEFAULT_VIS_DIR / 'tfidf_integrated_gradients_global.csv'

METHOD_LABELS = {
    'ig': 'Integrated Gradients',
    'lrp': 'LRP',
    'intersection': 'IG and LRP intersection',
}


def load_runtime_dependencies():
    global np, pd, torch, plt
    global load_datasets, MLPNet

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch

    try:
        from .data_loader import load_datasets
        from .model import MLPNet
    except ImportError:
        from data_loader import load_datasets
        from model import MLPNet


def parse_args():
    parser = argparse.ArgumentParser(
        description='Perturb top explanation terms for the fixed TF-IDF + MLP classifier.'
    )
    parser.add_argument(
        '--method',
        choices=('lrp', 'ig', 'intersection'),
        default='intersection',
        help='Explanation terms to use for perturbation.',
    )
    parser.add_argument(
        '--mode',
        choices=('remove', 'mask', 'swap'),
        default='remove',
        help='Perturbation mode.',
    )
    parser.add_argument(
        '--n_texts_per_class',
        '--n-texts-per-class',
        '--n_examples',
        '--n-examples',
        dest='n_texts_per_class',
        type=int,
        default=50,
        help='Number of correctly classified texts to select per class.',
    )
    parser.add_argument(
        '--n_terms',
        '--n-terms',
        dest='n_terms',
        type=int,
        default=20,
        help='Number of explanation terms to perturb per class.',
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
        '--lrp_global_path',
        '--lrp-global-path',
        dest='lrp_global_path',
        type=Path,
        default=DEFAULT_LRP_GLOBAL_PATH,
        help='Path to global LRP explanation terms.',
    )
    parser.add_argument(
        '--ig_global_path',
        '--ig-global-path',
        dest='ig_global_path',
        type=Path,
        default=DEFAULT_IG_GLOBAL_PATH,
        help='Path to global Integrated Gradients explanation terms.',
    )
    parser.add_argument(
        '--output_dir',
        '--output-dir',
        dest='output_dir',
        type=Path,
        default=DEFAULT_VIS_DIR,
        help='Directory for dynamically named perturbation outputs.',
    )
    parser.add_argument(
        '--output_csv',
        '--output-csv',
        dest='output_csv',
        type=Path,
        default=None,
        help='Optional explicit path for perturbation row-level CSV output.',
    )
    parser.add_argument(
        '--summary_json',
        '--summary-json',
        dest='summary_json',
        type=Path,
        default=None,
        help='Optional explicit path for perturbation summary JSON output.',
    )
    parser.add_argument(
        '--accuracy_plot',
        '--accuracy-plot',
        dest='accuracy_plot',
        type=Path,
        default=None,
        help='Optional explicit path for before/after accuracy plot.',
    )
    parser.add_argument(
        '--confidence_plot',
        '--confidence-plot',
        dest='confidence_plot',
        type=Path,
        default=None,
        help='Optional explicit path for confidence drop plot.',
    )
    parser.add_argument(
        '--device',
        default='auto',
        help='Torch device. Use auto, cpu, cuda, or cuda:0.',
    )
    return parser.parse_args()


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def make_run_slug(method, mode, n_texts, n_texts_per_class, n_terms):
    return (
        f'method-{method}_mode-{mode}_docs-{n_texts}'
        f'_per-class-{n_texts_per_class}_terms-{n_terms}'
    )


def resolve_output_paths(args, n_texts):
    slug = make_run_slug(
        args.method,
        args.mode,
        n_texts,
        args.n_texts_per_class,
        args.n_terms,
    )
    output_dir = args.output_dir

    return {
        'output_csv': args.output_csv or output_dir / f'tfidf_perturbation_rows_{slug}.csv',
        'summary_json': args.summary_json or output_dir / f'tfidf_perturbation_summary_{slug}.json',
        'accuracy_plot': args.accuracy_plot or output_dir / f'tfidf_perturbation_accuracy_drop_{slug}.png',
        'confidence_plot': args.confidence_plot or output_dir / f'tfidf_perturbation_confidence_drop_{slug}.png',
    }


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
        hidden_layers = tuple(checkpoint.get('hidden_layers', (64, 32)))
        output_dim = checkpoint.get('output_dim', 2)
        classes = checkpoint.get('classes', ['femme', 'homme'])
    else:
        state_dict = checkpoint
        hidden_layers = (64, 32)
        output_dim = 2
        classes = ['femme', 'homme']

    model = MLPNet(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        output_dim=output_dim,
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


def load_global_terms(path, n_terms):
    df = pd.read_csv(path)
    required_columns = {'class_label', 'term'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f'{path} is missing required columns: {sorted(missing_columns)}')

    terms_by_class = {}
    for class_label in ('homme', 'femme'):
        class_df = df[df['class_label'] == class_label].copy()
        if 'rank' in class_df.columns:
            class_df = class_df.sort_values('rank')
        terms_by_class[class_label] = [
            str(term)
            for term in class_df['term'].dropna().head(n_terms).tolist()
        ]

    return terms_by_class


def intersect_terms(primary_terms, secondary_terms):
    output = {}

    for class_label in ('homme', 'femme'):
        secondary_set = set(secondary_terms[class_label])
        output[class_label] = [
            term for term in primary_terms[class_label] if term in secondary_set
        ]

    return output


def get_terms_by_method(method, lrp_path, ig_path, n_terms):
    if method == 'lrp':
        return load_global_terms(lrp_path, n_terms)
    if method == 'ig':
        return load_global_terms(ig_path, n_terms)

    lrp_terms = load_global_terms(lrp_path, n_terms)
    ig_terms = load_global_terms(ig_path, n_terms)
    return intersect_terms(lrp_terms, ig_terms)


def compile_term_patterns(terms):
    patterns = []

    for term in terms:
        escaped = re.escape(term)
        patterns.append((term, re.compile(rf'(?<!\w){escaped}(?!\w)', re.IGNORECASE)))

    return patterns


def normalize_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()


def perturb_text(text, source_terms, mode, replacement_terms=None):
    changed_terms = []
    perturbed_text = text
    replacement_terms = replacement_terms or []

    for index, (term, pattern) in enumerate(compile_term_patterns(source_terms)):
        if not pattern.search(perturbed_text):
            continue

        changed_terms.append(term)

        if mode == 'remove':
            perturbed_text = pattern.sub(' ', perturbed_text)
        elif mode == 'mask':
            perturbed_text = pattern.sub('[MASK]', perturbed_text)
        elif mode == 'swap':
            replacement = replacement_terms[index % len(replacement_terms)]
            perturbed_text = pattern.sub(replacement, perturbed_text)
        else:
            raise ValueError(f'Unsupported perturbation mode: {mode}')

    return normalize_spaces(perturbed_text), changed_terms


def select_correct_examples(test_df, y_true, preds, classes, n_texts_per_class):
    selected_indices = []
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}

    for class_label in ('homme', 'femme'):
        class_index = class_to_index[class_label]
        indices = np.where((y_true == class_index) & (preds == class_index))[0]
        selected_indices.extend(indices[:n_texts_per_class].tolist())

    selected_df = test_df.iloc[selected_indices].copy()
    selected_df['row_index'] = selected_indices

    return selected_df


def build_perturbation_rows(
    selected_df,
    y_true,
    original_preds,
    original_confidences,
    perturbed_preds,
    perturbed_confidences,
    classes,
    terms_by_class,
    perturbed_texts,
    changed_terms_by_row,
    mode,
    method,
):
    rows = []

    for output_index, (_, row) in enumerate(selected_df.iterrows()):
        source_index = int(row['row_index'])
        true_label = row['label']
        original_prediction = classes[int(original_preds[source_index])]
        perturbed_prediction = classes[int(perturbed_preds[output_index])]

        rows.append({
            'file_path': row['path'],
            'true_label': true_label,
            'original_prediction': original_prediction,
            'original_confidence': float(original_confidences[source_index]),
            'perturbed_prediction': perturbed_prediction,
            'perturbed_confidence': float(perturbed_confidences[output_index]),
            'prediction_changed': bool(original_prediction != perturbed_prediction),
            'correct_before': bool(original_prediction == true_label),
            'correct_after': bool(perturbed_prediction == true_label),
            'removed_or_replaced_terms': json.dumps(
                changed_terms_by_row[output_index],
                ensure_ascii=False,
            ),
            'perturbation_mode': mode,
            'explanation_method': method,
        })

    return pd.DataFrame(rows)


def make_perturbed_texts(selected_df, terms_by_class, mode):
    perturbed_texts = []
    changed_terms_by_row = []

    for _, row in selected_df.iterrows():
        true_label = row['label']
        other_label = 'femme' if true_label == 'homme' else 'homme'
        source_terms = terms_by_class[true_label]
        replacement_terms = terms_by_class[other_label]

        if mode == 'swap' and not replacement_terms:
            raise ValueError(f'No replacement terms available for class {other_label}')

        perturbed_text, changed_terms = perturb_text(
            row['text'],
            source_terms,
            mode,
            replacement_terms=replacement_terms,
        )
        perturbed_texts.append(perturbed_text)
        changed_terms_by_row.append(changed_terms)

    return perturbed_texts, changed_terms_by_row


def safe_mean(values):
    if len(values) == 0:
        return None
    return float(np.mean(values))


def class_accuracy(rows_df, class_label, column):
    subset = rows_df[rows_df['true_label'] == class_label]
    if subset.empty:
        return None
    return float(subset[column].mean())


def build_summary(rows_df, method, mode, n_terms, n_texts_per_class, terms_by_class):
    accuracy_before = float(rows_df['correct_before'].mean())
    accuracy_after = float(rows_df['correct_after'].mean())
    confidence_drop_values = (
        rows_df['original_confidence'] - rows_df['perturbed_confidence']
    )

    return {
        'explanation_method': method,
        'perturbation_mode': mode,
        'n_terms_requested': n_terms,
        'n_terms_used': {
            'homme': len(terms_by_class['homme']),
            'femme': len(terms_by_class['femme']),
        },
        'n_texts_per_class_requested': int(n_texts_per_class),
        'selected_texts_by_class': {
            str(class_label): int(count)
            for class_label, count in rows_df['true_label'].value_counts().items()
        },
        'n_texts': int(len(rows_df)),
        'accuracy_before_perturbation': accuracy_before,
        'accuracy_after_perturbation': accuracy_after,
        'accuracy_drop': accuracy_before - accuracy_after,
        'number_of_flipped_predictions': int(rows_df['prediction_changed'].sum()),
        'flip_rate': float(rows_df['prediction_changed'].mean()),
        'confidence_drop': float(confidence_drop_values.mean()),
        'mean_original_confidence': float(rows_df['original_confidence'].mean()),
        'mean_perturbed_confidence': float(rows_df['perturbed_confidence'].mean()),
        'homme_accuracy_before': class_accuracy(rows_df, 'homme', 'correct_before'),
        'homme_accuracy_after': class_accuracy(rows_df, 'homme', 'correct_after'),
        'femme_accuracy_before': class_accuracy(rows_df, 'femme', 'correct_before'),
        'femme_accuracy_after': class_accuracy(rows_df, 'femme', 'correct_after'),
        'interpretation_note': (
            'These perturbation results describe what the classifier learned, '
            'not universal male/female writing rules.'
        ),
    }


def save_json(obj, path):
    ensure_parent_dir(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_plot_title(summary, metric_name):
    method_label = METHOD_LABELS.get(
        summary['explanation_method'],
        summary['explanation_method'],
    )
    n_terms_used = summary['n_terms_used']
    return (
        f'TF-IDF perturbation {metric_name}\n'
        f'method={method_label} | mode={summary["perturbation_mode"]}\n'
        f'docs={summary["n_texts"]} | requested/class={summary["n_texts_per_class_requested"]} | '
        f'terms={summary["n_terms_requested"]} | '
        f'used terms: homme={n_terms_used["homme"]}, femme={n_terms_used["femme"]}'
    )


def save_accuracy_plot(summary, output_path):
    labels = ['Before', 'After']
    values = [
        summary['accuracy_before_perturbation'],
        summary['accuracy_after_perturbation'],
    ]

    plt.figure(figsize=(9, 6))
    plt.bar(labels, values, color=['#4C78A8', '#F58518'])
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy')
    plt.title(build_plot_title(summary, 'accuracy drop'), fontsize=10, pad=12)
    for index, value in enumerate(values):
        plt.text(index, value + 0.025, f'{value:.3f}', ha='center')
    plt.figtext(
        0.5,
        0.01,
        (
            f'Accuracy drop: {summary["accuracy_drop"]:.3f} | '
            f'Flipped predictions: {summary["number_of_flipped_predictions"]}'
        ),
        ha='center',
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.07, 1, 0.96))
    ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_confidence_plot(summary, output_path):
    labels = ['Before', 'After']
    values = [
        summary['mean_original_confidence'],
        summary['mean_perturbed_confidence'],
    ]

    plt.figure(figsize=(9, 6))
    plt.bar(labels, values, color=['#54A24B', '#E45756'])
    plt.ylim(0, 1.1)
    plt.ylabel('Mean predicted-class confidence')
    plt.title(build_plot_title(summary, 'confidence drop'), fontsize=10, pad=12)
    for index, value in enumerate(values):
        plt.text(index, value + 0.025, f'{value:.3f}', ha='center')
    plt.figtext(
        0.5,
        0.01,
        f'Mean confidence drop: {summary["confidence_drop"]:.3f}',
        ha='center',
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.07, 1, 0.96))
    ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    load_runtime_dependencies()

    terms_by_class = get_terms_by_method(
        args.method,
        args.lrp_global_path,
        args.ig_global_path,
        args.n_terms,
    )
    if not terms_by_class['homme'] or not terms_by_class['femme']:
        raise ValueError(
            'No usable explanation terms found for one or both classes. '
            'Check the global explanation CSVs and selected method.'
        )

    device = resolve_device(args.device)
    vectorizer = load_vectorizer(args.vectorizer_path)
    vocabulary_size = len(vectorizer.get_feature_names_out())
    model, classes = load_model(args.model_path, vocabulary_size, device)
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}

    _, _, test_df = load_datasets(args.data_dir)
    missing_labels = sorted(set(test_df['label']) - set(class_to_index))
    if missing_labels:
        raise ValueError(f'Labels not present in checkpoint classes: {missing_labels}')

    X_test = vectorizer.transform(test_df['text']).toarray().astype(np.float32)
    y_true = test_df['label'].map(class_to_index).to_numpy(dtype=np.int64)
    original_preds, original_confidences = predict(model, X_test, device)

    selected_df = select_correct_examples(
        test_df,
        y_true,
        original_preds,
        classes,
        args.n_texts_per_class,
    )
    if selected_df.empty:
        raise ValueError('No correctly classified homme/femme texts were selected.')
    output_paths = resolve_output_paths(args, len(selected_df))

    perturbed_texts, changed_terms_by_row = make_perturbed_texts(
        selected_df,
        terms_by_class,
        args.mode,
    )
    X_perturbed = vectorizer.transform(perturbed_texts).toarray().astype(np.float32)
    perturbed_preds, perturbed_confidences = predict(model, X_perturbed, device)

    rows_df = build_perturbation_rows(
        selected_df,
        y_true,
        original_preds,
        original_confidences,
        perturbed_preds,
        perturbed_confidences,
        classes,
        terms_by_class,
        perturbed_texts,
        changed_terms_by_row,
        args.mode,
        args.method,
    )
    summary = build_summary(
        rows_df,
        args.method,
        args.mode,
        args.n_terms,
        args.n_texts_per_class,
        terms_by_class,
    )

    ensure_parent_dir(output_paths['output_csv'])
    rows_df.to_csv(output_paths['output_csv'], index=False, encoding='utf-8')
    save_json(summary, output_paths['summary_json'])
    save_accuracy_plot(summary, output_paths['accuracy_plot'])
    save_confidence_plot(summary, output_paths['confidence_plot'])

    print(f'Saved perturbation rows: {output_paths["output_csv"]}')
    print(f'Saved summary: {output_paths["summary_json"]}')
    print(f'Saved accuracy plot: {output_paths["accuracy_plot"]}')
    print(f'Saved confidence plot: {output_paths["confidence_plot"]}')
    print('Interpretation: drops in accuracy/confidence support explanation faithfulness to this model.')
    print('These results show what the classifier learned, not universal male/female writing rules.')


if __name__ == '__main__':
    main()
