import argparse
import copy
from pathlib import Path


BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
HIDDEN_LAYERS = (512, 128, 32)
DROPOUT_INPUT = 0.35
DROPOUT_HIDDEN = 0.45
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 1e-4
OUTPUT_DIM = 2

DEFAULT_BASE_DIR = Path('data') / 'datasetSujet3' / 'content' / 'dataset'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / 'artifacts'
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
DEFAULT_CONFUSION_MATRIX_PATH = PROJECT_ROOT / 'vis' / 'tfidf_confusion_matrix.png'


def load_runtime_dependencies(show_plot=True):
    global plt, np, pd, torch, nn, optim
    global classification_report, confusion_matrix, DataLoader, TensorDataset
    global accuracy_score, f1_score
    global sns
    global load_datasets, MLPNet
    global RANDOM_SEED, TFIDF_PARAMS
    global build_tfidf_features, encode_labels, ensure_parent_dir
    global save_json, save_pickle, set_random_seed

    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from torch.utils.data import DataLoader, TensorDataset

    import matplotlib

    if not show_plot:
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    import numpy as np

    try:
        import seaborn as sns
    except ImportError:
        sns = None

    try:
        from .data_loader import load_datasets
        from .model import MLPNet
        from .utils import (
            RANDOM_SEED,
            TFIDF_PARAMS,
            build_tfidf_features,
            encode_labels,
            ensure_parent_dir,
            save_json,
            save_pickle,
            set_random_seed,
        )
    except ImportError:
        from data_loader import load_datasets
        from model import MLPNet
        from utils import (
            RANDOM_SEED,
            TFIDF_PARAMS,
            build_tfidf_features,
            encode_labels,
            ensure_parent_dir,
            save_json,
            save_pickle,
            set_random_seed,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the lab-imposed TF-IDF + MLP pipeline.'
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=DEFAULT_BASE_DIR,
        help='Dataset directory containing train, val, and test folders.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory used for saved model, vectorizer, and history.',
    )
    parser.add_argument(
        '--artifact-dir',
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help='Directory used for saved evaluation metrics and predictions.',
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        default=None,
        help='Path for the saved PyTorch checkpoint.',
    )
    parser.add_argument(
        '--vectorizer-path',
        type=Path,
        default=None,
        help='Path for the saved fitted TF-IDF vectorizer pickle.',
    )
    parser.add_argument(
        '--metrics-path',
        type=Path,
        default=None,
        help='Path for the saved metrics JSON file.',
    )
    parser.add_argument(
        '--predictions-path',
        type=Path,
        default=None,
        help='Path for the saved test prediction CSV file.',
    )
    parser.add_argument(
        '--confusion-matrix-path',
        type=Path,
        default=DEFAULT_CONFUSION_MATRIX_PATH,
        help='Path for the saved confusion matrix PNG file.',
    )
    parser.add_argument(
        '--history-path',
        type=Path,
        default=None,
        help='Path for the saved training history CSV file.',
    )
    parser.add_argument(
        '--loss-plot-path',
        type=Path,
        default=None,
        help='Path for the saved train/validation loss curve PNG file.',
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=None,
        help='Optional maximum TF-IDF vocabulary size. Omit it to keep all retained terms.',
    )
    parser.add_argument(
        '--min-df',
        type=int,
        default=None,
        help='Ignore terms that appear in fewer than this many training texts.',
    )
    parser.add_argument(
        '--max-df',
        type=float,
        default=None,
        help='Ignore terms that appear in more than this document-frequency ratio.',
    )
    parser.add_argument(
        '--no-show-plot',
        action='store_true',
        help='Save metrics without opening the confusion matrix window.',
    )
    return parser.parse_args()


def print_dataset_summary(train_df, val_df, test_df):
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("\nDistribution des classes:")
    print("Train:\n", train_df['label'].value_counts())
    print("Val:\n", val_df['label'].value_counts())
    print("Test:\n", test_df['label'].value_counts())


def resolve_tfidf_params(args):
    params = dict(TFIDF_PARAMS)

    if args.max_features is not None:
        params['max_features'] = args.max_features
    if args.min_df is not None:
        params['min_df'] = args.min_df
    if args.max_df is not None:
        params['max_df'] = args.max_df

    return params


def make_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test):
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test).long()
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    patience=EARLY_STOPPING_PATIENCE,
    min_delta=EARLY_STOPPING_MIN_DELTA,
):
    history = []
    best_state_dict = None
    best_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        improved = val_loss < best_val_loss - min_delta
        if improved:
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'is_best': improved,
        })

        print(
            f'Epoch {epoch:02d}/{EPOCHS} | '
            f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} | '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}'
            f'{" | best" if improved else ""}'
        )

        if patience and epochs_without_improvement >= patience:
            print(
                f'Early stopping after {epoch} epochs '
                f'(best val_loss={best_val_loss:.4f} at epoch {best_epoch}).'
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f'Restored best validation checkpoint from epoch {best_epoch}.')

    history_df = pd.DataFrame(history)
    history_df.attrs['best_epoch'] = best_epoch
    history_df.attrs['best_val_loss'] = best_val_loss
    return history_df


def collect_predictions(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for features, labels in test_loader:
            logits = model(features.to(device))
            probabilities = torch.softmax(logits, dim=1)
            confidences, preds = probabilities.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_labels, all_preds, all_confidences


def plot_confusion_matrix(cm, classes):
    plt.figure()
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes,
        )
    else:
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        plt.xticks(np.arange(len(classes)), classes)
        plt.yticks(np.arange(len(classes)), classes)
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                plt.text(col, row, cm[row, col], ha='center', va='center')

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('MLP TF-IDF Confusion Matrix')
    plt.tight_layout()


def save_confusion_matrix(cm, classes, path):
    plot_confusion_matrix(cm, classes)
    ensure_parent_dir(path)
    plt.savefig(path, dpi=150, bbox_inches='tight')


def plot_loss_curve(history_df):
    plt.figure()
    plt.plot(
        history_df['epoch'],
        history_df['train_loss'],
        marker='o',
        label='Train loss',
    )
    plt.plot(
        history_df['epoch'],
        history_df['val_loss'],
        marker='o',
        label='Validation loss',
    )
    plt.xlabel('Epoch')
    plt.ylabel('Cross-entropy loss')
    plt.title('TF-IDF MLP Training Loss')
    plt.xticks(history_df['epoch'])
    plt.legend()
    plt.tight_layout()


def save_loss_curve(history_df, path):
    plot_loss_curve(history_df)
    ensure_parent_dir(path)
    plt.savefig(path, dpi=150, bbox_inches='tight')


def build_prediction_rows(test_df, le, all_preds, all_confidences, preview_chars=200):
    predicted_labels = le.inverse_transform(all_preds)

    return pd.DataFrame({
        'file_path': test_df['path'].values,
        'true_label': test_df['label'].values,
        'predicted_label': predicted_labels,
        'confidence': all_confidences,
        'text_preview': (
            test_df['text']
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
            .str[:preview_chars]
            .values
        ),
    })


def save_predictions_csv(test_df, le, all_preds, all_confidences, path):
    predictions_df = build_prediction_rows(test_df, le, all_preds, all_confidences)
    ensure_parent_dir(path)
    predictions_df.to_csv(path, index=False, encoding='utf-8')
    return predictions_df


def make_metrics(
    train_df,
    val_df,
    test_df,
    le,
    tfidf,
    tfidf_params,
    history_df,
    test_loss,
    test_acc,
    all_labels,
    all_preds,
    cm,
):
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=le.classes_,
        output_dict=True,
    )
    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=le.classes_,
    )
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    return {
        'random_seed': RANDOM_SEED,
        'tfidf_params': tfidf_params,
        'model': {
            'architecture': 'BatchNorm1d + MLP(512, 128, 32) + LayerNorm + GELU',
            'input_dim': len(tfidf.get_feature_names_out()),
            'hidden_layers': HIDDEN_LAYERS,
            'output_dim': OUTPUT_DIM,
            'dropout_input': DROPOUT_INPUT,
            'dropout_hidden': DROPOUT_HIDDEN,
            'uses_batch_norm': True,
            'uses_layer_norm': True,
        },
        'training': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'early_stopping_min_delta': EARLY_STOPPING_MIN_DELTA,
            'best_epoch': history_df.attrs.get('best_epoch'),
            'best_val_loss': history_df.attrs.get('best_val_loss'),
        },
        'class_mapping': dict(zip(le.classes_, range(len(le.classes_)))),
        'dataset': {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'train_distribution': train_df['label'].value_counts().to_dict(),
            'val_distribution': val_df['label'].value_counts().to_dict(),
            'test_distribution': test_df['label'].value_counts().to_dict(),
        },
        'features': {
            'x_train_shape': list(history_df.attrs['x_train_shape']),
            'vocabulary_size': len(tfidf.get_feature_names_out()),
        },
        'history': history_df.to_dict(orient='records'),
        'test': {
            'loss': test_loss,
            'accuracy': accuracy,
            'run_epoch_accuracy': test_acc,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'classification_report': report_dict,
            'classification_report_text': report_text,
            'confusion_matrix': cm,
        },
    }


def save_checkpoint(model, path, le, tfidf, tfidf_params):
    ensure_parent_dir(path)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'architecture': 'BatchNorm1d + MLP(512, 128, 32) + LayerNorm + GELU',
            'input_dim': len(tfidf.get_feature_names_out()),
            'hidden_layers': HIDDEN_LAYERS,
            'output_dim': OUTPUT_DIM,
            'dropout_input': DROPOUT_INPUT,
            'dropout_hidden': DROPOUT_HIDDEN,
            'uses_batch_norm': True,
            'uses_layer_norm': True,
            'classes': le.classes_.tolist(),
            'class_mapping': dict(zip(le.classes_, range(len(le.classes_)))),
            'tfidf_params': tfidf_params,
        },
        path,
    )


def main():
    args = parse_args()

    load_runtime_dependencies(show_plot=not args.no_show_plot)

    if sns is not None:
        sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 10

    model_path = args.model_path or args.output_dir / 'tfidf_mlp_model.pt'
    vectorizer_path = args.vectorizer_path or args.output_dir / 'tfidf_vectorizer.pkl'
    metrics_path = args.metrics_path or args.artifact_dir / 'metrics.json'
    predictions_path = args.predictions_path or args.artifact_dir / 'test_predictions.csv'
    history_path = args.history_path or args.output_dir / 'history.csv'
    loss_plot_path = args.loss_plot_path or args.output_dir / 'loss_curve.png'

    set_random_seed(RANDOM_SEED)

    train_df, val_df, test_df = load_datasets(args.base_dir)
    print_dataset_summary(train_df, val_df, test_df)

    le = encode_labels(train_df, val_df, test_df)
    tfidf_params = resolve_tfidf_params(args)
    tfidf, X_train, X_val, X_test, y_train, y_val, y_test = build_tfidf_features(
        train_df, val_df, test_df, tfidf_params=tfidf_params
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = make_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    model = MLPNet(
        input_dim=X_train.shape[1],
        output_dim=OUTPUT_DIM,
        dropout_input=DROPOUT_INPUT,
        dropout_hidden=DROPOUT_HIDDEN,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(model)
    print(f'Using device: {device}')

    history_df = train_model(model, train_loader, val_loader, criterion, optimizer, device)
    history_df.attrs['x_train_shape'] = X_train.shape

    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    print(f'Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}')

    all_labels, all_preds, all_confidences = collect_predictions(model, test_loader, device)
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    cm = confusion_matrix(all_labels, all_preds)

    metrics = make_metrics(
        train_df,
        val_df,
        test_df,
        le,
        tfidf,
        tfidf_params,
        history_df,
        test_loss,
        test_acc,
        all_labels,
        all_preds,
        cm,
    )

    save_checkpoint(model, model_path, le, tfidf, tfidf_params)
    save_pickle(tfidf, vectorizer_path)
    save_predictions_csv(test_df, le, all_preds, all_confidences, predictions_path)
    save_json(metrics, metrics_path)
    ensure_parent_dir(history_path)
    history_df.to_csv(history_path, index=False)
    save_loss_curve(history_df, loss_plot_path)
    save_confusion_matrix(cm, le.classes_, args.confusion_matrix_path)

    print(f'Saved model: {model_path}')
    print(f'Saved vectorizer: {vectorizer_path}')
    print(f'Saved predictions: {predictions_path}')
    print(f'Saved metrics: {metrics_path}')
    print(f'Saved history: {history_path}')
    print(f'Saved loss curve: {loss_plot_path}')
    print(f'Saved confusion matrix: {args.confusion_matrix_path}')

    if not args.no_show_plot:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
