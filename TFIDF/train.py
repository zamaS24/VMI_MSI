import argparse
from pathlib import Path


BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_LAYERS = (64, 32)
OUTPUT_DIM = 2

DEFAULT_BASE_DIR = Path('data') / 'datasetSujet3' / 'content' / 'dataset'
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'


def load_runtime_dependencies(require_plot=True):
    global plt, np, pd, torch, nn, optim
    global classification_report, confusion_matrix, DataLoader, TensorDataset
    global sns
    global load_datasets, MLPNet
    global RANDOM_SEED, TFIDF_PARAMS
    global build_tfidf_features, encode_labels, ensure_parent_dir
    global save_json, save_pickle, set_random_seed

    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import classification_report, confusion_matrix
    from torch.utils.data import DataLoader, TensorDataset

    if require_plot:
        import matplotlib.pyplot as plt
        import numpy as np

        try:
            import seaborn as sns
        except ImportError:
            sns = None
    else:
        plt = None
        np = None
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
        help='Directory used for saved model, vectorizer, and metrics.',
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
        '--history-path',
        type=Path,
        default=None,
        help='Path for the saved training history CSV file.',
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


def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })

        print(
            f'Epoch {epoch:02d}/{EPOCHS} | '
            f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} | '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}'
        )

    return pd.DataFrame(history)


def collect_predictions(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            logits = model(features.to(device))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_labels, all_preds


def plot_confusion_matrix(cm, classes):
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


def make_metrics(
    train_df,
    val_df,
    test_df,
    le,
    tfidf,
    history_df,
    test_loss,
    test_acc,
    all_labels,
    all_preds,
    cm,
):
    return {
        'random_seed': RANDOM_SEED,
        'tfidf_params': TFIDF_PARAMS,
        'model': {
            'input_dim': len(tfidf.get_feature_names_out()),
            'hidden_layers': HIDDEN_LAYERS,
            'output_dim': OUTPUT_DIM,
        },
        'training': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
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
            'accuracy': test_acc,
            'classification_report': classification_report(
                all_labels,
                all_preds,
                target_names=le.classes_,
                output_dict=True,
            ),
            'confusion_matrix': cm,
        },
    }


def save_checkpoint(model, path, le, tfidf):
    ensure_parent_dir(path)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'input_dim': len(tfidf.get_feature_names_out()),
            'hidden_layers': HIDDEN_LAYERS,
            'output_dim': OUTPUT_DIM,
            'classes': le.classes_.tolist(),
            'class_mapping': dict(zip(le.classes_, range(len(le.classes_)))),
            'tfidf_params': TFIDF_PARAMS,
        },
        path,
    )


def main():
    args = parse_args()

    load_runtime_dependencies(require_plot=not args.no_show_plot)

    if not args.no_show_plot:
        if sns is not None:
            sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['font.size'] = 10

    model_path = args.model_path or args.output_dir / 'tfidf_mlp_model.pt'
    vectorizer_path = args.vectorizer_path or args.output_dir / 'tfidf_vectorizer.pkl'
    metrics_path = args.metrics_path or args.output_dir / 'metrics.json'
    history_path = args.history_path or args.output_dir / 'history.csv'

    set_random_seed(RANDOM_SEED)

    train_df, val_df, test_df = load_datasets(args.base_dir)
    print_dataset_summary(train_df, val_df, test_df)

    le = encode_labels(train_df, val_df, test_df)
    tfidf, X_train, X_val, X_test, y_train, y_val, y_test = build_tfidf_features(
        train_df, val_df, test_df
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = make_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    model = MLPNet(
        input_dim=X_train.shape[1],
        hidden_layers=HIDDEN_LAYERS,
        output_dim=OUTPUT_DIM,
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

    all_labels, all_preds = collect_predictions(model, test_loader, device)
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    cm = confusion_matrix(all_labels, all_preds)

    metrics = make_metrics(
        train_df,
        val_df,
        test_df,
        le,
        tfidf,
        history_df,
        test_loss,
        test_acc,
        all_labels,
        all_preds,
        cm,
    )

    save_checkpoint(model, model_path, le, tfidf)
    save_pickle(tfidf, vectorizer_path)
    save_json(metrics, metrics_path)
    ensure_parent_dir(history_path)
    history_df.to_csv(history_path, index=False)

    print(f'Saved model: {model_path}')
    print(f'Saved vectorizer: {vectorizer_path}')
    print(f'Saved metrics: {metrics_path}')
    print(f'Saved history: {history_path}')

    if not args.no_show_plot:
        plot_confusion_matrix(cm, le.classes_)
        plt.show()


if __name__ == '__main__':
    main()
