"""Train and compare activity classifiers on the samples from collect.py.

    python train.py

Three models are trained (Random Forest, SVM, neural network) and tested on
held-out recordings. The best one is retrained on all data and saved to
models/activity_model.joblib, which main.py then loads automatically.
"""

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from collect import DATA_PATH
from features import FEATURE_NAMES
from model import DEFAULT_MODEL_PATH


CANDIDATES = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    # The calibration wrapper lets the SVM give probabilities (used as confidence).
    "SVM (RBF kernel)": make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            SVC(C=10, gamma="scale", class_weight="balanced", random_state=42), ensemble=False
        ),
    ),
    "Neural network (MLP)": make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    ),
}
MIN_SAMPLES_PER_LABEL = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Train an activity classifier.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="samples CSV from collect.py")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="where to save the model")
    return parser.parse_args()


def load_samples(path):
    if not path.exists():
        raise SystemExit(f"{path} not found. Record some samples first with:  python collect.py")
    with open(path, newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        if header[2:] != list(FEATURE_NAMES):
            raise SystemExit(f"{path} was recorded with different features. Record it again.")
        rows = list(reader)
    clips = np.array([row[0] for row in rows])
    labels = np.array([row[1] for row in rows])
    features = np.array([row[2:] for row in rows], dtype=np.float32)
    return features, labels, clips


def split_by_clip(labels, clips, test_share=0.2):
    """Choose test frames that were never next to a training frame.

    Neighbouring video frames are almost identical, so a random split would
    give an unrealistically high score. If an activity has several recorded
    clips, its last clip is the test set. If it has only one clip, the last
    20% of that clip (in time) is the test set.
    """
    test = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        label_clips = list(dict.fromkeys(clips[labels == label]))
        if len(label_clips) >= 2:
            test |= (labels == label) & (clips == label_clips[-1])
        else:
            indices = np.flatnonzero(labels == label)
            test[indices[int(len(indices) * (1 - test_share)):]] = True
    return ~test, test


def save_confusion_matrix(true, predicted, labels, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    counts = confusion_matrix(true, predicted, labels=labels)
    row_share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    blues = LinearSegmentedColormap.from_list(
        "blues", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"]
    )

    size = 1.0 + 0.75 * len(labels)
    fig, ax = plt.subplots(figsize=(size + 1.5, size), facecolor="#fcfcfb")
    ax.pcolormesh(row_share, cmap=blues, vmin=0, vmax=1, edgecolors="#fcfcfb", linewidth=2)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if counts[i, j]:
                color = "#ffffff" if row_share[i, j] > 0.55 else "#0b0b0b"
                ax.text(j + 0.5, i + 0.5, counts[i, j], ha="center", va="center", color=color, fontsize=10)

    ticks = np.arange(len(labels)) + 0.5
    ax.set_xticks(ticks, labels, rotation=35, ha="right", color="#52514e")
    ax.set_yticks(ticks, labels, color="#52514e")
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Predicted activity", color="#52514e")
    ax.set_ylabel("True activity", color="#52514e")
    ax.set_title("Confusion matrix (test frames)", loc="left", color="#0b0b0b", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    args = parse_args()
    features, labels, clips = load_samples(args.data)

    counts = Counter(labels)
    print("Samples per activity:")
    for label, count in sorted(counts.items()):
        print(f"  {label:<20}{count:>6}")
    too_few = [label for label, count in counts.items() if count < MIN_SAMPLES_PER_LABEL]
    if too_few or len(counts) < 2:
        raise SystemExit(
            f"\nNeed at least 2 activities with {MIN_SAMPLES_PER_LABEL}+ frames each. "
            f"Record more of: {', '.join(too_few) or 'another activity'}"
        )

    train, test = split_by_clip(labels, clips)
    print(f"\nTraining on {train.sum()} frames, testing on {test.sum()} held-out frames.\n")

    print(f"{'Model':<24}{'Accuracy':>10}{'Macro F1':>10}")
    results = {}
    for name, candidate in CANDIDATES.items():
        classifier = clone(candidate).fit(features[train], labels[train])
        predicted = classifier.predict(features[test])
        accuracy = accuracy_score(labels[test], predicted)
        macro_f1 = f1_score(labels[test], predicted, average="macro")
        results[name] = (macro_f1, accuracy, predicted)
        print(f"{name:<24}{accuracy:>10.1%}{macro_f1:>10.3f}")

    best_name = max(results, key=lambda name: results[name][0])
    best_f1, best_accuracy, best_predicted = results[best_name]
    label_names = sorted(counts)
    report = classification_report(labels[test], best_predicted, labels=label_names, zero_division=0)
    print(f"\nBest model: {best_name}\n\n{report}")

    # Retrain the winner on every frame before saving it.
    final_classifier = clone(CANDIDATES[best_name]).fit(features, labels)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": final_classifier,
            "model_name": best_name,
            "feature_names": list(FEATURE_NAMES),
            "test_accuracy": best_accuracy,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        },
        args.model,
    )

    report_path = args.model.parent / "report.txt"
    with open(report_path, "w") as file:
        file.write(f"Best model: {best_name}\n")
        file.write(f"Test accuracy: {best_accuracy:.1%}   Macro F1: {best_f1:.3f}\n\n")
        for name, (macro_f1, accuracy, _) in results.items():
            file.write(f"{name:<24}{accuracy:>10.1%}{macro_f1:>10.3f}\n")
        file.write(f"\n{report}")

    chart_path = args.model.parent / "confusion_matrix.png"
    save_confusion_matrix(labels[test], best_predicted, label_names, chart_path)

    print(f"Model saved to {args.model}")
    print(f"Report saved to {report_path}")
    print(f"Confusion matrix saved to {chart_path}")
    print("\nRun  python main.py  to use it.")


if __name__ == "__main__":
    main()
