
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = ROOT / "data" / "processed" / "features"

RESULT_DIR = ROOT / "results"
METRIC_DIR = RESULT_DIR / "metrics"
PLOT_DIR = RESULT_DIR / "plots"

MODEL_DIR = ROOT / "models"

for folder in [METRIC_DIR, PLOT_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. CLASSIFIERS
# ==========================================

def get_models():

    return {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42
            ))
        ]),

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }


# ==========================================
# 2. METRIC CALCULATION
# ==========================================

def calculate_metrics(y_true, y_pred, y_prob):

    return {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision": precision_score(
            y_true,
            y_pred,
            pos_label="fake",
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            pos_label="fake",
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            pos_label="fake",
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            (np.array(y_true) == "fake").astype(int),
            y_prob
        )
    }


# ==========================================
# 3. CONFUSION MATRIX
# ==========================================

def save_confusion_matrix(
    y_true, y_pred, title, filename
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["real", "fake"]
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Real", "Fake"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    ax.set_title(title)

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR / filename,
        dpi=300
    )

    plt.close(fig)


# ==========================================
# 4. GROUP VALIDATION
# ==========================================

def get_fold_count(labels, groups):

    group_table = pd.DataFrame({
        "label": np.asarray(labels),
        "group": np.asarray(groups)
    }).drop_duplicates()

    minimum_groups = group_table.groupby(
        "label"
    )["group"].nunique().min()

    folds = min(5, int(minimum_groups))

    if folds < 2:
        raise ValueError(
            "Insufficient independent groups for "
            "stratified group cross-validation. "
            "At least two groups per class are required."
        )

    return folds


def run_group_validation(X, y, groups, model):

    folds = get_fold_count(y, groups)

    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42
    )

    predictions = np.empty(len(y), dtype=object)
    probabilities = np.zeros(len(y))

    for train_idx, test_idx in splitter.split(
        X, y, groups
    ):

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        predictions[test_idx] = model.predict(
            X.iloc[test_idx]
        )

        fake_index = list(model.classes_).index("fake")

        probabilities[test_idx] = model.predict_proba(
            X.iloc[test_idx]
        )[:, fake_index]

    return predictions, probabilities


# ==========================================
# 5. MODEL EVALUATION AND SAVING
# ==========================================

def evaluate_and_save(
    X,
    y,
    groups,
    level,
    model_name,
    model,
    results
):

    predictions, probabilities = run_group_validation(
        X,
        y,
        groups,
        model
    )

    metrics = calculate_metrics(
        y,
        predictions,
        probabilities
    )

    print("\n====================================")
    print(level.upper(), "-", model_name)
    print("====================================")

    print("\nEvaluation Metrics:")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            labels=["real", "fake"],
            zero_division=0
        )
    )

    save_confusion_matrix(
        y,
        predictions,
        f"{level.title()} - {model_name}",
        f"{level}_{model_name}_confusion.png"
    )

    results.append({
        "classification_level": level,
        "model": model_name,
        "samples": len(y),
        "videos": pd.Series(groups).nunique(),
        **metrics
    })

    # Train final deployment model using all available data.

    model.fit(X, y)

    model_path = (
        MODEL_DIR /
        f"{level}_{model_name}.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X.columns),
            "classification_level": level,
            "positive_class": "fake"
        },
        model_path
    )

    print("\nSaved model:", model_path)


# ==========================================
# 6. MAIN TRAINING PIPELINE
# ==========================================

def main():

    print("\n========== LOADING HYBRID FEATURES ==========")

    hybrid_path = FEATURE_DIR / "hybrid_features.csv"

    if not hybrid_path.exists():
        raise FileNotFoundError(
            "hybrid_features.csv not found. "
            "Run merge_features.py first."
        )

    hybrid = pd.read_csv(hybrid_path)

    if hybrid.empty:
        raise ValueError("Hybrid feature dataset is empty.")

    required_columns = [
        "video_id",
        "label",
        "frame_index"
    ]

    for col in required_columns:
        if col not in hybrid.columns:
            raise ValueError(
                f"Missing required column: {col}"
            )

    hybrid["label"] = (
        hybrid["label"]
        .astype(str)
        .str.lower()
    )

    if hybrid["video_id"].isnull().any():
        raise ValueError("Missing video IDs detected.")

    if hybrid["label"].isnull().any():
        raise ValueError("Missing labels detected.")

    if set(hybrid["label"].unique()) != {"real", "fake"}:
        raise ValueError(
            "Both real and fake classes must be present."
        )

    if hybrid.groupby("video_id")["label"].nunique().max() > 1:
        raise ValueError(
            "Conflicting labels detected within a video."
        )

    print("Total frame records:", len(hybrid))
    print("Unique videos:", hybrid["video_id"].nunique())

    print("\nClass Distribution:")
    print(hybrid["label"].value_counts())


    # ======================================
    # 7. IMAGE-LEVEL CLASSIFICATION
    # ======================================

    print("\n========== IMAGE-LEVEL CLASSIFICATION ==========")

    meta = [
        "video_id",
        "label",
        "frame_index",
        "source_group"
    ]

    image_columns = [
        col for col in hybrid.columns
        if col not in meta
    ]

    if not image_columns:
        raise ValueError("No numerical features found.")

    X_image = hybrid[image_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    y_image = hybrid["label"]

    # Keep related source videos together if metadata exists.

    if "source_group" in hybrid.columns:
        image_groups = hybrid["source_group"]
    else:
        image_groups = hybrid["video_id"]

    if image_groups.isnull().any():
        raise ValueError("Missing image grouping information.")

    if not np.isfinite(X_image.to_numpy()).all():
        raise ValueError(
            "Invalid or missing values in image features."
        )

    results = []

    for model_name, model in get_models().items():

        evaluate_and_save(
            X_image,
            y_image,
            image_groups,
            "image",
            model_name,
            model,
            results
        )


    # ======================================
    # 8. VIDEO-LEVEL FEATURE AGGREGATION
    # ======================================

    print("\n========== VIDEO-LEVEL CLASSIFICATION ==========")

    # Aggregate frame-level spatial and frequency
    # features into one feature vector per video.

    video_features = hybrid.groupby(
        "video_id"
    )[image_columns].mean().reset_index()

    video_labels = hybrid.groupby(
        "video_id"
    )["label"].first().reset_index()

    video_data = video_features.merge(
        video_labels,
        on="video_id",
        validate="one_to_one"
    )

    # Preserve source grouping information if available.

    if "source_group" in hybrid.columns:

        source_groups = hybrid.groupby(
            "video_id"
        )["source_group"].first().reset_index()

        video_data = video_data.merge(
            source_groups,
            on="video_id",
            validate="one_to_one"
        )

        video_groups = video_data["source_group"]

    else:

        video_groups = video_data["video_id"]

    X_video = video_data[image_columns].astype(float)

    y_video = video_data["label"]

    if not np.isfinite(X_video.to_numpy()).all():
        raise ValueError(
            "Invalid values in video-level features."
        )

    print("Total videos:", len(video_data))

    print("\nVideo Class Distribution:")
    print(y_video.value_counts())

    # Train and evaluate both video classifiers.

    for model_name, model in get_models().items():

        evaluate_and_save(
            X_video,
            y_video,
            video_groups,
            "video",
            model_name,
            model,
            results
        )


    # ======================================
    # 9. SAVE FINAL RESULTS
    # ======================================

    results_df = pd.DataFrame(results)

    results_path = (
        METRIC_DIR / "model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n========== FINAL MODEL COMPARISON ==========")

    print(
        results_df.to_string(index=False)
    )

    print("\n========== TRAINING COMPLETED ==========")

    print("\nModels saved in:", MODEL_DIR)

    print("Metrics saved in:", results_path)

    print("Confusion matrices saved in:", PLOT_DIR)

    print("\nAll training and evaluation tasks completed successfully.")


if __name__ == "__main__":
    main()
