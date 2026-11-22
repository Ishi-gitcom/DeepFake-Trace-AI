
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = ROOT / "data" / "processed" / "features"

spatial_path = FEATURE_DIR / "spatial_features.csv"
frequency_path = FEATURE_DIR / "frequency_features.csv"
output_path = FEATURE_DIR / "hybrid_features.csv"

KEYS = ["video_id", "label", "frame_index"]


def main():

    spatial = pd.read_csv(spatial_path)
    frequency = pd.read_csv(frequency_path)

    for name, data in [
        ("Spatial", spatial),
        ("Frequency", frequency)
    ]:
        missing = set(KEYS) - set(data.columns)

        if missing:
            raise ValueError(
                f"{name} dataset missing columns: {missing}"
            )

        if data.duplicated(KEYS).any():
            raise ValueError(
                f"Duplicate records in {name} dataset"
            )

    spatial["label"] = spatial["label"].str.lower()
    frequency["label"] = frequency["label"].str.lower()

    merged = pd.merge(
        spatial,
        frequency,
        on=KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_spatial", "_frequency")
    )

    if len(merged) != len(spatial) or len(merged) != len(frequency):
        raise ValueError(
            "Spatial and frequency datasets do not match."
        )

    feature_columns = [
        col for col in merged.columns
        if col not in KEYS
    ]

    merged[feature_columns] = merged[
        feature_columns
    ].apply(pd.to_numeric, errors="coerce")

    if merged[feature_columns].isnull().any().any():
        raise ValueError("Missing or non-numeric feature values.")

    if not np.isfinite(
        merged[feature_columns].to_numpy()
    ).all():
        raise ValueError("Infinite feature values detected.")

    if set(merged["label"].unique()) != {"real", "fake"}:
        raise ValueError("Dataset must contain real and fake labels.")

    merged.to_csv(output_path, index=False)

    print("\n========== HYBRID FEATURE SUMMARY ==========")
    print("Spatial records:", len(spatial))
    print("Frequency records:", len(frequency))
    print("Merged records:", len(merged))
    print("Total features:", len(feature_columns))
    print("Unique videos:", merged["video_id"].nunique())
    print("\nClass distribution:")
    print(merged["label"].value_counts())
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
