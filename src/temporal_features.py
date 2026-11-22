
import cv2
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = ROOT / "data" / "manifest.csv"

OUTPUT_DIR = ROOT / "data" / "processed" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "temporal_features.csv"

FAILURE_FILE = OUTPUT_DIR / "temporal_feature_failures.csv"


# ==========================================
# 2. LOAD MANIFEST AND GROUP BY VIDEO
# ==========================================

videos = defaultdict(list)

with open(
    MANIFEST_PATH,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        if row["source_type"] != "video_frame":
            continue

        video_id = row["source_id"]
        videos[video_id].append(row)


# ==========================================
# 3. TEMPORAL FEATURE EXTRACTION
# ==========================================

def extract_temporal_features(frame_rows):

    # Sort frames by their original sampling order
    frame_rows = sorted(
        frame_rows,
        key=lambda row: int(row["frame_index"])
    )

    frames = []

    for row in frame_rows:

        image_path = ROOT / row["image_path"]

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Standardise image dimensions
        gray = cv2.resize(gray, (128, 128))

        gray = gray.astype(np.float32) / 255.0

        frames.append(gray)

    # At least two frames are needed
    if len(frames) < 2:
        return None

    differences = []
    correlations = []
    flow_means = []
    flow_stds = []

    # Analyse consecutive sampled frame pairs
    for i in range(len(frames) - 1):

        first = frames[i]
        second = frames[i + 1]

        # ----------------------------------
        # A. Frame difference
        # ----------------------------------

        difference = np.abs(first - second)

        differences.append(np.mean(difference))

        # ----------------------------------
        # B. Difference variation
        # ----------------------------------

        # Correlation between image intensities
        first_flat = first.flatten()
        second_flat = second.flatten()

        if (
            np.std(first_flat) > 1e-8
            and np.std(second_flat) > 1e-8
        ):

            correlation = np.corrcoef(
                first_flat,
                second_flat
            )[0, 1]

            if np.isfinite(correlation):
                correlations.append(correlation)

        # ----------------------------------
        # C. Optical flow
        # ----------------------------------

        first_uint8 = (
            first * 255
        ).astype(np.uint8)

        second_uint8 = (
            second * 255
        ).astype(np.uint8)

        flow = cv2.calcOpticalFlowFarneback(
            first_uint8,
            second_uint8,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        magnitude, angle = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1]
        )

        flow_means.append(np.mean(magnitude))

        flow_stds.append(np.std(magnitude))

    if not differences:
        return None

    # ======================================
    # 4. AGGREGATE VIDEO-LEVEL FEATURES
    # ======================================

    features = {}

    features["valid_frame_count"] = len(frames)

    features["frame_pair_count"] = len(differences)

    features["frame_difference_mean"] = np.mean(
        differences
    )

    features["frame_difference_std"] = np.std(
        differences
    )

    features["frame_difference_max"] = np.max(
        differences
    )

    features["frame_correlation_mean"] = (
        np.mean(correlations)
        if correlations else 0.0
    )

    features["optical_flow_mean"] = np.mean(
        flow_means
    )

    features["optical_flow_std"] = np.mean(
        flow_stds
    )

    return features


# ==========================================
# 5. PROCESS EACH VIDEO
# ==========================================

results = []
failures = []

print("Videos found:", len(videos))

for index, (video_id, frame_rows) in enumerate(
    videos.items(),
    start=1
):

    labels = {
        row["label"].lower()
        for row in frame_rows
    }

    if len(labels) != 1:
        failures.append([
            video_id,
            "INCONSISTENT_LABELS"
        ])
        continue

    label = labels.pop()

    features = extract_temporal_features(
        frame_rows
    )

    if features is None:

        failures.append([
            video_id,
            "INSUFFICIENT_VALID_FRAMES"
        ])
        continue

    result = {
        "video_id": video_id,
        "label": label,
        **features
    }

    results.append(result)

    if index % 10 == 0:

        print(
            f"Processed {index}/{len(videos)} videos"
        )


# ==========================================
# 6. SAVE TEMPORAL FEATURES
# ==========================================

if results:

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)


# ==========================================
# 7. SAVE FAILURE LOG
# ==========================================

with open(
    FAILURE_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "video_id",
        "reason"
    ])

    writer.writerows(failures)


# ==========================================
# 8. FINAL SUMMARY
# ==========================================

real_count = sum(
    row["label"] == "real"
    for row in results
)

fake_count = sum(
    row["label"] == "fake"
    for row in results
)

print("\n========== TEMPORAL FEATURE SUMMARY ==========")

print("Videos found:", len(videos))

print("Videos with temporal features:", len(results))

print("Videos skipped:", len(failures))

print("Real videos:", real_count)

print("Fake videos:", fake_count)

print("Temporal features:", 8)

print("\nSaved to:", OUTPUT_FILE)

print("Temporal feature extraction completed.")
