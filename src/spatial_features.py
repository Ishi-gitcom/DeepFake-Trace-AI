
import cv2
import csv
import numpy as np
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = ROOT / "data" / "manifest.csv"

OUTPUT_DIR = ROOT / "data" / "processed" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "spatial_features.csv"

# ==========================================
# 2. SPATIAL FEATURE EXTRACTION
# ==========================================

def extract_spatial_features(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalise pixel intensities to [0, 1]
    gray = gray.astype(np.float32) / 255.0

    features = {}

    # --------------------------------------
    # A. Basic intensity statistics
    # --------------------------------------

    features["intensity_mean"] = np.mean(gray)
    features["intensity_std"] = np.std(gray)

    features["intensity_min"] = np.min(gray)
    features["intensity_max"] = np.max(gray)

    # --------------------------------------
    # B. High-pass residual analysis
    # --------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        1.0
    )

    residual = gray - blurred

    features["residual_mean"] = np.mean(residual)
    features["residual_std"] = np.std(residual)

    features["residual_abs_mean"] = np.mean(
        np.abs(residual)
    )

    features["residual_energy"] = np.mean(
        residual ** 2
    )

    # --------------------------------------
    # C. Laplacian sharpness
    # --------------------------------------

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_32F
    )

    features["laplacian_variance"] = np.var(
        laplacian
    )

    features["laplacian_abs_mean"] = np.mean(
        np.abs(laplacian)
    )

    # --------------------------------------
    # D. Sobel gradient analysis
    # --------------------------------------

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1, 0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0, 1,
        ksize=3
    )

    gradient_magnitude = np.sqrt(
        sobel_x ** 2 + sobel_y ** 2
    )

    features["sobel_x_mean"] = np.mean(
        np.abs(sobel_x)
    )

    features["sobel_y_mean"] = np.mean(
        np.abs(sobel_y)
    )

    features["gradient_mean"] = np.mean(
        gradient_magnitude
    )

    features["gradient_std"] = np.std(
        gradient_magnitude
    )

    # --------------------------------------
    # E. Canny edge density
    # --------------------------------------

    gray_uint8 = (gray * 255).astype(np.uint8)

    edges = cv2.Canny(
        gray_uint8,
        100,
        200
    )

    features["edge_density"] = np.mean(
        edges > 0
    )

    # --------------------------------------
    # F. Local texture variation
    # --------------------------------------

    local_mean = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    local_variation = gray - local_mean

    features["local_texture_std"] = np.std(
        local_variation
    )

    features["local_texture_energy"] = np.mean(
        local_variation ** 2
    )

    return features


# ==========================================
# 3. READ IMAGE MANIFEST
# ==========================================

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}"
    )

records = []

with open(
    MANIFEST_PATH,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        # This experiment currently uses
        # facial frames extracted from videos.
        if row["source_type"] != "video_frame":
            continue

        records.append(row)


print("Total facial images:", len(records))


# ==========================================
# 4. EXTRACT FEATURES
# ==========================================

results = []
failed_images = []

for index, row in enumerate(records, start=1):

    image_path = ROOT / row["image_path"]

    features = extract_spatial_features(image_path)

    if features is None:

        failed_images.append(row["image_path"])
        continue

    result = {
        "video_id": row["source_id"],
        "label": row["label"],
        "frame_index": row["frame_index"],
        **features
    }

    results.append(result)

    if index % 50 == 0:
        print(
            f"Processed {index}/{len(records)} images"
        )


# ==========================================
# 5. SAVE FEATURE DATASET
# ==========================================

if results:

    fieldnames = list(results[0].keys())

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)


# ==========================================
# 6. SAVE FAILED IMAGE LOG
# ==========================================

failure_file = OUTPUT_DIR / "spatial_feature_failures.csv"

with open(
    failure_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(["image_path"])

    for path in failed_images:
        writer.writerow([path])


# ==========================================
# 7. FINAL SUMMARY
# ==========================================

real_count = sum(
    row["label"] == "real"
    for row in results
)

fake_count = sum(
    row["label"] == "fake"
    for row in results
)

print("\n========== SPATIAL FEATURE SUMMARY ==========")

print("Images in manifest:", len(records))
print("Feature vectors generated:", len(results))
print("Failed images:", len(failed_images))

print("Real samples:", real_count)
print("Fake samples:", fake_count)

print("Number of spatial features:",
      len(results[0]) - 3 if results else 0)

print("\nSaved to:", OUTPUT_FILE)
print("Spatial feature extraction completed.")
