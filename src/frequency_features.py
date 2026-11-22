
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

OUTPUT_FILE = OUTPUT_DIR / "frequency_features.csv"


# ==========================================
# 2. CREATE FREQUENCY BANDS
# ==========================================

def create_frequency_bands(height, width, centered=True):

    if centered:
        fy = np.fft.fftshift(np.fft.fftfreq(height))
        fx = np.fft.fftshift(np.fft.fftfreq(width))
    else:
        fy = np.arange(height) / height
        fx = np.arange(width) / width

    X, Y = np.meshgrid(fx, fy)

    radius = np.sqrt(X**2 + Y**2)

    # Normalize radial frequency to [0, 1]
    radius = radius / np.sqrt(0.5**2 + 0.5**2)

    low = radius < 0.15

    middle = (radius >= 0.15) & (radius < 0.45)

    high = radius >= 0.45

    return low, middle, high, radius


# ==========================================
# 3. FFT FEATURE EXTRACTION
# ==========================================

def extract_fft_features(gray):

    height, width = gray.shape

    # Fourier Transform
    fft = np.fft.fft2(gray)

    # Shift zero frequency to the centre
    fft_shifted = np.fft.fftshift(fft)

    # Power spectrum
    power = np.abs(fft_shifted) ** 2

    low, middle, high, radius = create_frequency_bands(
        height, width, centered=True
    )

    # Exclude the DC component
    dc_y = height // 2
    dc_x = width // 2

    power[dc_y, dc_x] = 0

    total_energy = np.sum(power) + 1e-12

    features = {}

    # Frequency energy ratios
    features["fft_low_energy_ratio"] = (
        np.sum(power[low]) / total_energy
    )

    features["fft_middle_energy_ratio"] = (
        np.sum(power[middle]) / total_energy
    )

    features["fft_high_energy_ratio"] = (
        np.sum(power[high]) / total_energy
    )

    # Log power spectrum
    log_power = np.log1p(power)

    features["fft_log_energy_mean"] = np.mean(log_power)

    features["fft_log_energy_std"] = np.std(log_power)

    # Spectral entropy
    probability = power.flatten() / total_energy

    probability = probability[probability > 0]

    entropy = -np.sum(
        probability * np.log(probability + 1e-12)
    )

    features["fft_spectral_entropy"] = (
        entropy / np.log(len(probability))
        if len(probability) > 1 else 0
    )

    # Spectral centroid
    features["fft_spectral_centroid"] = (
        np.sum(power * radius) / total_energy
    )

    return features


# ==========================================
# 4. DCT FEATURE EXTRACTION
# ==========================================

def extract_dct_features(gray):

    height, width = gray.shape

    # 2D Discrete Cosine Transform
    dct = cv2.dct(gray.astype(np.float32))

    power = dct ** 2

    low, middle, high, radius = create_frequency_bands(
        height, width, centered=False
    )

    # Exclude DC coefficient
    power[0, 0] = 0

    total_energy = np.sum(power) + 1e-12

    features = {}

    features["dct_low_energy_ratio"] = (
        np.sum(power[low]) / total_energy
    )

    features["dct_middle_energy_ratio"] = (
        np.sum(power[middle]) / total_energy
    )

    features["dct_high_energy_ratio"] = (
        np.sum(power[high]) / total_energy
    )

    features["dct_log_energy"] = np.log1p(
        np.sum(power)
    )

    return features


# ==========================================
# 5. COMBINED FREQUENCY FEATURES
# ==========================================

def extract_frequency_features(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Normalize to [0, 1]
    gray = gray.astype(np.float32) / 255.0

    features = {}

    features.update(
        extract_fft_features(gray)
    )

    features.update(
        extract_dct_features(gray)
    )

    return features


# ==========================================
# 6. READ MANIFEST
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

        if row["source_type"] != "video_frame":
            continue

        records.append(row)


print("Total facial images:", len(records))


# ==========================================
# 7. EXTRACT FEATURES
# ==========================================

results = []
failed_images = []

for index, row in enumerate(records, start=1):

    image_path = ROOT / row["image_path"]

    features = extract_frequency_features(image_path)

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
# 8. SAVE FEATURE DATASET
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
# 9. SAVE FAILURE LOG
# ==========================================

failure_file = (
    OUTPUT_DIR / "frequency_feature_failures.csv"
)

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
# 10. FINAL SUMMARY
# ==========================================

real_count = sum(
    row["label"] == "real"
    for row in results
)

fake_count = sum(
    row["label"] == "fake"
    for row in results
)

print("\n========== FREQUENCY FEATURE SUMMARY ==========")

print("Images in manifest:", len(records))

print("Feature vectors generated:", len(results))

print("Failed images:", len(failed_images))

print("Real samples:", real_count)

print("Fake samples:", fake_count)

print(
    "Number of frequency features:",
    len(results[0]) - 3 if results else 0
)

print("\nSaved to:", OUTPUT_FILE)

print("Frequency feature extraction completed.")
