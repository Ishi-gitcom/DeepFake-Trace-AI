
import cv2
import csv
import numpy as np
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = ROOT / "data" / "raw" / "sdfvd"
IMAGE_DIR = ROOT / "data" / "raw" / "standalone_images"

PROCESSED_DIR = ROOT / "data" / "processed"

VIDEO_OUTPUT = PROCESSED_DIR / "frames"
IMAGE_OUTPUT = PROCESSED_DIR / "standalone_images"

FRAMES_PER_VIDEO = 8
IMAGE_SIZE = (128, 128)
FACE_MARGIN = 0.15

# First run: test only one video per class.
# After successful testing, change to False.
TEST_MODE = False

LABEL_FOLDERS = {
    "Real": "real",
    "Fake": "fake"
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ==========================================
# 2. INITIALISE FACE DETECTOR
# ==========================================

cascade_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Face detector could not be loaded.")

# ==========================================
# 3. CREATE OUTPUT DIRECTORIES
# ==========================================

for label in LABEL_FOLDERS.values():
    (VIDEO_OUTPUT / label).mkdir(parents=True, exist_ok=True)
    (IMAGE_OUTPUT / label).mkdir(parents=True, exist_ok=True)

# ==========================================
# 4. LOGGING
# ==========================================

manifest = []
failure_log = []
video_log = []
image_log = []

# ==========================================
# 5. COMMON FACE EXTRACTION FUNCTION
# ==========================================

def extract_face(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None

    # Select the largest detected face.
    x, y, w, h = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    # Add margin around the face.
    mx = int(w * FACE_MARGIN)
    my = int(h * FACE_MARGIN)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(frame.shape[1], x + w + mx)
    y2 = min(frame.shape[0], y + h + my)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.resize(
        crop,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    return crop

# ==========================================
# 6. PROCESS VIDEO
# ==========================================

def process_video(video_path, label):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        video_log.append([
            video_path.name, label, 0, 0, "OPEN_FAILED"
        ])
        cap.release()
        return

    try:
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            video_log.append([
                video_path.name, label, 0, 0,
                "INVALID_VIDEO"
            ])
            return

        selected = np.linspace(
            0,
            total_frames - 1,
            num=min(FRAMES_PER_VIDEO, total_frames),
            dtype=int
        )

        video_id = f"{label}_{video_path.stem}"
        saved_count = 0

        for frame_index in selected:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index)
            )

            success, frame = cap.read()

            if not success:
                failure_log.append([
                    video_id, int(frame_index),
                    "FRAME_READ_FAILED"
                ])
                continue

            face = extract_face(frame)

            if face is None:
                failure_log.append([
                    video_id, int(frame_index),
                    "FACE_NOT_DETECTED"
                ])
                continue

            filename = (
                f"{video_id}_frame_{frame_index:06d}.jpg"
            )

            output_path = VIDEO_OUTPUT / label / filename

            if not cv2.imwrite(str(output_path), face):
                failure_log.append([
                    video_id, int(frame_index),
                    "IMAGE_SAVE_FAILED"
                ])
                continue

            saved_count += 1

            manifest.append([
                "video_frame",
                video_id,
                label,
                int(frame_index),
                str(output_path.relative_to(ROOT))
            ])

        status = (
            "SUCCESS" if saved_count > 0
            else "NO_FACES_SAVED"
        )

        video_log.append([
            video_id,
            label,
            total_frames,
            saved_count,
            status
        ])

        print(
            f"[VIDEO] {video_id}: "
            f"{saved_count} faces saved"
        )

    finally:
        cap.release()

# ==========================================
# 7. PROCESS STANDALONE IMAGES
# ==========================================

def process_image(image_path, label):

    image = cv2.imread(str(image_path))

    if image is None:
        image_log.append([
            image_path.name, label, "READ_FAILED"
        ])
        return

    face = extract_face(image)

    if face is None:
        image_log.append([
            image_path.name, label, "FACE_NOT_DETECTED"
        ])
        return

    filename = f"{label}_{image_path.stem}.jpg"
    output_path = IMAGE_OUTPUT / label / filename

    if not cv2.imwrite(str(output_path), face):
        image_log.append([
            image_path.name, label, "SAVE_FAILED"
        ])
        return

    image_log.append([
        image_path.name, label, "SUCCESS"
    ])

    manifest.append([
        "standalone_image",
        image_path.stem,
        label,
        -1,
        str(output_path.relative_to(ROOT))
    ])

    print(f"[IMAGE] {filename} saved")

# ==========================================
# 8. RUN VIDEO PREPROCESSING
# ==========================================

for folder_name, label in LABEL_FOLDERS.items():

    folder = VIDEO_DIR / folder_name

    videos = sorted(
        folder.rglob("*.mp4")
    )

    if TEST_MODE:
        videos = videos[:1]

    print(
        f"\nProcessing {label} videos: "
        f"{len(videos)} selected"
    )

    for video in videos:
        process_video(video, label)

# ==========================================
# 9. RUN IMAGE PREPROCESSING
# ==========================================

for label in LABEL_FOLDERS.values():

    folder = IMAGE_DIR / label

    images = sorted([
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    for image_path in images:
        process_image(image_path, label)

# ==========================================
# 10. SAVE CSV LOGS
# ==========================================

def save_csv(filename, headers, rows):

    output_path = ROOT / "data" / filename

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


save_csv(
    "manifest.csv",
    ["source_type", "source_id", "label",
     "frame_index", "image_path"],
    manifest
)

save_csv(
    "preprocessing_failures.csv",
    ["video_id", "frame_index", "reason"],
    failure_log
)

save_csv(
    "preprocessing_video_log.csv",
    ["video_id", "label", "total_frames",
     "saved_faces", "status"],
    video_log
)

save_csv(
    "preprocessing_image_log.csv",
    ["image_name", "label", "status"],
    image_log
)

# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print("\n========== PREPROCESSING SUMMARY ==========")

print("Test mode:", TEST_MODE)
print("Videos processed:", len(video_log))
print("Video face crops:", sum(
    row[3] for row in video_log
))

print("Standalone images processed:", len(image_log))
print("Standalone faces saved:", len(image_log) - sum(
    row[2] != "SUCCESS" for row in image_log
))

print("Face extraction failures:", len(failure_log))

print("\nPreprocessing completed.")
