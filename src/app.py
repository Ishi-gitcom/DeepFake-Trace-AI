
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import tempfile
import os

from pathlib import Path


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

IMAGE_SIZE = (128, 128)
FACE_MARGIN = 0.15
FRAMES_PER_VIDEO = 8

st.set_page_config(
    page_title="DeepTrace AI",
    page_icon="🔍",
    layout="wide"
)


# ==========================================
# 2. LOAD FACE DETECTOR
# ==========================================

cascade_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    st.error("Face detector could not be loaded.")
    st.stop()


# ==========================================
# 3. LOAD TRAINED MODELS
# ==========================================

@st.cache_resource
def load_model(filename):

    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    return joblib.load(path)


# ==========================================
# 4. FACE PREPROCESSING
# ==========================================

def extract_face(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    mx = int(w * FACE_MARGIN)
    my = int(h * FACE_MARGIN)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)

    x2 = min(frame.shape[1], x + w + mx)
    y2 = min(frame.shape[0], y + h + my)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    resized = cv2.resize(
        crop,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    success, encoded = cv2.imencode(
        ".jpg",
        resized
    )

    if not success:
        return None

    return cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR
    )


# ==========================================
# 5. SPATIAL FEATURE EXTRACTION
# ==========================================

def extract_spatial_features(face):

    gray = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    gray = gray.astype(np.float32) / 255.0

    features = {}

    features["intensity_mean"] = np.mean(gray)
    features["intensity_std"] = np.std(gray)
    features["intensity_min"] = np.min(gray)
    features["intensity_max"] = np.max(gray)

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

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = np.sqrt(
        sobel_x ** 2 + sobel_y ** 2
    )

    features["sobel_x_mean"] = np.mean(
        np.abs(sobel_x)
    )

    features["sobel_y_mean"] = np.mean(
        np.abs(sobel_y)
    )

    features["gradient_mean"] = np.mean(gradient)
    features["gradient_std"] = np.std(gradient)

    gray_uint8 = (gray * 255).astype(np.uint8)

    edges = cv2.Canny(
        gray_uint8,
        100,
        200
    )

    features["edge_density"] = np.mean(edges > 0)

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
# 6. FREQUENCY BAND GENERATION
# ==========================================

def create_frequency_bands(
    height,
    width,
    centered=True
):

    if centered:

        fy = np.fft.fftshift(
            np.fft.fftfreq(height)
        )

        fx = np.fft.fftshift(
            np.fft.fftfreq(width)
        )

    else:

        fy = np.arange(height) / height
        fx = np.arange(width) / width

    X, Y = np.meshgrid(fx, fy)

    radius = np.sqrt(X ** 2 + Y ** 2)

    radius = radius / np.sqrt(
        0.5 ** 2 + 0.5 ** 2
    )

    low = radius < 0.15

    middle = (
        (radius >= 0.15) &
        (radius < 0.45)
    )

    high = radius >= 0.45

    return low, middle, high, radius


# ==========================================
# 7. FFT FEATURE EXTRACTION
# ==========================================

def extract_fft_features(gray):

    height, width = gray.shape

    fft = np.fft.fft2(gray)

    fft_shifted = np.fft.fftshift(fft)

    power = np.abs(fft_shifted) ** 2

    low, middle, high, radius = create_frequency_bands(
        height,
        width,
        centered=True
    )

    power[
        height // 2,
        width // 2
    ] = 0

    total_energy = np.sum(power) + 1e-12

    features = {}

    features["fft_low_energy_ratio"] = (
        np.sum(power[low]) / total_energy
    )

    features["fft_middle_energy_ratio"] = (
        np.sum(power[middle]) / total_energy
    )

    features["fft_high_energy_ratio"] = (
        np.sum(power[high]) / total_energy
    )

    log_power = np.log1p(power)

    features["fft_log_energy_mean"] = np.mean(
        log_power
    )

    features["fft_log_energy_std"] = np.std(
        log_power
    )

    probability = power.flatten() / total_energy

    probability = probability[probability > 0]

    entropy = -np.sum(
        probability * np.log(probability + 1e-12)
    )

    features["fft_spectral_entropy"] = (
        entropy / np.log(len(probability))
        if len(probability) > 1 else 0
    )

    features["fft_spectral_centroid"] = (
        np.sum(power * radius) / total_energy
    )

    return features


# ==========================================
# 8. DCT FEATURE EXTRACTION
# ==========================================

def extract_dct_features(gray):

    height, width = gray.shape

    dct = cv2.dct(
        gray.astype(np.float32)
    )

    power = dct ** 2

    low, middle, high, radius = create_frequency_bands(
        height,
        width,
        centered=False
    )

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
# 9. COMPLETE FEATURE EXTRACTION
# ==========================================

def extract_all_features(face):

    features = {}

    features.update(
        extract_spatial_features(face)
    )

    gray = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    gray = gray.astype(np.float32) / 255.0

    features.update(
        extract_fft_features(gray)
    )

    features.update(
        extract_dct_features(gray)
    )

    return features


# ==========================================
# 10. MODEL PREDICTION
# ==========================================

def predict_features(features, bundle):

    model = bundle["model"]

    columns = bundle["feature_columns"]

    missing = set(columns) - set(features.keys())

    if missing:
        raise ValueError(
            f"Missing model features: {sorted(missing)}"
        )

    X = pd.DataFrame(
        [[features[col] for col in columns]],
        columns=columns
    )

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    classes = list(model.classes_)

    if "fake" not in classes or "real" not in classes:
        raise ValueError(
            f"Unexpected model classes: {classes}. "
            "Expected 'fake' and 'real'."
        )

    fake_index = classes.index("fake")
    real_index = classes.index("real")

    return {
        "prediction": prediction,
        "fake_score": float(probabilities[fake_index]),
        "real_score": float(probabilities[real_index])
    }


# ==========================================
# 11. VIDEO FRAME SAMPLING
# ==========================================

def process_video(video_bytes, suffix=".mp4"):

    temporary_path = None

    results = []

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temporary_file:

            temporary_file.write(video_bytes)
            temporary_path = temporary_file.name

        cap = cv2.VideoCapture(temporary_path)

        try:

            if not cap.isOpened():
                raise ValueError(
                    "Unable to open uploaded video."
                )

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if total_frames <= 0:
                raise ValueError(
                    "Video contains no readable frames."
                )

            selected = np.linspace(
                0,
                total_frames - 1,
                num=min(FRAMES_PER_VIDEO, total_frames),
                dtype=int
            )

            for frame_index in selected:

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    int(frame_index)
                )

                success, frame = cap.read()

                if not success:
                    continue

                face = extract_face(frame)

                if face is None:
                    continue

                features = extract_all_features(face)

                results.append({
                    "frame_index": int(frame_index),
                    "face": face,
                    "features": features
                })

        finally:

            cap.release()

    finally:

        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)

    if not results:
        raise ValueError(
            "No detectable faces were found in the sampled frames."
        )

    return results


# ==========================================
# 12. APPLICATION HEADER
# ==========================================

st.title("🔍 DeepTrace AI")

st.markdown(
    "### Intelligent DeepFake Image & Video Detection"
)

st.markdown(
    "A signal-processing-based media authenticity "
    "analysis system using spatial and frequency-domain features."
)

st.divider()


# ==========================================
# 13. SIDEBAR
# ==========================================

st.sidebar.title("DeepTrace AI")

st.sidebar.markdown(
    "**Detection Configuration**"
)

model_choice = st.sidebar.selectbox(
    "Select Classifier",
    ["SVM", "RandomForest"]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "**Analysis Techniques**"
)

st.sidebar.write("• Haar Cascade Face Detection")
st.sidebar.write("• Spatial Feature Analysis")
st.sidebar.write("• 2D FFT Analysis")
st.sidebar.write("• 2D DCT Analysis")
st.sidebar.write("• Video Frame Aggregation")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Research prototype | CPU-based implementation"
)


# ==========================================
# 14. LOAD SELECTED MODELS
# ==========================================

try:

    image_bundle = load_model(
        f"image_{model_choice}.joblib"
    )

    video_bundle = load_model(
        f"video_{model_choice}.joblib"
    )

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()


# ==========================================
# 15. INPUT MODE
# ==========================================

mode = st.radio(
    "Choose media type",
    ["Image Detection", "Video Detection"],
    horizontal=True
)


# ==========================================
# 16. IMAGE DETECTION
# ==========================================

if mode == "Image Detection":

    st.subheader("Image Authenticity Analysis")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image_bytes = np.frombuffer(
            uploaded_image.getvalue(),
            np.uint8
        )

        image = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error("Invalid image file.")

        else:

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2RGB
                    ),
                    caption="Uploaded Image",
                    use_container_width=True
                )

            face = extract_face(image)

            if face is None:

                st.warning(
                    "No face detected. Please upload "
                    "an image containing a clearly visible face."
                )

            else:

                with col2:

                    st.image(
                        cv2.cvtColor(
                            face,
                            cv2.COLOR_BGR2RGB
                        ),
                        caption="Detected Face",
                        use_container_width=True
                    )

                if st.button(
                    "Analyze Image",
                    type="primary",
                    use_container_width=True
                ):

                    try:

                        with st.spinner(
                            "Extracting spatial and frequency features..."
                        ):

                            features = extract_all_features(face)

                            result = predict_features(
                                features,
                                image_bundle
                            )

                        st.divider()

                        st.subheader("Detection Results")

                        label = result["prediction"].upper()

                        c1, c2, c3 = st.columns(3)

                        c1.metric(
                            "Prediction",
                            label
                        )

                        c2.metric(
                            "Fake-class score",
                            f"{result['fake_score'] * 100:.2f}%"
                        )

                        c3.metric(
                            "Real-class score",
                            f"{result['real_score'] * 100:.2f}%"
                        )

                        st.progress(
                            result["fake_score"]
                        )

                        if label == "FAKE":

                            st.error(
                                "The classifier predicts that "
                                "the uploaded image is FAKE."
                            )

                        else:

                            st.success(
                                "The classifier predicts that "
                                "the uploaded image is REAL."
                            )

                        with st.expander("Extracted Feature Analysis"):

                            feature_df = pd.DataFrame(
                                list(features.items()),
                                columns=["Feature", "Value"]
                            )

                            st.dataframe(
                                feature_df,
                                use_container_width=True
                            )

                        st.caption(
                            "The fake-class score is the model's "
                            "predicted probability, not a calibrated "
                            "measure of forensic certainty."
                        )

                    except Exception as error:

                        st.error(
                            f"Image analysis failed: {error}"
                        )


# ==========================================
# 17. VIDEO DETECTION
# ==========================================

else:

    st.subheader("Video Authenticity Analysis")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        video_bytes = uploaded_video.getvalue()

        st.video(video_bytes)

        if st.button(
            "Analyze Video",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Sampling frames and analyzing facial features..."
                ):

                    video_suffix = (
                        Path(uploaded_video.name).suffix.lower()
                    )

                    frame_results = process_video(
                        video_bytes,
                        video_suffix
                    )

                    feature_vectors = [
                        item["features"]
                        for item in frame_results
                    ]

                    video_columns = video_bundle["feature_columns"]

                    missing_video_features = (
                        set(video_columns) -
                        set(feature_vectors[0].keys())
                    )

                    if missing_video_features:

                        raise ValueError(
                            "The video model requires features "
                            "that are not available during inference: "
                            f"{sorted(missing_video_features)}. "
                            "Check whether the trained video model "
                            "includes temporal features."
                        )

                    video_features = {}

                    for column in video_columns:

                        video_features[column] = float(
                            np.mean([
                                item[column]
                                for item in feature_vectors
                            ])
                        )

                    result = predict_features(
                        video_features,
                        video_bundle
                    )

                    frame_predictions = []

                    for item in frame_results:

                        frame_result = predict_features(
                            item["features"],
                            image_bundle
                        )

                        frame_predictions.append({
                            "Frame": item["frame_index"],
                            "Prediction": frame_result["prediction"].upper(),
                            "Fake Score (%)": round(
                                frame_result["fake_score"] * 100,
                                2
                            )
                        })

                st.divider()

                st.subheader("Video Detection Results")

                label = result["prediction"].upper()

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Video Prediction",
                    label
                )

                c2.metric(
                    "Fake-class score",
                    f"{result['fake_score'] * 100:.2f}%"
                )

                c3.metric(
                    "Frames Analyzed",
                    len(frame_results)
                )

                st.progress(
                    result["fake_score"]
                )

                if label == "FAKE":

                    st.error(
                        "The video classifier predicts "
                        "that the uploaded video is FAKE."
                    )

                else:

                    st.success(
                        "The video classifier predicts "
                        "that the uploaded video is REAL."
                    )

                st.subheader("Frame-wise Analysis")

                st.dataframe(
                    pd.DataFrame(frame_predictions),
                    use_container_width=True
                )

                st.subheader("Detected Face Samples")

                display_columns = st.columns(
                    min(4, len(frame_results))
                )

                for index, item in enumerate(frame_results):

                    with display_columns[
                        index % len(display_columns)
                    ]:

                        st.image(
                            cv2.cvtColor(
                                item["face"],
                                cv2.COLOR_BGR2RGB
                            ),
                            caption=f"Frame {item['frame_index']}",
                            use_container_width=True
                        )

                st.caption(
                    "The final video prediction comes from "
                    "the video-level classifier. Frame-wise "
                    "predictions are supplementary image-level analyses."
                )

                st.caption(
                    "The fake-class score is a model-estimated score, "
                    "not a calibrated measure of forensic certainty."
                )

            except Exception as error:

                st.error(
                    f"Video analysis failed: {error}"
                )


# ==========================================
# 18. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    **DeepTrace AI**

    Research-based DeepFake detection using
    spatial and frequency-domain signal analysis.

    *Research prototype – predictions should not be
    treated as definitive proof of media authenticity.*
    """
)
