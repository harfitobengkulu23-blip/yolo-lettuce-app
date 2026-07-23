"""
APP.PY - Backend Flask untuk Deteksi Kondisi Tanaman Selada Hidroponik
=====================================================================

Fitur:
1. Deteksi melalui unggah gambar.
2. Deteksi melalui unggah video.
3. Deteksi real-time webcam.
4. Penyimpanan video hasil deteksi dalam format MP4 H.264 agar kompatibel
   dengan browser, Windows Media Player, PowerPoint, VLC, dan CapCut.

PENTING:
- Letakkan bobot model pada: models/best.pt
- Instal FFmpeg dan pastikan perintah `ffmpeg -version` dapat dijalankan
  melalui Command Prompt/Terminal.
"""

import base64
import io
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file
from ultralytics import YOLO
from werkzeug.utils import secure_filename


# ============================================================================
# KONFIGURASI APLIKASI
# ============================================================================

# Memuat variabel dari file .env saat pengembangan lokal.
# Pada Render/Railway, variabel lingkungan diatur melalui dashboard platform.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")
UPLOAD_FOLDER = BASE_DIR / "uploads"
TEMP_FOLDER = UPLOAD_FOLDER / "temp"

CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640

CLASS_NAMES = ["Chlorosis", "Healthy", "Rot"]

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)


# ============================================================================
# INISIALISASI FLASK DAN LOGGING
# ============================================================================

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# MEMUAT MODEL YOLO
# ============================================================================

model = None
model_load_error = None

try:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"File model tidak ditemukan di '{MODEL_PATH}'. "
            "Pastikan best.pt diletakkan di folder models."
        )

    model = YOLO(MODEL_PATH)
    logger.info("Model YOLO berhasil dimuat dari: %s", MODEL_PATH)
    logger.info("Nama kelas dari model: %s", model.names)

except Exception as exc:
    model_load_error = str(exc)
    logger.exception("Gagal memuat model YOLO: %s", model_load_error)


# ============================================================================
# FUNGSI BANTU UMUM
# ============================================================================

def allowed_file(filename: str) -> bool:
    """Memeriksa apakah ekstensi file gambar diizinkan."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def allowed_video_file(filename: str) -> bool:
    """Memeriksa apakah ekstensi file video diizinkan."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
    )


def create_unique_filename(original_filename: str, prefix: str = "") -> str:
    """
    Membuat nama file aman dan unik agar file lama tidak tertimpa.
    """
    safe_name = secure_filename(original_filename)

    if not safe_name:
        safe_name = "file"

    stem, extension = os.path.splitext(safe_name)
    unique_id = uuid.uuid4().hex[:10]

    return f"{prefix}{stem}_{unique_id}{extension.lower()}"


def get_class_name(class_id: int) -> str:
    """
    Mengambil nama kelas dari model.names, kemudian CLASS_NAMES sebagai fallback.
    """
    class_name = None

    if model is not None:
        if isinstance(model.names, dict):
            class_name = model.names.get(class_id)
        elif isinstance(model.names, list) and 0 <= class_id < len(model.names):
            class_name = model.names[class_id]

    if class_name is None and 0 <= class_id < len(CLASS_NAMES):
        class_name = CLASS_NAMES[class_id]

    if class_name is None:
        class_name = f"Class-{class_id}"

    return str(class_name)


def get_box_color(class_name: str) -> tuple[int, int, int]:
    """
    Mengembalikan warna bounding box dalam format BGR OpenCV.
    """
    name = class_name.lower().strip()

    if name == "chlorosis":
        return 0, 150, 150
    if name == "healthy":
        return 0, 180, 0
    if name in {"rot", "soft rot"}:
        return 0, 0, 170

    return 0, 180, 0


def get_box_color_hex(class_name: str) -> str:
    """Mengembalikan warna kelas dalam format HEX untuk frontend."""
    name = class_name.lower().strip()

    if name == "chlorosis":
        return "#969600"
    if name == "healthy":
        return "#00B400"
    if name in {"rot", "soft rot"}:
        return "#AA0000"

    return "#00B400"


def draw_label(
    image_bgr: np.ndarray,
    label: str,
    x1: int,
    y1: int,
    box_color: tuple[int, int, int],
) -> None:
    """Menggambar label kelas dan confidence di atas bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    text_color = (255, 255, 255)
    shadow_color = (0, 0, 0)

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        font_thickness,
    )

    padding_x = 6
    padding_y = 6

    label_x1 = max(x1, 0)
    label_y1 = y1 - text_height - baseline - (padding_y * 2)
    label_x2 = label_x1 + text_width + (padding_x * 2)
    label_y2 = y1

    if label_y1 < 0:
        label_y1 = y1
        label_y2 = y1 + text_height + baseline + (padding_y * 2)

    image_height, image_width = image_bgr.shape[:2]

    label_x2 = min(label_x2, image_width - 1)
    label_y1 = max(label_y1, 0)
    label_y2 = min(label_y2, image_height - 1)

    cv2.rectangle(
        image_bgr,
        (label_x1, label_y1),
        (label_x2, label_y2),
        box_color,
        -1,
    )

    text_x = label_x1 + padding_x
    text_y = label_y2 - padding_y - baseline

    # Bayangan hitam terlebih dahulu.
    cv2.putText(
        image_bgr,
        label,
        (text_x + 1, text_y + 1),
        font,
        font_scale,
        shadow_color,
        font_thickness + 1,
        cv2.LINE_AA,
    )

    # Teks putih di atas bayangan.
    cv2.putText(
        image_bgr,
        label,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )


# ============================================================================
# INFERENSI GAMBAR/FRAME
# ============================================================================

def run_inference(image_bgr: np.ndarray):
    """
    Menjalankan inferensi YOLO pada satu frame BGR.

    Return:
        annotated_image_bgr
        detections_list
    """
    if model is None:
        raise RuntimeError(model_load_error or "Model belum berhasil dimuat.")

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Frame/gambar input kosong.")

    results = model.predict(
        source=image_bgr,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )

    result = results[0]
    annotated_image_bgr = image_bgr.copy()
    detections_list = []

    if result.boxes is None:
        return annotated_image_bgr, detections_list

    image_height, image_width = annotated_image_bgr.shape[:2]

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        class_name = get_class_name(class_id)
        box_color = get_box_color(class_name)
        color_hex = get_box_color_hex(class_name)

        detections_list.append(
            {
                "class_name": class_name,
                "confidence": round(confidence * 100, 2),
                "box": [x1, y1, x2, y2],
                "color_hex": color_hex,
            }
        )

        label = f"{class_name} {confidence * 100:.1f}%"

        cv2.rectangle(
            annotated_image_bgr,
            (x1, y1),
            (x2, y2),
            box_color,
            3,
        )

        draw_label(
            image_bgr=annotated_image_bgr,
            label=label,
            x1=x1,
            y1=y1,
            box_color=box_color,
        )

    return annotated_image_bgr, detections_list


def encode_image_to_base64(image_bgr: np.ndarray) -> str:
    """Mengubah gambar OpenCV menjadi data URL JPEG base64."""
    success, buffer = cv2.imencode(".jpg", image_bgr)

    if not success:
        raise RuntimeError("Gagal mengencode gambar hasil deteksi.")

    base64_string = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_string}"


def decode_base64_to_image(base64_string: str) -> np.ndarray:
    """Mengubah data URL/base64 webcam menjadi gambar BGR OpenCV."""
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    image_bytes = base64.b64decode(base64_string)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_rgb = np.array(pil_image)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


# ============================================================================
# PEMROSESAN VIDEO
# ============================================================================

def get_valid_video_fps(cap: cv2.VideoCapture) -> float:
    """Mengambil FPS video dan memberikan fallback jika nilainya tidak valid."""
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if not np.isfinite(fps) or fps <= 0 or fps > 240:
        return 25.0

    return fps


def find_ffmpeg() -> str:
    """
    Mencari executable FFmpeg melalui PATH.

    Jalankan `ffmpeg -version` pada terminal untuk memastikan FFmpeg tersedia.
    """
    ffmpeg_path = shutil.which("ffmpeg")

    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg tidak ditemukan. Instal FFmpeg terlebih dahulu, kemudian "
            "tutup dan buka kembali terminal/VS Code. Pastikan perintah "
            "'ffmpeg -version' dapat dijalankan."
        )

    return ffmpeg_path


def convert_to_h264(
    temporary_video_path: str,
    output_path: str,
) -> None:
    """
    Mengonversi video sementara menjadi MP4 H.264 + yuv420p.
    Format ini paling kompatibel dengan browser, Windows, PowerPoint, dan CapCut.
    """
    ffmpeg_path = find_ffmpeg()

    output_file = Path(output_path)

    if output_file.exists():
        output_file.unlink()

    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        temporary_video_path,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if process.returncode != 0:
        logger.error("FFmpeg error:\n%s", process.stderr)
        raise RuntimeError(
            "FFmpeg gagal mengonversi video menjadi MP4 H.264. "
            f"Detail: {process.stderr.strip() or 'Tidak ada detail error.'}"
        )

    if not output_file.exists() or output_file.stat().st_size < 1024:
        raise RuntimeError(
            "File MP4 hasil konversi tidak terbentuk dengan benar."
        )


def process_video(
    input_path: str,
    output_path: str,
    num_samples: int = 4,
) -> dict:
    """
    Memproses video frame demi frame menggunakan YOLO.

    Tahap:
    1. Membaca video input.
    2. Menggambar bounding box pada setiap frame.
    3. Menulis video sementara MJPG/AVI.
    4. Mengonversi menjadi MP4 H.264 menggunakan FFmpeg.
    """
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError(
            "Gagal membuka file video. Pastikan file tidak rusak dan formatnya didukung."
        )

    fps = get_valid_video_fps(cap)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_width <= 0 or frame_height <= 0:
        cap.release()
        raise RuntimeError(
            "Resolusi video tidak valid atau video tidak dapat dibaca OpenCV."
        )

    # H.264/yuv420p membutuhkan dimensi genap.
    output_width = frame_width - (frame_width % 2)
    output_height = frame_height - (frame_height % 2)

    if output_width <= 0 or output_height <= 0:
        cap.release()
        raise RuntimeError("Resolusi output video tidak valid.")

    if total_frames > 0 and num_samples > 0:
        sample_positions = np.linspace(
            0,
            max(total_frames - 1, 0),
            num=min(num_samples, total_frames),
            dtype=int,
        )
        sample_indices = set(int(index) for index in sample_positions)
    else:
        sample_indices = set()

    temporary_video_path = str(
        TEMP_FOLDER / f"temp_{uuid.uuid4().hex}.avi"
    )

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(
        temporary_video_path,
        fourcc,
        fps,
        (output_width, output_height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            "VideoWriter gagal dibuka. Codec MJPG tidak tersedia pada OpenCV."
        )

    detection_summary = {}
    sample_frames = []
    frame_count = 0

    logger.info(
        "Memproses video input=%s | frame=%d | fps=%.2f | ukuran=%dx%d",
        input_path,
        total_frames,
        fps,
        output_width,
        output_height,
    )

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            annotated_frame, detections = run_inference(frame)

            if (
                annotated_frame.shape[1] != output_width
                or annotated_frame.shape[0] != output_height
            ):
                annotated_frame = cv2.resize(
                    annotated_frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA,
                )

            # Ringkasan ini menghitung jumlah frame yang memuat setiap kelas.
            detected_classes = {
                detection["class_name"]
                for detection in detections
            }

            for class_name in detected_classes:
                detection_summary[class_name] = (
                    detection_summary.get(class_name, 0) + 1
                )

            if frame_count in sample_indices:
                sample_frame = annotated_frame.copy()
                timestamp = frame_count / fps
                label_text = f"Frame {frame_count + 1} ({timestamp:.1f}s)"

                # Bayangan.
                cv2.putText(
                    sample_frame,
                    label_text,
                    (11, output_height - 11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )

                # Teks putih.
                cv2.putText(
                    sample_frame,
                    label_text,
                    (10, output_height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                sample_frames.append(
                    encode_image_to_base64(sample_frame)
                )

            writer.write(annotated_frame)
            frame_count += 1

            if frame_count % 50 == 0:
                logger.info(
                    "Progress video: %d/%d frame",
                    frame_count,
                    total_frames,
                )

    finally:
        cap.release()
        writer.release()

    try:
        if frame_count == 0:
            raise RuntimeError(
                "Tidak ada frame yang berhasil diproses dari video."
            )

        temporary_file = Path(temporary_video_path)

        if (
            not temporary_file.exists()
            or temporary_file.stat().st_size < 1024
        ):
            raise RuntimeError(
                "Video sementara gagal dibuat atau ukuran file tidak valid."
            )

        convert_to_h264(
            temporary_video_path=temporary_video_path,
            output_path=output_path,
        )

        logger.info(
            "Video hasil deteksi berhasil dibuat: %s (%d byte)",
            output_path,
            Path(output_path).stat().st_size,
        )

    finally:
        temporary_file = Path(temporary_video_path)

        if temporary_file.exists():
            try:
                temporary_file.unlink()
            except OSError:
                logger.warning(
                    "File sementara tidak dapat dihapus: %s",
                    temporary_video_path,
                )

    return {
        "total_frames": frame_count,
        "fps": round(fps, 2),
        "summary": detection_summary,
        "sample_frames": sample_frames,
    }


# ============================================================================
# ROUTES FLASK
# ============================================================================

@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(
        {
            "success": False,
            "error": "Ukuran file terlalu besar. Maksimum 200 MB.",
        }
    ), 413


@app.route("/")
def index():
    """Menampilkan halaman utama aplikasi."""
    return render_template(
        "index.html",
        model_ready=(model is not None),
        model_error=model_load_error,
        class_names=CLASS_NAMES,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )


@app.route("/predict/image", methods=["POST"])
def predict_image():
    """Menerima gambar dan mengembalikan hasil deteksi dalam base64."""
    if model is None:
        return jsonify(
            {
                "success": False,
                "error": f"Model tidak tersedia: {model_load_error}",
            }
        ), 500

    if "image" not in request.files:
        return jsonify(
            {
                "success": False,
                "error": "Tidak ada file gambar yang dikirim.",
            }
        ), 400

    uploaded_file = request.files["image"]

    if uploaded_file.filename == "":
        return jsonify(
            {
                "success": False,
                "error": "Nama file gambar kosong.",
            }
        ), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify(
            {
                "success": False,
                "error": "Format gambar tidak didukung. Gunakan JPG, JPEG, atau PNG.",
            }
        ), 400

    input_path = None

    try:
        input_filename = create_unique_filename(uploaded_file.filename)
        input_path = UPLOAD_FOLDER / input_filename

        uploaded_file.save(str(input_path))

        image_bgr = cv2.imread(str(input_path))

        if image_bgr is None:
            raise ValueError("Gagal membaca file gambar.")

        annotated_image_bgr, detections_list = run_inference(image_bgr)
        result_base64 = encode_image_to_base64(annotated_image_bgr)

        return jsonify(
            {
                "success": True,
                "result_image": result_base64,
                "detections": detections_list,
                "total_detections": len(detections_list),
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "iou_threshold": IOU_THRESHOLD,
            }
        )

    except Exception as exc:
        logger.exception("Error saat memproses gambar")
        return jsonify(
            {
                "success": False,
                "error": f"Terjadi kesalahan: {exc}",
            }
        ), 500

    finally:
        if input_path and input_path.exists():
            try:
                input_path.unlink()
            except OSError:
                logger.warning("Gagal menghapus gambar input: %s", input_path)


@app.route("/predict/frame", methods=["POST"])
def predict_frame():
    """Menerima satu frame webcam dalam base64."""
    if model is None:
        return jsonify(
            {
                "success": False,
                "error": f"Model tidak tersedia: {model_load_error}",
            }
        ), 500

    data = request.get_json(silent=True)

    if not data or "frame" not in data:
        return jsonify(
            {
                "success": False,
                "error": "Data frame tidak ditemukan.",
            }
        ), 400

    try:
        image_bgr = decode_base64_to_image(data["frame"])
        _, detections_list = run_inference(image_bgr)

        return jsonify(
            {
                "success": True,
                "detections": detections_list,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "iou_threshold": IOU_THRESHOLD,
            }
        )

    except Exception as exc:
        logger.exception("Error saat memproses frame webcam")
        return jsonify(
            {
                "success": False,
                "error": f"Terjadi kesalahan: {exc}",
            }
        ), 500


@app.route("/predict/video", methods=["POST"])
def predict_video():
    """
    Menerima video, menjalankan inferensi, dan menghasilkan MP4 H.264.
    """
    if model is None:
        return jsonify(
            {
                "success": False,
                "error": f"Model tidak tersedia: {model_load_error}",
            }
        ), 500

    if "video" not in request.files:
        return jsonify(
            {
                "success": False,
                "error": "Tidak ada file video yang dikirim.",
            }
        ), 400

    uploaded_file = request.files["video"]

    if uploaded_file.filename == "":
        return jsonify(
            {
                "success": False,
                "error": "Nama file video kosong.",
            }
        ), 400

    if not allowed_video_file(uploaded_file.filename):
        return jsonify(
            {
                "success": False,
                "error": (
                    "Format video tidak didukung. "
                    "Gunakan MP4, AVI, MOV, MKV, atau WEBM."
                ),
            }
        ), 400

    input_path = None

    try:
        input_filename = create_unique_filename(uploaded_file.filename)
        input_path = UPLOAD_FOLDER / input_filename

        original_stem = secure_filename(
            Path(uploaded_file.filename).stem
        ) or "video"

        output_filename = (
            f"result_{original_stem}_{uuid.uuid4().hex[:10]}.mp4"
        )
        output_path = UPLOAD_FOLDER / output_filename

        uploaded_file.save(str(input_path))
        logger.info("Video diterima: %s", input_path)

        stats = process_video(
            input_path=str(input_path),
            output_path=str(output_path),
        )

        return jsonify(
            {
                "success": True,
                "output_filename": output_filename,
                "video_url": f"/download/video/{output_filename}",
                "download_url": f"/download/video/{output_filename}?download=1",
                "total_frames": stats["total_frames"],
                "fps": stats["fps"],
                "summary": stats["summary"],
                "sample_frames": stats["sample_frames"],
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "iou_threshold": IOU_THRESHOLD,
            }
        )

    except Exception as exc:
        logger.exception("Error saat memproses video upload")
        return jsonify(
            {
                "success": False,
                "error": f"Terjadi kesalahan: {exc}",
            }
        ), 500

    finally:
        if input_path and input_path.exists():
            try:
                input_path.unlink()
            except OSError:
                logger.warning("Gagal menghapus video input: %s", input_path)


@app.route("/download/video/<filename>")
def download_video(filename):
    """
    Menampilkan atau mengunduh video hasil deteksi.

    - /download/video/nama.mp4
      Dibuka secara inline untuk preview browser.

    - /download/video/nama.mp4?download=1
      Diunduh sebagai attachment.
    """
    safe_filename = secure_filename(filename)
    file_path = UPLOAD_FOLDER / safe_filename

    if not file_path.is_file():
        return jsonify(
            {
                "success": False,
                "error": "File video tidak ditemukan.",
            }
        ), 404

    if file_path.stat().st_size < 1024:
        return jsonify(
            {
                "success": False,
                "error": "File video tidak valid atau proses encoding gagal.",
            }
        ), 500

    force_download = request.args.get("download") == "1"

    return send_file(
        str(file_path),
        mimetype="video/mp4",
        as_attachment=force_download,
        download_name=safe_filename if force_download else None,
        conditional=True,
        max_age=0,
    )


@app.route("/health")
def health_check():
    """Memeriksa status aplikasi, model, dan FFmpeg."""
    ffmpeg_path = shutil.which("ffmpeg")

    return jsonify(
        {
            "success": True,
            "model_ready": model is not None,
            "model_error": model_load_error,
            "model_path": str(MODEL_PATH),
            "class_names_config": CLASS_NAMES,
            "model_names": model.names if model is not None else None,
            "ffmpeg_ready": ffmpeg_path is not None,
            "ffmpeg_path": ffmpeg_path,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "image_size": IMAGE_SIZE,
            "colors": {
                "Chlorosis": {
                    "bgr": get_box_color("Chlorosis"),
                    "hex": get_box_color_hex("Chlorosis"),
                },
                "Healthy": {
                    "bgr": get_box_color("Healthy"),
                    "hex": get_box_color_hex("Healthy"),
                },
                "Rot": {
                    "bgr": get_box_color("Rot"),
                    "hex": get_box_color_hex("Rot"),
                },
            },
        }
    )


# ============================================================================
# MENJALANKAN APLIKASI
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug_mode,
    )
