from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timezone
import os
import shutil
import subprocess
import time
import traceback
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

# --- 1. Dynamic Windows PATH Repair for FFmpeg & Winget ---
possible_ffmpeg_paths = [
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links"),
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
]
for p in possible_ffmpeg_paths:
    if os.path.exists(p) and p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

FFMPEG_AVAILABLE = (
    shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
)
if FFMPEG_AVAILABLE:
    print("[SYSTEM] FFmpeg and FFprobe successfully discovered on system PATH.")
else:
    print(
        "[SYSTEM WARNING] FFmpeg/FFprobe CLI not in system PATH. Audio"
        " normalization will use native fallbacks."
    )

# Optimize CPU multithreading for PyTorch backbones
try:
    import torch

    torch.set_num_threads(min(4, os.cpu_count() or 4))
except ImportError:
    pass

import cv2
from evidenceapp.core.hashing import get_chain_of_custody
from evidenceapp.core.metadata_parser import parse_image_metadata
from evidenceapp.core.pdf_generator import generate_evidence_pdf
from evidenceapp.models.audio_forensics import AudioForensicsDetector
from evidenceapp.models.fusion_engine import FusionEngine
from evidenceapp.models.gemini_explainer import GeminiForensicsExplainer
from evidenceapp.models.image_forensics import ImageForensicsDetector
from evidenceapp.models.video_forensics import VideoForensicsDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "evidenceapp", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "evidenceapp", "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "evidenceapp", "uploads")
REPORT_DIR = os.path.join(BASE_DIR, "evidenceapp", "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

app = Flask(
    __name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR
)

CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["REPORT_FOLDER"] = REPORT_DIR
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB Limit

# Extension Sets for Strict Routing (Including MPEG formats)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".mpeg",
    ".mpg",
    ".m4v",
}
AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".ogg",
    ".flac",
    ".m4a",
    ".aac",
    ".mp2",
    ".opus",
}

# Initialize Forensic Detectors and Explainer
print("[SYSTEM] Initializing Forensic Pipelines & Multi-Modal Backbones...")
image_detector = ImageForensicsDetector()
video_detector = VideoForensicsDetector()

potential_aasist_paths = [
    os.path.join(MODELS_DIR, "AASIST.pth"),
    os.path.join(BASE_DIR, "weights", "AASIST.pth"),
    os.path.join(BASE_DIR, "AASIST.pth"),
]
aasist_weights = next(
    (p for p in potential_aasist_paths if os.path.isfile(p)), None
)

if aasist_weights:
    print(f"[SYSTEM] AASIST pretrained weights found at: {aasist_weights}")
else:
    print(
        "[SYSTEM] No AASIST weights file detected. Running with baseline filter"
        " parameters."
    )

audio_detector = AudioForensicsDetector(aasist_weights_path=aasist_weights)
gemini_explainer = GeminiForensicsExplainer()

EVIDENCE_REGISTRY = []
DOSSIER_STORE = {}


def has_active_video_stream(file_path: str) -> bool:
    """Uses ffprobe, falling back to cv2, to verify whether dynamic visual frames exist."""
    if FFMPEG_AVAILABLE:
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path,
        ]
        try:
            output = (
                subprocess.check_output(
                    probe_cmd, stderr=subprocess.DEVNULL, timeout=5
                )
                .decode()
                .strip()
            )
            if output and output.lower() not in ["png", "mjpeg", "jpeg"]:
                return True
            return False
        except Exception:
            pass

    # Resilient fallback using OpenCV
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return total_frames > 1 and width > 0 and height > 0
    except Exception:
        return False


def _convert_to_clean_wav(input_path: str) -> str:
    """Converts container audio (mp4, mpeg, m4a, aac) to a clean 16kHz mono WAV to avoid audioread hangs."""
    target_wav = os.path.join(
        app.config["UPLOAD_FOLDER"],
        f"temp_16k_{os.path.splitext(os.path.basename(input_path))[0]}.wav",
    )
    if FFMPEG_AVAILABLE:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            target_wav,
        ]
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=15,
            )
            if os.path.exists(target_wav) and os.path.getsize(target_wav) > 1024:
                return target_wav
        except Exception as e:
            print(f"[Warning] Fast WAV normalization failed: {e}")
    return input_path


def _extract_and_analyze_embedded_audio(
    video_path: str, base_filename: str
) -> dict | None:
    """Demuxes audio track and analyzes it via AASIST within a strict timeout."""
    if not FFMPEG_AVAILABLE:
        return None

    temp_audio_name = f"extracted_audio_{os.path.splitext(base_filename)[0]}.wav"
    temp_audio_path = os.path.join(app.config["UPLOAD_FOLDER"], temp_audio_name)

    extract_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-t",
        "6",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        temp_audio_path,
    ]

    try:
        subprocess.run(
            extract_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=8,
        )
        if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 1024:
            audio_data = audio_detector.analyze_audio(temp_audio_path)
            try:
                os.remove(temp_audio_path)
            except OSError:
                pass
            return audio_data
    except Exception as demux_err:
        print(f"[Warning] Embedded audio extraction failed or bypassed: {demux_err}")

    if os.path.exists(temp_audio_path):
        try:
            os.remove(temp_audio_path)
        except OSError:
            pass
    return None


def process_uploaded_artifact(uploaded_file):
    """Core pipeline to analyze an artifact, seal custody, and generate dossier."""
    start_time = time.time()
    raw_filename = uploaded_file.filename
    clean_filename = secure_filename(raw_filename)

    if not clean_filename:
        ext = os.path.splitext(raw_filename)[1].lower() or ".bin"
        clean_filename = (
            f"artifact_{int(datetime.now(timezone.utc).timestamp())}{ext}"
        )

    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], clean_filename)
    uploaded_file.save(saved_path)

    print(
        f"\n[PIPELINE] Ingested: {clean_filename} ({os.path.getsize(saved_path)}"
        " bytes)"
    )

    # 1. Cryptographic Chain of Custody (SHA-256)
    print("[PIPELINE] Computing SHA-256 Chain of Custody...")
    custody = get_chain_of_custody(saved_path)

    # 2. Strict media routing based on file extension and stream layout
    ext = os.path.splitext(clean_filename)[1].lower()

    if ext in IMAGE_EXTENSIONS:
        media_type = "image"
        print("[PIPELINE] Routing to Branch-A: Image Forensics (ELA / TruFor / SAM / ManTra-Net / ViT)...")
        forensic_data = image_detector.analyze_evidence(saved_path)
        metadata = parse_image_metadata(saved_path)

    elif ext in VIDEO_EXTENSIONS:
        is_real_video = has_active_video_stream(saved_path)

        if not is_real_video:
            print(
                f"[PIPELINE] Container ({ext}) has no active video track. Rerouting to"
                " Branch-C Audio Forensics..."
            )
            media_type = "audio"
            clean_wav_path = _convert_to_clean_wav(saved_path)
            print("[PIPELINE] Executing AASIST & Vocoder acoustic analysis...")
            forensic_data = audio_detector.analyze_audio(clean_wav_path)

            if clean_wav_path != saved_path and os.path.exists(clean_wav_path):
                try:
                    os.remove(clean_wav_path)
                except OSError:
                    pass

            metadata = {
                "has_exif": False,
                "device_make": "Audio Stream (Container Transcode)",
                "device_model": f"{forensic_data.get('duration_seconds', 0.0)}s Ingest",
                "software": "AASIST GNN / HuBERT Speech / Spectral Vocoder Engine",
                "is_suspicious": (
                    forensic_data.get("ai_voice_clone_confidence", 0.0) >= 0.50
                    or forensic_data.get("ai_generation_confidence", 0.0) >= 0.50
                ),
                "finding": forensic_data.get(
                    "interpretation", "Audio forensics complete."
                ),
            }
        else:
            media_type = "video"
            print(
                "[PIPELINE] Routing to Branch-B: Parallel Temporal Video & Audio"
                " Forensics..."
            )

            with ThreadPoolExecutor(max_workers=2) as executor:
                video_future = executor.submit(video_detector.analyze_video, saved_path)
                audio_future = executor.submit(
                    _extract_and_analyze_embedded_audio, saved_path, clean_filename
                )
                forensic_data = video_future.result()
                audio_data = audio_future.result()

            has_embedded_audio = audio_data is not None
            if has_embedded_audio:
                forensic_data["embedded_audio_analysis"] = audio_data

                if "models_used" in audio_data:
                    for model in audio_data["models_used"]:
                        if model not in forensic_data.get("models_used", []):
                            forensic_data.setdefault("models_used", []).append(model)

                audio_ai = audio_data.get(
                    "ai_generation_confidence",
                    audio_data.get("ai_voice_clone_confidence", 0.0),
                )
                audio_manip = audio_data.get("manipulation_confidence", 0.0)

                if audio_ai > 0.55 or audio_manip > 0.55:
                    threat_boost = max(audio_ai, audio_manip)
                    forensic_data["ai_generation_confidence"] = max(
                        forensic_data.get("ai_generation_confidence", 0.0), threat_boost
                    )
                    forensic_data["ai_generation_percentage"] = (
                        f"{round(forensic_data['ai_generation_confidence'] * 100, 1)}%"
                    )

                    if (
                        forensic_data.get("verdict")
                        == "Authentic Temporal Footage"
                    ):
                        forensic_data["verdict"] = "Synthetic Audio / Visual Disparity"
                        forensic_data["interpretation"] = (
                            "Visual frames coherent, but embedded acoustic track exhibits"
                            " synthetic vocoder artifacts."
                        )

            vid_ai = forensic_data.get("ai_generation_confidence", 0.0)
            vid_manip = forensic_data.get("manipulation_confidence", 0.0)
            f3net_risk = forensic_data.get("f3net_face_manipulation_score", 0.0)
            is_suspicious = (
                vid_ai >= 0.55 or vid_manip >= 0.42 or f3net_risk >= 0.65
            )

            metadata = {
                "has_exif": False,
                "device_make": forensic_data.get("ffmpeg_metadata", {}).get(
                    "encoder", "N/A"
                ),
                "device_model": "Dual-Stream Video/Audio",
                "software": (
                    "OpenCV / YOLO11 / TruFor / ViT / F3-Net / FFmpeg Engine"
                ),
                "is_suspicious": is_suspicious,
                "finding": (
                    "Video frames, F3-Net frequency spectra, and audio track"
                    " evaluated in parallel."
                    if has_embedded_audio
                    else "Video stream and F3-Net spectra analyzed (No active audio)."
                ),
            }

    elif ext in AUDIO_EXTENSIONS:
        media_type = "audio"
        print(
            f"[PIPELINE] Routing to Branch-C: Audio Forensics on {clean_filename}..."
        )

        clean_wav_path = saved_path
        if ext != ".wav" and FFMPEG_AVAILABLE:
            clean_wav_path = _convert_to_clean_wav(saved_path)

        print("[PIPELINE] Executing AASIST & HuBERT feature extraction...")
        forensic_data = audio_detector.analyze_audio(clean_wav_path)
        print("[PIPELINE] Audio forensic inference completed.")

        if clean_wav_path != saved_path and os.path.exists(clean_wav_path):
            try:
                os.remove(clean_wav_path)
            except OSError:
                pass

        metadata = {
            "has_exif": False,
            "device_make": "Acoustic Transducer / Raw Signal",
            "device_model": f"{forensic_data.get('duration_seconds', 0.0)}s Ingest",
            "software": "AASIST GNN / HuBERT Speech / Spectral Vocoder Engine",
            "is_suspicious": (
                forensic_data.get("ai_voice_clone_confidence", 0.0) >= 0.50
                or forensic_data.get("ai_generation_confidence", 0.0) >= 0.50
            ),
            "finding": forensic_data.get(
                "interpretation", "Acoustic forensics complete."
            ),
        }

    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. Upload audio (.wav, .mp3, .mpeg),"
            " video (.mp4, .mpeg), or images (.jpg, .png)."
        )

    # 3. Multimodal Fusion & Dossier Assembly
    print("[PIPELINE] Fusing pipeline evidence...")
    dossier = FusionEngine.fuse_analysis_pipeline(
        file_path=saved_path,
        forensic_results=forensic_data,
        metadata_results=metadata,
        custody_info=custody,
    )
    dossier["filename"] = clean_filename
    dossier["media_type"] = media_type

    # --- CRITICAL AUDIO OVERRIDE & SYNCHRONIZATION ---
    if media_type == "audio":
        ai_risk = float(
            forensic_data.get("ai_generation_confidence")
            or forensic_data.get("ai_voice_clone_confidence")
            or 0.0
        )
        manip_risk = float(forensic_data.get("manipulation_confidence", 0.0))

        real_authenticity = forensic_data.get(
            "authenticity",
            f"{max(2.5, round((1.0 - max(ai_risk, manip_risk)) * 100, 1))}%",
        )
        dossier["authenticity"] = real_authenticity
        dossier["authenticity_percentage"] = real_authenticity

        real_ai_pct = forensic_data.get(
            "ai_generation_percentage", f"{round(ai_risk * 100, 1)}%"
        )
        dossier["ai_generation_percentage"] = real_ai_pct
        dossier["ai_generation_confidence"] = ai_risk

        real_manip_pct = forensic_data.get(
            "manipulation_percentage", f"{round(manip_risk * 100, 1)}%"
        )
        dossier["manipulation_percentage"] = real_manip_pct
        dossier["manipulation_confidence"] = manip_risk

        if "metrics" in dossier and isinstance(dossier["metrics"], dict):
            dossier["metrics"]["authenticity"] = real_authenticity
            dossier["metrics"]["ai_generation"] = real_ai_pct
            dossier["metrics"]["manipulation"] = real_manip_pct

        if ai_risk >= 0.50:
            dossier["verdict"] = forensic_data.get(
                "verdict", "Synthetic / AI Voice Clone"
            )
            resolved_threat = "HIGH"
        elif manip_risk >= 0.45:
            dossier["verdict"] = "Manipulated / Spliced Audio Track"
            resolved_threat = "HIGH"
        elif ai_risk >= 0.35 or manip_risk >= 0.25:
            dossier["verdict"] = forensic_data.get(
                "verdict", "Suspicious Acoustic Artifacts Detected"
            )
            resolved_threat = "MEDIUM"
        else:
            dossier["verdict"] = forensic_data.get(
                "verdict", "Authentic Acoustic Capture"
            )
            resolved_threat = "LOW"
    else:
        ai_risk = float(forensic_data.get("ai_generation_confidence", 0.0))
        manip_risk = float(forensic_data.get("manipulation_confidence", 0.0))
        f3_score = float(forensic_data.get("f3net_face_manipulation_score", 0.0))

        if ai_risk >= 0.60 or manip_risk >= 0.48 or f3_score >= 0.70:
            resolved_threat = "HIGH"
        elif ai_risk >= 0.35 or manip_risk >= 0.28 or f3_score >= 0.40:
            resolved_threat = "MEDIUM"
        else:
            resolved_threat = "LOW"

    if "assessment" not in dossier:
        dossier["assessment"] = {}
    dossier["assessment"]["risk_level"] = resolved_threat

    # 4. Multimodal Explainer with extended 35-second execution limit
    print("[PIPELINE] Generating AI explanation...")
    try:
        with ThreadPoolExecutor(max_workers=1) as explainer_pool:
            future = explainer_pool.submit(
                gemini_explainer.explain_artifact,
                file_path=saved_path,
                media_type=media_type,
                forensic_stats=forensic_data,
            )
            gemini_result = future.result(timeout=35)
            dossier["gemini_explanation"] = gemini_result.get("explanation_text", "")
            dossier["gemini_status"] = gemini_result.get("status", "READY")
    except TimeoutError:
        print("[Warning] Explainer call timed out (35s limit reached). Bypassing.")
        dossier["gemini_explanation"] = (
            "Acoustic/Media forensic metrics verified. Explainer timed out."
        )
        dossier["gemini_status"] = "TIMEOUT_BYPASS"
    except Exception as gemini_err:
        print(f"[Warning] Explainer bypassed: {gemini_err}")
        dossier["gemini_explanation"] = (
            "Acoustic/Media forensic metrics verified. Automated summary bypassed."
        )
        dossier["gemini_status"] = "BYPASS"

    # Static URLs & Playback
    dossier["display_image_url"] = (
        f"/media/{clean_filename}" if media_type != "audio" else None
    )
    dossier["audio_playback_url"] = (
        f"/media/{clean_filename}" if media_type == "audio" else None
    )

    # Error Level Analysis / Heatmap Localization (TruFor / ELA)
    active_mask_path = dossier.get("ela_mask_path") or forensic_data.get(
        "ela_mask_path"
    )
    if active_mask_path and os.path.exists(active_mask_path):
        dossier["display_ela_url"] = f"/reports/{os.path.basename(active_mask_path)}"
    else:
        dossier["display_ela_url"] = None

    # ManTra-Net Heatmap Localization Support
    mantra_path = (
        dossier.get("mantranet_map_path")
        or forensic_data.get("mantranet_map_path")
        or forensic_data.get("manipulation_details", {}).get("mantranet_map_path")
    )
    if mantra_path and os.path.exists(mantra_path):
        dossier["display_mantranet_url"] = f"/reports/{os.path.basename(mantra_path)}"
        dossier["mantranet_map_path"] = mantra_path
    else:
        dossier["display_mantranet_url"] = None

    # Promptable Segmentation (SAM) Mask Support
    sam_mask_path = (
        dossier.get("sam_mask_path")
        or forensic_data.get("sam_mask_path")
        or forensic_data.get("manipulation_details", {}).get("sam_mask_path")
    )
    if sam_mask_path and os.path.exists(sam_mask_path):
        dossier["display_sam_url"] = f"/reports/{os.path.basename(sam_mask_path)}"
        dossier["sam_mask_path"] = sam_mask_path
    else:
        dossier["display_sam_url"] = None

    # 5. Generate PDF Audit Certificate
    pdf_name = f"Audit_{os.path.splitext(clean_filename)[0]}.pdf"
    pdf_path = os.path.join(app.config["REPORT_FOLDER"], pdf_name)
    try:
        generate_evidence_pdf(dossier, pdf_path)
    except Exception as pdf_err:
        print(f"[Warning] PDF generation deferred: {pdf_err}")

    dossier["pdf_name"] = pdf_name
    dossier["pdf_download_url"] = f"/download/{pdf_name}"

    # 6. Save in Registry
    case_id = dossier.get(
        "case_id", f"NCFU-{int(datetime.now(timezone.utc).timestamp())}"
    )
    dossier["case_id"] = case_id
    DOSSIER_STORE[case_id] = dossier

    registry_entry = {
        "case_id": case_id,
        "filename": clean_filename,
        "threat_level": resolved_threat,
        "classification": dossier.get("verdict", "Authentic Capture"),
        "authenticity": dossier.get("authenticity", "100%"),
        "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
    }
    EVIDENCE_REGISTRY.insert(0, registry_entry)

    elapsed_time = round(time.time() - start_time, 2)
    print(
        f"[PIPELINE] Analysis completed in {elapsed_time}s. Threat Level:"
        f" {resolved_threat}"
    )
    return dossier, pdf_name, case_id


# ==========================================
# PAGE ROUTES
# ==========================================


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            print("[ERROR] No 'file' key found in request.files")
            return redirect(request.url)

        uploaded_file = request.files["file"]
        if not uploaded_file or uploaded_file.filename == "":
            print("[ERROR] Empty filename submitted")
            return redirect(request.url)

        try:
            dossier, pdf_name, case_id = process_uploaded_artifact(uploaded_file)
            return render_template("report.html", report=dossier, pdf_name=pdf_name)
        except ValueError as val_err:
            return render_template("index.html", error_message=str(val_err))
        except Exception as err:
            traceback.print_exc()
            return render_template(
                "index.html", error_message=f"Forensic intake failed: {str(err)}"
            )

    return render_template("index.html")


@app.route("/dashboard", methods=["GET"])
def dashboard():
    total_files = len(EVIDENCE_REGISTRY)
    high_threats = sum(
        1 for e in EVIDENCE_REGISTRY if e["threat_level"] == "HIGH"
    )
    medium_threats = sum(
        1 for e in EVIDENCE_REGISTRY if e["threat_level"] == "MEDIUM"
    )
    low_threats = sum(1 for e in EVIDENCE_REGISTRY if e["threat_level"] == "LOW")

    pass_ratio = (
        round((low_threats / total_files * 100), 1) if total_files > 0 else 100.0
    )

    stats = {
        "total_cases": total_files,
        "high_threats": high_threats,
        "medium_threats": medium_threats,
        "low_threats": low_threats,
        "pass_ratio": f"{pass_ratio}%",
    }

    return render_template(
        "dashboard.html", stats=stats, cases=EVIDENCE_REGISTRY[:15]
    )


@app.route("/report/<case_id>", methods=["GET"])
def view_report(case_id):
    dossier = DOSSIER_STORE.get(case_id)
    if not dossier:
        return redirect(url_for("index"))
    return render_template(
        "report.html", report=dossier, pdf_name=dossier.get("pdf_name")
    )


# ==========================================
# ASYNC REST APIS
# ==========================================


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        dossier, pdf_name, case_id = process_uploaded_artifact(uploaded_file)
        return (
            jsonify({
                "success": True,
                "case_id": case_id,
                "redirect_url": f"/report/{case_id}",
                "dossier": dossier,
                "pdf_name": pdf_name,
            }),
            200,
        )
    except ValueError as val_err:
        return jsonify({"error": str(val_err)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/segment_sam", methods=["POST"])
def api_segment_sam():
    """
    Interactive prompt-based segmentation endpoint using SAM.
    Accepts JSON payload:
    {
        "filename": "target_image.jpg",
        "points": [[x, y], ...],      # Optional prompt points
        "boxes": [[x1, y1, x2, y2]]   # Optional prompt bounding boxes
    }
    """
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "Filename required for SAM segmentation."}), 400

    clean_name = secure_filename(filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], clean_name)
    if not os.path.exists(file_path):
        return jsonify({"error": f"Image '{clean_name}' not found."}), 404

    points = data.get("points")
    boxes = data.get("boxes")

    try:
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            return jsonify({"error": "Unable to read image."}), 400
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = image_detector.promptable_segmentor.segment_with_prompts(
            img_rgb, input_points=points, input_boxes=boxes
        )

        base_name = os.path.splitext(clean_name)[0]
        mask_filename = f"{base_name}_interactive_sam.png"
        mask_path = os.path.join(app.config["REPORT_FOLDER"], mask_filename)
        cv2.imwrite(mask_path, mask)

        return jsonify({
            "success": True,
            "mask_url": f"/reports/{mask_filename}",
            "filename": mask_filename
        }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"SAM interactive segmentation failed: {str(e)}"}), 500


# ==========================================
# STATIC ASSET & REPORT STREAMING
# ==========================================


@app.route("/media/<filename>", methods=["GET"])
def serve_uploaded_file(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    return jsonify({"error": "File not found"}), 404


@app.route("/reports/<filename>", methods=["GET"])
def serve_report_file(filename):
    file_path = os.path.join(app.config["REPORT_FOLDER"], filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    return jsonify({"error": "Report file not found"}), 404


@app.route("/download/<pdf_name>", methods=["GET"])
def download(pdf_name):
    pdf_path = os.path.join(app.config["REPORT_FOLDER"], pdf_name)
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True)
    return jsonify({"error": "PDF not found"}), 404


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,
        threaded=True,
    )