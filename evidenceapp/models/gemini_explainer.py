import mimetypes
import os
import time
from google import genai
from google.genai import types


class GeminiForensicsExplainer:

  def __init__(self):
    # Reads from GEMINI_API_KEY environment variable if present, else fallback
    self.api_key = os.environ.get(
        "GEMINI_API_KEY",
        "",
    ).strip()

    self.client = None
    if self.api_key:
      try:
        self.client = genai.Client(api_key=self.api_key)
        print("[SYSTEM] Gemini Forensics Explainer initialized successfully.")
      except Exception as e:
        print(f"[Warning] Failed to initialize Gemini Client: {e}")
        self.client = None
    else:
      print(
          "[Warning] GEMINI_API_KEY not configured. Explainer is disabled."
      )

  def _resolve_mime_type(self, file_path: str, media_type: str) -> str:
    """Determines the exact standard MIME type for media uploads."""
    ext = os.path.splitext(file_path)[1].lower()
    ext_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".mpeg": "video/mpeg",
        ".mpg": "video/mpeg",
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/m4a",
        ".aac": "audio/aac",
    }
    if ext in ext_map:
      return ext_map[ext]

    mime, _ = mimetypes.guess_type(file_path)
    if mime:
      return mime

    defaults = {
        "video": "video/mp4",
        "audio": "audio/wav",
        "image": "image/jpeg",
    }
    return defaults.get(media_type, "application/octet-stream")

  def explain_artifact(
      self, file_path: str, media_type: str, forensic_stats: dict
  ) -> dict:
    """Multimodal pipeline:

    Ingests image, video, or audio artifacts, merges computational forensic
    metrics,
    and generates a plain-language root-cause analysis.
    """
    if not self.client:
      return {
          "status": "DISABLED",
          "explanation_text": (
              "[GEMINI TRACE DISABLED] GEMINI_API_KEY is not configured or"
              " client failed.\nConfigure your key to enable automated"
              " forensic explainers."
          ),
      }

    if not os.path.exists(file_path):
      return {
          "status": "ERROR",
          "explanation_text": f"Artifact file not located at: {file_path}",
      }

    uploaded_file = None
    mime_type = self._resolve_mime_type(file_path, media_type)
    file_size = os.path.getsize(file_path)

    # 1. Media anomaly prompt guidelines
    if media_type == "video":
      has_embedded_audio = "embedded_audio_analysis" in forensic_stats
      audio_note = (
          "Embedded audio was demuxed and analyzed alongside video frames."
          if has_embedded_audio
          else "Pure visual stream (No embedded audio track detected)."
      )
      anomaly_guidance = (
          f"- Audio/Visual Context: {audio_note}\n"
          "- Temporal Coherence: Are objects/backgrounds rigidly moving, or are"
          " there melting/warping/boiling artifacts?\n"
          "- Facial Dynamics: Notice any synthetic blinking, unnatural"
          " micro-jitter, or seam boundaries around faces.\n"
          "- Sensor Grain vs Diffusion: Check whether high-frequency camera"
          " sensor noise exists or if textures are over-smoothed."
      )
    elif media_type == "audio":
      anomaly_guidance = (
          "- Acoustic Floor: Listen for synthetic vocoder robotic buzzing,"
          " sudden cutoffs, or absent room reverberation.\n"
          "- Speech Naturalness: Are breathing patterns, pauses, and phoneme"
          " transitions authentic or synthetic TTS/voice-clones?"
      )
    else:  # image
      anomaly_guidance = (
          "- Pixel & Edge Coherence: Are there splicing seams, ELA noise"
          " mismatches, or warped background lines?\n"
          "- Generative Artifacts: Check hands, teeth, asymmetric eyes, text"
          " gibberish, and lighting direction inconsistency."
      )

    prompt = f"""
You are an expert digital forensics examiner.
We ran an automated signal inspection pipeline on this uploaded {media_type.upper()} file.

Here are the computational metrics from our forensic models:
- Verdict: {forensic_stats.get('verdict', 'Under Review')}
- Authenticity Score: {forensic_stats.get('authenticity_percentage', forensic_stats.get('authenticity', 'N/A'))}
- AI-Generation Probability: {forensic_stats.get('ai_generation_percentage', 'N/A')}
- Tampering / Splicing Probability: {forensic_stats.get('manipulation_percentage', 'N/A')}
- Forensic Models Used: {', '.join(forensic_stats.get('models_used', ['Signal Diagnostics']))}

Perform a rigorous multi-modal analysis of the provided media:
{anomaly_guidance}

Format your final report strictly in the following three sections:
[01 // SCENE & CONTENT BREAKDOWN]
State precisely what is visible or audible in this media (subjects, actions, background details).

[02 // FORENSIC SIGNAL VERIFICATION]
Explain the specific physical evidence in the file. Explain why the computed scores match (or contradict) the physical reality of the media (e.g. lighting, acoustics, sensor noise, edge artifacts).

[03 // FINAL LEGAL & FORENSIC CONCLUSION]
Provide a definitive summary explaining why this media is an Authentic Capture, an AI-Generated Synthesis, or a Manipulated Deepfake.
"""

    try:
      content_payload = None

      # 2. In-Memory Direct Byte Ingestion for files < 15MB (Drastically faster, avoids polling delays)
      if file_size < 15 * 1024 * 1024:
        try:
          with open(file_path, "rb") as f:
            file_bytes = f.read()
          content_payload = [
              types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
              prompt,
          ]
        except Exception as byte_err:
          print(
              f"[Notice] Direct byte ingestion failed ({byte_err}), falling"
              " back to Files API..."
          )

      # Fallback to Files API for larger payloads
      if content_payload is None:
        uploaded_file = self.client.files.upload(
            file=file_path, config=types.UploadFileConfig(mime_type=mime_type)
        )

        if media_type in ["video", "audio"]:
          max_retries = 15
          while max_retries > 0:
            status = getattr(
                uploaded_file.state, "name", str(uploaded_file.state)
            )
            if status == "ACTIVE":
              break
            elif status in ["FAILED", "ERROR"]:
              raise ValueError("Gemini file processing reported failure.")
            time.sleep(1.5)
            uploaded_file = self.client.files.get(name=uploaded_file.name)
            max_retries -= 1

        content_payload = [uploaded_file, prompt]

      # 3. Model invocation targeting currently active model identifiers
      explanation = ""
      errors = []
      model_candidates = [
          "gemini-3.6-flash",
          "gemini-3.6-pro",
          "gemini-2.0-flash",
      ]

      for model_name in model_candidates:
        try:
          response = self.client.models.generate_content(
              model=model_name, contents=content_payload
          )
          if response and response.text:
            explanation = response.text
            break
        except Exception as model_err:
          err_msg = f"{model_name}: {str(model_err)}"
          print(f"[Gemini Model Failure] {err_msg}")
          errors.append(err_msg)

      if not explanation:
        raise RuntimeError(" | ".join(errors))

      # 4. Cleanup cloud file
      if uploaded_file:
        try:
          self.client.files.delete(name=uploaded_file.name)
        except Exception:
          pass

      return {"status": "SUCCESS", "explanation_text": explanation}

    except Exception as err:
      if uploaded_file:
        try:
          self.client.files.delete(name=uploaded_file.name)
        except Exception:
          pass

      return {
          "status": "ERROR",
          "explanation_text": (
              f"Gemini Explainer exception occurred: {str(err)}"
          ),
      }


if __name__ == "__main__":
  explainer = GeminiForensicsExplainer()
  print(f"Gemini client initialized: {explainer.client is not None}")