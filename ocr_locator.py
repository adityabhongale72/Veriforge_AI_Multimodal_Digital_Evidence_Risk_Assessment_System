import os
import re
import cv2
import numpy as np

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

class OCRLocator:
    def __init__(self, tesseract_cmd_path: str = None):
        if PYTESSERACT_AVAILABLE and tesseract_cmd_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path

    def locate_text_regions(self, image_path: str) -> list[dict]:
        if not os.path.exists(image_path):
            return []
        img = cv2.imread(image_path)
        if img is None:
            return []

        h_orig, w_orig = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h + 1e-5)
            if 30 < w < (w_orig * 0.95) and 10 < h < (h_orig * 0.40) and aspect_ratio > 1.2:
                boxes.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "confidence": 0.85
                })

        return sorted(boxes, key=lambda b: (b["bbox"][1] // 20, b["bbox"][0]))

    def extract_document_entities(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            return {"error": "Image not found"}

        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Failed to decode image"}

        text_boxes = self.locate_text_regions(image_path)
        extracted_fields = {"reference_id": None, "case_id": None, "date": None}

        if PYTESSERACT_AVAILABLE:
            try:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                raw_text = pytesseract.image_to_string(rgb_img, config="--psm 6")
                ref_match = re.search(r'(Ref(?:erence)?\s*(?:ID|No)?[:\-\s]+([A-Z0-9\-]+))', raw_text, re.I)
                case_match = re.search(r'(?:Case\s*(?:ID|No)?[:\-\s]+)([A-Z0-9\/\-]+)', raw_text, re.I)
                if ref_match:
                    extracted_fields["reference_id"] = ref_match.group(1).strip()
                if case_match:
                    extracted_fields["case_id"] = case_match.group(1).strip()
            except Exception:
                pass

        return {
            "ocr_engine": "Tesseract-OCR" if PYTESSERACT_AVAILABLE else "OpenCV Heuristic Contours",
            "total_regions": len(text_boxes),
            "regions": text_boxes,
            "extracted_fields": extracted_fields
        }

def extract_text_regions(image_path: str) -> list:
    locator = OCRLocator()
    detected = locator.locate_text_regions(image_path)
    return [d["bbox"] for d in detected]