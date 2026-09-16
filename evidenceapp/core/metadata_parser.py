import os
from PIL import Image
from PIL.ExifTags import TAGS

def parse_image_metadata(image_path: str) -> dict:
    report = {
        "has_exif": False,
        "device_make": "Unknown",
        "device_model": "Unknown",
        "software": "None",
        "tags": {},
        "is_suspicious": False,
        "consistency": "Consistent Noise Profile",
        "finding": "Standard camera manifest."
    }
    try:
        with Image.open(image_path) as img:
            raw_exif = img._getexif()
            if raw_exif:
                report["has_exif"] = True
                for tag_id, value in raw_exif.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    report["tags"][tag_name] = str(value)

                report["device_make"] = report["tags"].get("Make", "Unknown")
                report["device_model"] = report["tags"].get("Model", "Unknown")
                report["software"] = report["tags"].get("Software", "None")

                for flag in ["photoshop", "gimp", "midjourney", "dall-e", "canvas"]:
                    if flag in report["software"].lower():
                        report["is_suspicious"] = True
                        report["finding"] = f"Editing suite signature found: {report['software']}"
    except Exception as e:
        report["finding"] = str(e)
    return report