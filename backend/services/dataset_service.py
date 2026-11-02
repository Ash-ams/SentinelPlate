import os
import re
import xml.etree.ElementTree as ET
import base64
import io
from typing import List, Optional, Dict, Any

try:
    import cv2
except ImportError:
    cv2 = None

from PIL import Image


DATASET_METADATA: Dict[str, Dict[str, str]] = {
    "HSRP_0_MH12.png": {"plate_text": "MH12DE1433", "plate_type": "White Private"},
    "HSRP_1_KA01.png": {"plate_text": "KA01AB1234", "plate_type": "Yellow Commercial"},
    "HSRP_2_DL3C.png": {"plate_text": "DL3CAB5678", "plate_type": "Green EV"},
    "HSRP_3_TN09.png": {"plate_text": "TN09CD9012", "plate_type": "White Private"},
    "HSRP_4_GJ05.png": {"plate_text": "GJ05EF3456", "plate_type": "Yellow Commercial"},
    "Cars0.png": {"plate_text": "KL54A2670", "plate_type": "Standard Private"},
    "Cars1.png": {"plate_text": "DL8CAF5030", "plate_type": "Standard Private"},
    "Cars2.png": {"plate_text": "MH20EE7598", "plate_type": "Standard Private"},
    "Cars3.png": {"plate_text": "HR26BC5514", "plate_type": "Standard Private"},
    "Cars4.png": {"plate_text": "KA04MH1435", "plate_type": "Standard Private"},
}


class DatasetService:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.annotations_dir = os.path.join(dataset_dir, "annotations")

    def _natural_sort_key(self, s: str):
        """Natural sort order with HSRP calibration images first, then Cars0..Cars432."""
        if s.startswith("HSRP_"):
            m = re.search(r'\d+', s)
            num = int(m.group(0)) if m else 0
            return (0, num, s)
        if s.startswith("Cars"):
            m = re.search(r'\d+', s)
            num = int(m.group(0)) if m else 0
            return (1, num, s)
        return (2, [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)])

    def _generate_thumbnail(self, image_path: str, width: int = 100) -> Optional[str]:
        """Generate a JPEG base64 data URI thumbnail resized to the given width."""
        try:
            if cv2 is not None:
                img = cv2.imread(image_path)
                if img is not None:
                    h, w = img.shape[:2]
                    if w > 0 and h > 0:
                        new_h = max(1, int(h * (width / w)))
                        resized = cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)
                        success, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if success:
                            b64 = base64.b64encode(buffer).decode("utf-8")
                            return f"data:image/jpeg;base64,{b64}"

            # Fallback to Pillow
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                w, h = im.size
                if w <= 0 or h <= 0:
                    return None
                new_h = max(1, int(h * (width / w)))
                im_resized = im.resize((width, new_h), Image.Resampling.LANCZOS)
                buffered = io.BytesIO()
                im_resized.save(buffered, format="JPEG", quality=80)
                b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/jpeg;base64,{b64}"
        except Exception:
            return None

    def get_metadata(self, filename: str) -> Dict[str, str]:
        """Get ground truth metadata for an image filename."""
        base = os.path.basename(filename)
        if base in DATASET_METADATA:
            return DATASET_METADATA[base]
        if base.startswith("Cars"):
            return {"plate_text": f"Kaggle Plate ({base})", "plate_type": "Standard Private"}
        return {"plate_text": "MH12DE1433", "plate_type": "White Private"}

    def list_images(self) -> List[Dict[str, Any]]:
        """List all images with their annotation status, bounding boxes, plate text, category, and thumbnails."""
        if not os.path.exists(self.images_dir):
            return []

        supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        filenames = [
            f for f in os.listdir(self.images_dir)
            if os.path.splitext(f)[1].lower() in supported_extensions
        ]
        filenames.sort(key=self._natural_sort_key)

        results = []
        for fname in filenames:
            img_full_path = os.path.join(self.images_dir, fname)
            xml_path = self.get_annotation_path(fname)
            bbox = self.parse_annotation(xml_path) if xml_path else None
            thumbnail = self._generate_thumbnail(img_full_path, width=100)

            meta = self.get_metadata(fname)
            plate_text = meta.get("plate_text", "MH12DE1433")
            plate_type = meta.get("plate_type", "White Private")
            bndbox_dict = {
                "xmin": bbox[0],
                "ymin": bbox[1],
                "xmax": bbox[2],
                "ymax": bbox[3]
            } if bbox else None

            results.append({
                "filename": fname,
                "has_annotation": bbox is not None,
                "bbox": bbox,
                "thumbnail": thumbnail,
                "plate_text": plate_text,
                "plate_type": plate_type,
                "bndbox": bndbox_dict
            })

        return results

    def parse_annotation(self, xml_path: str) -> Optional[List[int]]:
        """Parse Pascal VOC XML to extract first bndbox as [xmin, ymin, xmax, ymax]."""
        if not xml_path or not os.path.exists(xml_path):
            return None
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            obj = root.find(".//object")
            if obj is None:
                return None
            bbox = obj.find("bndbox")
            if bbox is None:
                return None
            xmin = int(round(float(bbox.find("xmin").text)))
            ymin = int(round(float(bbox.find("ymin").text)))
            xmax = int(round(float(bbox.find("xmax").text)))
            ymax = int(round(float(bbox.find("ymax").text)))
            return [xmin, ymin, xmax, ymax]
        except Exception:
            return None

    def get_annotation_path(self, image_filename: str) -> Optional[str]:
        """Get annotation XML path for a given image filename."""
        if not image_filename:
            return None
        name = os.path.splitext(os.path.basename(image_filename))[0]
        xml_path = os.path.join(self.annotations_dir, f"{name}.xml")
        return xml_path if os.path.exists(xml_path) else None

    def get_image_path(self, image_filename: str) -> Optional[str]:
        """Get full path of an image file."""
        if not image_filename:
            return None
        if os.path.isabs(image_filename) and os.path.exists(image_filename):
            return image_filename
        base = os.path.basename(image_filename)
        path = os.path.join(self.images_dir, base)
        if os.path.exists(path):
            return path
        path = os.path.join(self.dataset_dir, image_filename)
        if os.path.exists(path):
            return path
        return None
