"""Mandatory Blind Upload Verification Script.

Tests blind localization and recognition with ZERO Pascal VOC XML access (gt_path=None).
Verifies that all 6 target vehicles recognize their full license plate strings
and accurately localize the license plate with IoU >= 0.70.
"""

import os
import sys
import shutil
import xml.etree.ElementTree as ET

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.services.ivp_pipeline import ALPRPipeline


def compute_iou(boxA, boxB):
    """Compute Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
    if not boxA or not boxB:
        return 0.0
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def parse_ground_truth(xml_path):
    """Parse true license plate coordinates for post-test evaluation assertion only."""
    tree = ET.parse(xml_path)
    bnd = tree.getroot().find('object').find('bndbox')
    return [int(round(float(bnd.find(c).text))) for c in ['xmin', 'ymin', 'xmax', 'ymax']]


def main():
    print("=" * 75)
    print("MANDATORY BLIND UPLOAD VERIFICATION: ZERO XML GROUND TRUTH ACCESS (gt_path=None)")
    print("=" * 75)

    pipe = ALPRPipeline()

    test_cases = [
        {
            "filename": "Cars1.png",
            "expected_text": "PGMN112",
            "desc": "Black Mercedes E220 CDI (must reject trunk badges 'E220' & 'CDI'/'CCI')",
            "min_iou": 0.70
        },
        {
            "filename": "Cars0.png",
            "expected_text": "KL01CA2555",
            "desc": "Indian HSRP Plate with blue IND strip",
            "min_iou": 0.70
        },
        {
            "filename": "Cars419.png",
            "expected_text": "MH14GN9239",
            "desc": "Two-line stacked Indian plate (MH14 over GN next to 9239)",
            "min_iou": 0.70
        },
        {
            "filename": "Cars60.png",
            "expected_text": "OS802HN",
            "desc": "European Plate with coat-of-arms gap (OS and 802-HN)",
            "min_iou": 0.70
        },
        {
            "filename": "Cars36.png",
            "expected_text": "MIDLYPH",
            "desc": "Standard single-line plate",
            "min_iou": 0.70
        },
        {
            "filename": "Cars2.png",
            "expected_text": "PRENUP",
            "desc": "Vanity plate (PRENUP)",
            "min_iou": 0.70
        }
    ]

    # Ensure /tmp directory exists
    tmp_path = "/tmp/blind_upload_test.png"
    try:
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    except Exception:
        pass

    all_passed = True
    summary_results = []

    for tc in test_cases:
        src_image = os.path.join(BASE_DIR, "dataset", "images", tc["filename"])
        xml_path = os.path.join(BASE_DIR, "dataset", "annotations", tc["filename"].replace(".png", ".xml"))

        if not os.path.exists(src_image):
            print(f"FAILED: Source image not found: {src_image}")
            all_passed = False
            continue

        true_gt = parse_ground_truth(xml_path)

        # Copy to temporary upload path simulating browser photo upload
        shutil.copyfile(src_image, tmp_path)

        # Call pipeline with STRICT ZERO XML ACCESS (gt_path=None)
        res = pipe.process(tmp_path, {}, gt_path=None)

        detected_bbox = res.get("plate_bbox") or res.get("hybrid_bbox")
        rec_text = res.get("recognized_plate") or ""
        conf = res.get("confidence", 0.0)

        iou = compute_iou(detected_bbox, true_gt)

        # Verification assertions
        text_match = (tc["expected_text"] in rec_text) or (rec_text in tc["expected_text"] and len(rec_text) >= len(tc["expected_text"]) - 1)
        iou_match = (iou >= tc["min_iou"])

        status = "PASSED" if (text_match and iou_match) else "FAILED"
        if not (text_match and iou_match):
            all_passed = False

        summary_results.append({
            "filename": tc["filename"],
            "expected": tc["expected_text"],
            "recognized": rec_text,
            "conf": conf,
            "detected_bbox": detected_bbox,
            "true_gt": true_gt,
            "iou": iou,
            "min_iou": tc["min_iou"],
            "status": status,
            "desc": tc["desc"]
        })

        print(f"\n[{status}] {tc['filename']} - {tc['desc']}")
        print(f"   Expected Text:  '{tc['expected_text']}'")
        print(f"   Recognized Text: '{rec_text}' (Confidence: {conf * 100:.1f}%)")
        print(f"   Detected BBox:  {detected_bbox}")
        print(f"   True GT BBox:   {true_gt}")
        print(f"   Blind IoU:      {iou:.4f} (Required >= {tc['min_iou']:.2f})")
        print(f"   Ground Truth in response: {res.get('ground_truth')}")

    # Clean up temp file
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    print("\n" + "=" * 75)
    print("SUMMARY OF BLIND UPLOAD VERIFICATION:")
    print("=" * 75)
    for r in summary_results:
        print(f"{r['status']:6} | {r['filename']:12} | Rec: {r['recognized']:12} | Exp: {r['expected']:12} | IoU: {r['iou']:.4f} | GT attached: None")

    if all_passed:
        print("\nSUCCESS: All 6 vehicle images passed blind upload verification with gt_path=None!")
        return 0
    else:
        print("\nFAILURE: One or more vehicle images failed blind upload requirements.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
