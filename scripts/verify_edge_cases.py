"""Verification script for ALPR edge cases:
- Cars279.png (White Toyota tilted dark-red plate MMN 306)
- Cars59.png (Angled black/red car with plate MH. 46. P. 1661)
Both are tested with gt_path=None to ensure 100% blind pixel localization.
"""

import os
import sys
import xml.etree.ElementTree as ET

# Ensure backend can be imported
sys.path.insert(0, os.path.abspath("."))
from backend.services.ivp_pipeline import ALPRPipeline

def parse_gt_bbox(xml_path):
    if not os.path.exists(xml_path):
        return None
    tree = ET.parse(xml_path)
    root = tree.getroot()
    obj = root.find("object")
    if obj is not None:
        bnd = obj.find("bndbox")
        if bnd is not None:
            return [
                int(bnd.find("xmin").text),
                int(bnd.find("ymin").text),
                int(bnd.find("xmax").text),
                int(bnd.find("ymax").text),
            ]
    return None

def calc_iou(b1, b2):
    if not b1 or not b2:
        return 0.0
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / float(max(1, union))

def main():
    pipeline = ALPRPipeline()
    test_cases = [
        {
            "img": "dataset/images/Cars279.png",
            "xml": "dataset/annotations/Cars279.xml",
            "expected_plate": "MMN306",
            "desc": "White Toyota tilted dark-red plate (MMN 306)"
        },
        {
            "img": "dataset/images/Cars59.png",
            "xml": "dataset/annotations/Cars59.xml",
            "expected_plate": "MH46P1661",
            "desc": "Angled black/red car front bumper plate (MH 46 P 1661)"
        }
    ]

    all_passed = True
    print("=" * 70)
    print("RUNNING ALPR EDGE-CASE VERIFICATION WITH gt_path=None (BLIND LOCALIZATION)")
    print("=" * 70)

    for tc in test_cases:
        img_path = tc["img"]
        gt_box = parse_gt_bbox(tc["xml"])
        print(f"\nEvaluating: {tc['desc']} ({os.path.basename(img_path)})")
        print(f"  Ground Truth BBox: {gt_box}")

        # Run pipeline with NO ground truth path (blind localization from pixels)
        res = pipeline.process(image_path=img_path, gt_path=None)

        detected_box = res.get("hybrid_bbox") or res.get("plate_bbox")
        rec_plate = res.get("recognized_plate") or ""
        conf = res.get("confidence", 0.0)
        iou = calc_iou(detected_box, gt_box)

        print(f"  Detected BBox:     {detected_box}")
        print(f"  Calculated IoU:    {iou:.4f} (Required: >= 0.6500)")
        print(f"  Recognized Plate:  '{rec_plate}' (Expected: '{tc['expected_plate']}')")
        print(f"  OCR Confidence:    {conf * 100:.1f}%")

        # Assertions
        iou_ok = (iou >= 0.65)
        plate_ok = (tc["expected_plate"] in rec_plate or rec_plate == tc["expected_plate"])

        if iou_ok and plate_ok:
            print(f"  --> [PASS] {tc['desc']}")
        else:
            print(f"  --> [FAIL] IoU OK: {iou_ok}, Plate OK: {plate_ok}")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL EDGE-CASE TESTS PASSED PERFECTLY!")
        sys.exit(0)
    else:
        print("SOME EDGE-CASE TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
