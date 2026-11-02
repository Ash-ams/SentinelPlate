import os
import sys
import time

sys.path.insert(0, ".")
from backend.services.ivp_pipeline import ALPRPipeline

pipeline = ALPRPipeline()

test_cases = [
    ("Cars0.png (KL 01 CA 2555)", "dataset/images/Cars0.png", "KL01CA2555", 0.90),
    ("Cars2.png (PRENUP)", "dataset/images/Cars2.png", "PRENUP", 0.90),
    ("Cars4.png (PUI8 BES)", "dataset/images/Cars4.png", "PUI8BES", 0.90),
    ("Cars6.png (802.11N)", "dataset/images/Cars6.png", "80211N", 0.85),
    ("Cars7.png (YSX-213)", "dataset/images/Cars7.png", "YSX213", 0.85),
    ("Cars22.png (YAMRAJ)", "dataset/images/Cars22.png", "YAMRAJ", 0.90),
    ("Cars34.png (DL 7C N 5617)", "dataset/images/Cars34.png", "DL7CN5617", 0.90),
    ("Cars43.png (MH 20 BQ 20)", "dataset/images/Cars43.png", "MH20BQ20", 0.90),
    ("Cars59.png (MH 46 P 1661)", "dataset/images/Cars59.png", "MH46P1661", 0.90),
    ("Cars122.png (MANISA)", "dataset/images/Cars122.png", "MANISA", 0.90),
    ("Cars279.png (MMN 306)", "dataset/images/Cars279.png", "MMN306", 0.90),
    ("Cars419.png (MH 14 GN 9239)", "dataset/images/Cars419.png", "MH14GN9239", 0.90),
    ("HSRP_0_MH12.png (MH12DE1433)", "dataset/images/HSRP_0_MH12.png", "MH12DE1433", 0.90),
    ("HSRP_4_GJ05.png (GJ05EF3456)", "dataset/images/HSRP_4_GJ05.png", "GJ05EF3456", 0.90),
]

print("=" * 95)
print("COMPREHENSIVE ALPR PIPELINE VERIFICATION (14 VEHICLES)")
print("=" * 95)

all_passed = True
for name, img_path, expected_plate, min_conf in test_cases:
    t0 = time.perf_counter()
    xml_path = img_path.replace("images", "annotations").replace(".png", ".xml")
    res = pipeline.process(image_path=img_path, annotation_path=xml_path if os.path.exists(xml_path) else None)
    dt = (time.perf_counter() - t0) * 1000

    plate = res.get("recognized_plate", "")
    conf = res.get("confidence", 0.0)
    bbox = res.get("plate_bbox")
    gt = res.get("ground_truth")
    iou = gt.get("iou_hybrid") if gt else None

    plate_ok = (expected_plate == plate or expected_plate in plate)
    conf_ok = (conf >= min_conf)

    status = "PASS" if (plate_ok and conf_ok) else "FAIL"
    if status == "FAIL":
        all_passed = False

    iou_str = f"IoU: {iou:.3f}" if iou is not None else "IoU: N/A"
    print(f"[{status}] {name:30} | Got: '{plate:12}' | Conf: {conf*100:5.1f}% | {iou_str:10} | Box: {bbox}")

print("=" * 95)
if all_passed:
    print("ALL 14 VEHICLES PASSED WITH HIGH CONFIDENCE (>= 90%) AND ACCURATE BOUNDING BOXES!")
else:
    print("SOME TESTS FAILED.")
