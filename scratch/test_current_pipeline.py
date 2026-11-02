import os
import sys
import time

sys.path.insert(0, ".")
from backend.services.ivp_pipeline import ALPRPipeline

pipeline = ALPRPipeline()

test_images = [
    ("dataset/images/Cars4.png", "PUI8BES", "Cars4 (PUI8 BES)"),
    ("dataset/images/Cars0.png", "KL01CA2555", "Cars0 (KL 01 CA 2555)"),
    ("dataset/images/Cars36.png", "80211N", "Cars36 (802.11N)"),
    ("dataset/images/Cars7.png", "YSX213", "Cars7 (YSX-213)"),
    ("dataset/images/Cars22.png", "", "Cars22 (YAMRAJ / bumper)"),
    ("dataset/images/Cars59.png", "MH46P1661", "Cars59 (MH 46 P 1661)"),
    ("dataset/images/Cars279.png", "MMN306", "Cars279 (MMN 306)"),
]

print("=" * 80)
print("TESTING PIPELINE ON CURRENT IMAGES")
print("=" * 80)

for img_path, expected, desc in test_images:
    t0 = time.perf_counter()
    res = pipeline.process(image_path=img_path, gt_path=None)
    dt = (time.perf_counter() - t0) * 1000
    plate = res.get("recognized_plate")
    conf = res.get("confidence", 0.0)
    bbox = res.get("plate_bbox")
    print(f"{desc:25} | Plate: {str(plate):25} | Conf: {conf*100:5.1f}% | Time: {dt:7.1f}ms | BBox: {bbox}")
