"""Verification Script for Kaggle Dataset Integration & 7-Stage ALPR Pipeline.

Verifies:
1. Presence of all 433 Kaggle images (Cars0.png - Cars432.png) + 5 Indian HSRP images.
2. Runs ALPRPipeline on HSRP_0_MH12.png, HSRP_4_GJ05.png, and real Kaggle photos
   Cars0.png, Cars1.png, Cars2.png, Cars3.png.
3. Prints native resolutions, Classical & Hybrid IoU scores, and recognized license plate strings.
"""
import os
import sys

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.services.ivp_pipeline import ALPRPipeline
from backend.services.dataset_service import DatasetService


def main():
    print("=" * 80)
    print("STEP 1: DATASET VERIFICATION (KAGGLE + INDIAN HSRP)")
    print("=" * 80)

    dataset_dir = os.path.join(BASE_DIR, "dataset")
    service = DatasetService(dataset_dir)
    images_dir = os.path.join(dataset_dir, "images")
    annotations_dir = os.path.join(dataset_dir, "annotations")

    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_xmls = [f for f in os.listdir(annotations_dir) if f.lower().endswith('.xml')]

    print(f"Total images found in dataset/images/: {len(all_images)}")
    print(f"Total XML annotations found in dataset/annotations/: {len(all_xmls)}")

    hsrp_images = [f for f in all_images if f.startswith("HSRP_")]
    kaggle_images = [f for f in all_images if f.startswith("Cars")]

    print(f"Preserved Indian HSRP calibration images: {len(hsrp_images)} -> {sorted(hsrp_images)}")
    print(f"Real Kaggle camera photographs: {len(kaggle_images)} (Expected: 433)")

    assert len(hsrp_images) == 5, f"Expected 5 HSRP images, got {len(hsrp_images)}"
    assert len(kaggle_images) == 433, f"Expected 433 Kaggle images, got {len(kaggle_images)}"
    print(">>> STEP 1 PASSED: All 438 dataset files verified successfully!\n")

    print("=" * 80)
    print("STEP 2 & 3: ALPR PIPELINE VERIFICATION (RESOLUTIONS, IOU, OCR)")
    print("=" * 80)

    pipeline = ALPRPipeline()
    test_files = [
        "HSRP_0_MH12.png",
        "HSRP_4_GJ05.png",
        "Cars0.png",
        "Cars1.png",
        "Cars2.png",
        "Cars3.png",
    ]

    for fname in test_files:
        img_path = os.path.join(images_dir, fname)
        xml_path = service.get_annotation_path(fname)

        if not os.path.exists(img_path):
            print(f"[ERROR] Image not found: {img_path}")
            continue

        res = pipeline.process(
            image_path=img_path,
            config={},
            annotation_path=xml_path
        )

        w = res.get("image_width")
        h = res.get("image_height")
        gt = res.get("ground_truth", {})
        iou_c = gt.get("iou_classical")
        iou_h = gt.get("iou_hybrid")
        plate = res.get("recognized_plate")
        conf = res.get("confidence")
        p_type = res.get("plate_type")
        time_ms = res.get("total_time_ms")

        iou_c_str = f"{iou_c:.4f}" if iou_c is not None else "N/A"
        iou_h_str = f"{iou_h:.4f}" if iou_h is not None else "N/A"

        print(f"File: {fname:16} | Res: {w}x{h:3} | Plate: {str(plate):12} | Conf: {conf*100:5.1f}% | Type: {p_type:18} | Classical IoU: {iou_c_str:6} | Hybrid IoU: {iou_h_str:6} | Latency: {time_ms:.1f}ms")

        # Verify resolution fields exist
        assert w is not None and h is not None, f"Resolution missing for {fname}"
        # For images with ground truth annotations, hybrid IoU should exceed 0.50 (standard Pascal VOC benchmark)
        if iou_h is not None:
            assert iou_h >= 0.50, f"Hybrid IoU {iou_h} < 0.50 on {fname}"

    print("\n>>> STEP 2 & 3 PASSED: High-accuracy Hybrid IoU (>0.50) & dynamic resolutions verified!")
    print("=" * 80)


if __name__ == "__main__":
    main()
