import os
import sys
import glob
import sqlite3
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from backend.main import app, STATIC_IMAGES_DIR, ensure_data_uri
from backend.services.ivp_pipeline import ALPRPipeline

def test_static_files():
    print("=== Testing Static Image Serving ===")
    client = TestClient(app)
    res = client.get("/api/static-images/Cars0.png")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert len(res.content) > 1000, "Image content too small"
    print("  [PASS] /api/static-images/Cars0.png returned 200 with", len(res.content), "bytes")

    res2 = client.get("/api/dataset/image/Cars0.png")
    assert res2.status_code == 200, f"Expected 200, got {res2.status_code}"
    print("  [PASS] /api/dataset/image/Cars0.png returned 200 with", len(res2.content), "bytes")

def test_toll_database():
    print("\n=== Testing toll_parking.db Clean INR Records ===")
    conn = sqlite3.connect("toll_parking.db")
    c = conn.cursor()
    c.execute("SELECT id, plate_number, plate_type, tariff FROM toll_logs")
    rows = c.fetchall()
    print(f"  Found {len(rows)} records:")
    total_rev = 0.0
    for r in rows:
        print(f"    ID: {r[0]}, Plate: {r[1]}, Type: {r[2]}, Tariff: INR {r[3]}")
        total_rev += float(r[3])
        assert r[3] != 1.44, f"Found stale $1.44 record: {r}"

    assert len(rows) == 5, f"Expected 5 records, got {len(rows)}"
    assert round(total_rev, 2) == 510.00, f"Expected INR 510.00, got {total_rev}"
    print(f"  [PASS] Total Revenue = INR {total_rev:.2f} (Clean 5 records, no 1.44 rows)")
    conn.close()

def test_all_11_dataset_images():
    print("\n=== Testing All 11 Dataset Cars with Pipeline ===")
    img_files = sorted(glob.glob("dataset/images/*.png"))
    print(f"  Found {len(img_files)} images to test:")

    pipeline = ALPRPipeline()
    for img_path in img_files:
        fname = os.path.basename(img_path)
        ann_name = os.path.splitext(fname)[0] + ".xml"
        ann_path = os.path.join("dataset/annotations", ann_name)
        if not os.path.exists(ann_path):
            ann_path = None

        res = pipeline.process(img_path, {}, ann_path)
        
        # Verify original_image
        orig_img = res.get("original_image")
        assert orig_img, f"Missing original_image for {fname}"
        assert orig_img.startswith("data:image/png;base64,"), f"original_image not valid data URI for {fname}"
        assert len(orig_img) > 200, f"original_image payload too small for {fname}"

        # Verify stages
        stages = res.get("stages", [])
        assert len(stages) >= 7, f"Expected 7 stages, got {len(stages)} for {fname}"
        
        # Check intermediate images
        img_count = 0
        for s in stages:
            s_imgs = s.get("images", {})
            for k, v in s_imgs.items():
                v_uri = ensure_data_uri(v)
                assert v_uri.startswith("data:image/"), f"Stage {s['stage_number']} image {k} invalid URI"
                img_count += 1

        plate = res.get("recognized_plate") or "UNKNOWN"
        iou = res.get("ground_truth", {}).get("iou_classical")
        iou_str = f"{iou:.3f}" if iou is not None else "N/A"
        print(f"  [PASS] {fname:14s} | Stages: {len(stages)} | Images: {img_count:2d} | Plate: {plate:10s} | IoU: {iou_str}")

    print("  [SUCCESS] All dataset images produced valid, non-empty Base64 data URIs!")

if __name__ == '__main__':
    test_static_files()
    test_toll_database()
    test_all_11_dataset_images()
    print("\n=== ALL VERIFICATION CHECKS PASSED SUCCESSFULLY ===")
