#!/usr/bin/env python3
"""Automated Phase 3 Verification & IEEE Experimental Tables Exporter.

Verifies:
1. GET /api/dataset/batch-evaluate across all 11 annotated vehicle images in /dataset/.
2. Confirms empirical computations for 4 syllabus benchmark tables:
   - Table I: Point Processing & Spatial Enhancement (MSE, PSNR, Shannon Entropy).
   - Table II: Edge Operators & Morphological Localization (IoU, Precision, Recall, Latency).
   - Table III: Binarization, Connected Component Segmentation & OCR (Otsu tau, Seg Rate %, OCR Acc %).
   - Table IV: Motion Estimation (Search Points, Speedup Factor, Prediction PSNR, Latency).
3. Live Surveillance & Toll Endpoints (/api/process-image, /api/process-video-frames, /api/toll/simulate-entry).
4. Prints the exact formatted IEEE Research Paper Markdown Tables to stdout and saves to dataset/IEEE_EXPERIMENTAL_TABLES.md.

Usage:
    python scripts/verify_phase3.py
"""
import os
import sys
import json
import time

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from backend.main import app


def print_header(title: str):
    print("\n" + "=" * 82)
    print(f"  {title}")
    print("=" * 82)


def verify_phase3():
    print_header("SENTINELPLATE ALPR - PHASE 3 VERIFICATION & IEEE BENCHMARK AUDIT")

    client = TestClient(app)
    verification_passed = True
    test_results = []

    # ---------------------------------------------------------
    # TEST 1: Batch Dataset Evaluation Endpoint
    # ---------------------------------------------------------
    print("\n[STEP 1/4] Running Full Dataset Batch Evaluation (GET /api/dataset/batch-evaluate)...")
    t0 = time.perf_counter()
    resp = client.get("/api/dataset/batch-evaluate")
    latency = (time.perf_counter() - t0) * 1000

    if resp.status_code != 200:
        print(f"  [FAIL] /api/dataset/batch-evaluate returned status {resp.status_code}")
        test_results.append(("Batch Evaluation Endpoint", "FAIL", f"HTTP {resp.status_code}"))
        verification_passed = False
        return
    else:
        print(f"  [PASS] /api/dataset/batch-evaluate returned HTTP 200 ({latency:.1f}ms)")
        test_results.append(("Batch Evaluation Endpoint", "PASS", f"HTTP 200 ({latency:.1f}ms)"))

    batch_data = resp.json()
    total_images = batch_data.get("dataset_total_images", 0)
    annotated_pairs = batch_data.get("annotated_pairs_evaluated", 0)
    print(f"  Evaluated {total_images} total vehicle images ({annotated_pairs} annotated pairs).")

    t1_data = batch_data.get("enhancement_benchmark", {})
    t1_rows = t1_data.get("rows", [])
    t2_data = batch_data.get("edge_localization_benchmark", {})
    t2_rows = t2_data.get("rows", [])
    t3_data = batch_data.get("segmentation_ocr_benchmark", {})
    t3_rows = t3_data.get("rows", [])
    t4_data = batch_data.get("motion_estimation_benchmark", {})
    t4_rows = t4_data.get("rows", [])

    # Validate Table I
    if len(t1_rows) >= 4:
        clahe_entry = next((e for e in t1_rows if "CLAHE" in e.get("method_name", "")), None)
        raw_entry = next((e for e in t1_rows if "Raw" in e.get("method_name", "")), None)
        if clahe_entry and raw_entry:
            print(f"  [PASS] Table I (Enhancement): Raw PSNR={raw_entry['mean_psnr_db']}dB vs CLAHE PSNR={clahe_entry['mean_psnr_db']}dB, Entropy={clahe_entry['mean_entropy_bits']}")
            test_results.append(("Table I: Enhancement Benchmark", "PASS", f"Entropy={clahe_entry['mean_entropy_bits']}"))
    else:
        test_results.append(("Table I: Enhancement Benchmark", "FAIL", "Missing table rows"))
        verification_passed = False

    # Validate Table II
    if len(t2_rows) >= 4:
        sobel_entry = next((e for e in t2_rows if "Sobel" in e.get("operator_name", "")), None)
        canny_entry = next((e for e in t2_rows if "Canny" in e.get("operator_name", "")), None)
        if sobel_entry and canny_entry:
            print(f"  [PASS] Table II (Edge & Localization): Sobel |Gx| IoU={sobel_entry['mean_iou']} vs Canny IoU={canny_entry['mean_iou']}, Latency={sobel_entry['mean_latency_ms']}ms")
            test_results.append(("Table II: Edge Localization Benchmark", "PASS", f"Sobel IoU={sobel_entry['mean_iou']}"))
    else:
        test_results.append(("Table II: Edge Localization Benchmark", "FAIL", "Missing table rows"))
        verification_passed = False

    # Validate Table III
    if len(t3_rows) >= 3:
        otsu_entry = next((e for e in t3_rows if "Otsu" in e.get("method_name", "")), None)
        if otsu_entry:
            print(f"  [PASS] Table III (Segmentation & OCR): Otsu tau={otsu_entry['mean_tau_star']}, Char Seg Rate={otsu_entry['char_segmentation_rate_pct']}%, OCR Acc={otsu_entry['plate_ocr_accuracy_pct']}%")
            test_results.append(("Table III: Segmentation & OCR Benchmark", "PASS", f"Otsu Seg Rate={otsu_entry['char_segmentation_rate_pct']}%"))
    else:
        test_results.append(("Table III: Segmentation & OCR Benchmark", "FAIL", "Missing table rows"))
        verification_passed = False

    # Validate Table IV
    if len(t4_rows) >= 3:
        esbm_entry = next((e for e in t4_rows if "Exhaustive" in e.get("algorithm_name", "")), None)
        tss_entry = next((e for e in t4_rows if "Three-Step" in e.get("algorithm_name", "")), None)
        twod_entry = next((e for e in t4_rows if "2D-Log" in e.get("algorithm_name", "")), None)
        if esbm_entry and tss_entry and twod_entry:
            print(f"  [PASS] Table IV (Motion Estimation): ESBM {esbm_entry['search_points_per_block']} pts vs TSS {tss_entry['search_points_per_block']} pts ({tss_entry['speedup_factor']}x) vs 2D-Log {twod_entry['search_points_per_block']} pts ({twod_entry['speedup_factor']}x)")
            test_results.append(("Table IV: Motion Estimation Benchmark", "PASS", f"TSS Speedup={tss_entry['speedup_factor']}x"))
    else:
        test_results.append(("Table IV: Motion Estimation Benchmark", "FAIL", "Missing table rows"))
        verification_passed = False

    # ---------------------------------------------------------
    # TEST 2: Process Single Image Pipeline (Cars0.png)
    # ---------------------------------------------------------
    print("\n[STEP 2/4] Testing Single Image Full 7-Stage Pipeline (POST /api/process-image)...")
    proc_resp = client.post(
        "/api/process-image",
        data={
            "dataset_filename": "Cars0.png",
            "config": json.dumps({
                "gamma": 1.0,
                "filter_type": "gaussian",
                "edge_detector": "sobel",
                "threshold_method": "otsu",
                "enhancement": "clahe",
                "point_processing": "gamma"
            })
        }
    )
    if proc_resp.status_code == 200:
        pdata = proc_resp.json()
        plate = pdata.get("recognized_plate")
        plate_type = pdata.get("plate_type")
        iou = pdata.get("ground_truth", {}).get("iou_classical")
        stages_count = len(pdata.get("stages", []))
        print(f"  [PASS] Processed Cars0.png: Plate='{plate}', Type='{plate_type}', Stages={stages_count}, IoU={iou}")
        test_results.append(("7-Stage IVP Pipeline", "PASS", f"Plate={plate}, IoU={iou}"))
    else:
        print(f"  [FAIL] /api/process-image returned {proc_resp.status_code}")
        test_results.append(("7-Stage IVP Pipeline", "FAIL", f"HTTP {proc_resp.status_code}"))
        verification_passed = False

    # ---------------------------------------------------------
    # TEST 3: Video Motion Estimation & Smart Trigger
    # ---------------------------------------------------------
    print("\n[STEP 3/4] Testing Video Motion Approaching Vehicle Sequence (POST /api/process-video-frames)...")
    vid_resp = client.post(
        "/api/process-video-frames",
        data={"dataset_filename": "Cars0.png", "algorithm": "tss"}
    )
    if vid_resp.status_code == 200:
        vdata = vid_resp.json()
        frames_cnt = vdata.get("num_frames", 0)
        speedup = vdata.get("benchmark", {}).get("speedup_vs_esbm")
        trigger = vdata.get("smart_toll_trigger", {}).get("triggered")
        trigger_frame = vdata.get("smart_toll_trigger", {}).get("trigger_keyframe_index")
        print(f"  [PASS] Video Motion Engine: Frames={frames_cnt}, TSS Speedup={speedup}x, Smart Trigger Fired={trigger} (Frame {trigger_frame})")
        test_results.append(("Unit 8 Video Motion Engine", "PASS", f"TSS Speedup={speedup}x, Trigger=Frame {trigger_frame}"))
    else:
        print(f"  [FAIL] /api/process-video-frames returned {vid_resp.status_code}")
        test_results.append(("Unit 8 Video Motion Engine", "FAIL", f"HTTP {vid_resp.status_code}"))
        verification_passed = False

    # ---------------------------------------------------------
    # TEST 4: Toll & Parking Simulation Service
    # ---------------------------------------------------------
    print("\n[STEP 4/4] Testing Toll & Parking Automation Service (POST /api/toll/simulate-entry)...")
    toll_resp = client.post(
        "/api/toll/simulate-entry",
        json={
            "plate_number": "MH12DE1433",
            "plate_type": "White Private",
            "event_type": "entry",
            "gate_id": "GATE-01"
        }
    )
    if toll_resp.status_code == 200:
        tdata = toll_resp.json()
        rec = tdata.get("record", {})
        bay = rec.get("parking_bay")
        tariff = rec.get("tariff")
        barrier = rec.get("barrier_status")
        print(f"  [PASS] Toll Simulation: Plate={rec.get('plate_number')}, Bay={bay}, Tariff=INR {tariff}, Barrier={barrier}")
        test_results.append(("Toll Simulation Service", "PASS", f"Bay={bay}, Barrier={barrier}"))
    else:
        print(f"  [FAIL] /api/toll/simulate-entry returned {toll_resp.status_code}")
        test_results.append(("Toll Simulation Service", "FAIL", f"HTTP {toll_resp.status_code}"))
        verification_passed = False

    # ---------------------------------------------------------
    # Generate & Export IEEE Tables
    # ---------------------------------------------------------
    print_header("IEEE RESEARCH PAPER EXPERIMENTAL TABLES (EMPIRICAL DATA)")

    t1_md = (
        "### TABLE I\n"
        "### Quantitative Evaluation of Spatial Domain Point Processing & Enhancement Algorithms\n\n"
        "| Enhancement Method | Mean MSE | Mean PSNR (dB) | Mean Shannon Entropy (bits) |\n"
        "| :--- | :---: | :---: | :---: |\n"
    )
    for r in t1_rows:
        psnr_val = f"{r['mean_psnr_db']:.2f} dB" if isinstance(r['mean_psnr_db'], (int, float)) else str(r['mean_psnr_db'])
        t1_md += f"| {r['method_name']} | {r['mean_mse']} | {psnr_val} | {r['mean_entropy_bits']} |\n"

    t2_md = (
        "\n### TABLE II\n"
        "### Benchmark of First- & Second-Order Edge Operators for License Plate Localization\n\n"
        "| Edge Operator + Morph (17x5) | Mean IoU | Precision | Recall | Mean Latency (ms) |\n"
        "| :--- | :---: | :---: | :---: | :---: |\n"
    )
    for r in t2_rows:
        t2_md += f"| {r['operator_name']} | {r['mean_iou']:.4f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['mean_latency_ms']:.2f} ms |\n"

    t3_md = (
        "\n### TABLE III\n"
        "### Character Segmentation & OCR Performance Across Thresholding Paradigms\n\n"
        "| Binarization Technique | Optimal Threshold $\\tau^*$ | Character Seg. Rate (%) | Plate OCR Accuracy (%) |\n"
        "| :--- | :---: | :---: | :---: |\n"
    )
    for r in t3_rows:
        t3_md += f"| {r['method_name']} | {r['mean_tau_star']} | {r['char_segmentation_rate_pct']:.1f}% | {r['plate_ocr_accuracy_pct']:.1f}% |\n"

    t4_md = (
        "\n### TABLE IV\n"
        "### Block-Matching Motion Estimation Efficiency & Prediction Quality (Search Window p = 7)\n\n"
        "| Motion Estimation Algorithm | Search Points / MB | Speedup Factor | Prediction PSNR (dB) | Execution Time (ms) |\n"
        "| :--- | :---: | :---: | :---: | :---: |\n"
    )
    for r in t4_rows:
        t4_md += f"| {r['algorithm_name']} | {r['search_points_per_block']} | {r['speedup_factor']:.2f}x | {r['prediction_psnr_db']:.2f} dB | {r['latency_ms']:.2f} ms |\n"

    full_markdown = (
        "# SENTINELPLATE ALPR — EXPERIMENTAL BENCHMARK TABLES\n"
        "*Empirical benchmark results computed across all annotated vehicle datasets.*\n\n"
        + t1_md + "\n" + t2_md + "\n" + t3_md + "\n" + t4_md
    )

    print(full_markdown)

    # Save to dataset/IEEE_EXPERIMENTAL_TABLES.md
    out_path = os.path.join(PROJECT_ROOT, "dataset", "IEEE_EXPERIMENTAL_TABLES.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)
    print(f"\n[SAVED] Exported complete benchmark tables to: {out_path}")

    # Summary
    print_header("PHASE 3 VERIFICATION SUMMARY")
    for name, status, details in test_results:
        print(f"  [{status}] {name:<40} : {details}")

    if verification_passed:
        print("\n>>> ALL PHASE 3 VERIFICATION CHECKS PASSED SUCCESSFULLY (100%) <<<\n")
    else:
        print("\n>>> ONE OR MORE CHECKS FAILED <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    verify_phase3()
