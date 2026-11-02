import cv2
import numpy as np
import re
import easyocr
import time

reader = easyocr.Reader(['en'], gpu=False)

discard_words = {
    "IND", "IKD", "INO", "IN", "SUZUKI", "SKOJO", "CALIFORNIA", "USA",
    "TOYOTA", "HONDA", "HYUNDAI", "FORD", "BMW", "MERCEDES", "DICKRAN", "OUZOUNIAN", "LTD", "MOTORS",
    "VIRGINIA", "TEXAS", "FLORIDA", "NEWYORK", "WASHINGTON", "MARYLAND", "PENNSYLVANIA", "ILLINOIS",
    "OHIO", "MICHIGAN", "ARIZONA", "COLORADO", "GEORGIA", "NORTHCAROLINA", "NEWJERSEY",
    "MASSACHUSETTS", "TENNESSEE", "INDIANA", "MISSOURI", "WISCONSIN", "MINNESOTA", "ALABAMA",
    "SOUTHCAROLINA", "LOUISIANA", "KENTUCKY", "OREGON", "OKLAHOMA", "CONNECTICUT", "UTAH", "IOWA",
    "NEVADA", "ARKANSAS", "MISSISSIPPI", "KANSAS", "NEWMEXICO", "NEBRASKA", "IDAHO", "WESTVIRGINIA",
    "HAWAII", "NEWHAMPSHIRE", "MAINE", "RHODEISLAND", "MONTANA", "DELAWARE", "SOUTHDAKOTA", "NORTHDAKOTA",
    "ALASKA", "VERMONT", "WYOMING",
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
}

def parse_plate_ocr(plate_roi):
    t0 = time.perf_counter()
    if plate_roi is None or plate_roi.size == 0:
        return "", 0.0, 0.0

    roi_h, roi_w = plate_roi.shape[:2]
    scale_f = max(140.0 / float(max(roi_h, 1)), 200.0 / float(max(roi_w, 1)))
    ocr_input = cv2.resize(plate_roi, (max(1, int(roi_w * scale_f)), max(1, int(roi_h * scale_f))), interpolation=cv2.INTER_CUBIC)

    # Pass 1: Standard Upscaled BGR + LAB CLAHE
    if len(ocr_input.shape) == 3:
        lab = cv2.cvtColor(ocr_input, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        p1 = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        p1 = clahe.apply(ocr_input)

    # Mask blue strip on left if detected
    if len(p1.shape) == 3:
        hsv_roi = cv2.cvtColor(p1, cv2.COLOR_BGR2HSV)
        raw_blue = cv2.inRange(hsv_roi, np.array([100, 90, 50]), np.array([135, 255, 255]))
        raw_blue[:, int(0.20 * p1.shape[1]):] = 0
        col_counts = np.sum(raw_blue > 0, axis=0)
        if len(col_counts) > 0 and np.max(col_counts) > (0.50 * p1.shape[0]):
            blue_cols = np.where(col_counts > (0.35 * p1.shape[0]))[0]
            if len(blue_cols) > 0:
                p1[:, :np.max(blue_cols) + 1] = [255, 255, 255]

    def _eval_pass(img_pass):
        padded = cv2.copyMakeBorder(img_pass, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        pad_h, pad_w = padded.shape[:2]
        res = reader.readtext(padded, detail=1, text_threshold=0.25, low_text=0.20, contrast_ths=0.05, adjust_contrast=0.7)
        boxes = []
        for r in res:
            box, txt, conf = r[0], r[1], float(r[2])
            clean = "".join(re.findall(r'[A-Za-z0-9]', txt)).upper()
            if not clean or clean in discard_words or conf < 0.15:
                continue
            pts = np.array(box, dtype=np.float32)
            xmin, xmax = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
            ymin, ymax = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
            bh = ymax - ymin
            bw = xmax - xmin
            xc = (xmin + xmax) / 2.0
            yc = (ymin + ymax) / 2.0

            # Discard bottom-frame dealer text
            if ymin > 0.65 * pad_h and bh < 0.28 * pad_h:
                continue
            # Discard 1-char border artifacts
            if len(clean) == 1 and (bw < 0.12 * pad_w or xc < 0.18 * pad_w):
                continue

            boxes.append({
                'box': box, 'txt': clean, 'raw': txt, 'conf': conf,
                'xc': xc, 'yc': yc, 'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax,
                'bw': bw, 'bh': bh
            })

        # Discard tiny 2-digit tax-sticker numbers (e.g. 03, 07, 16)
        if any(len(b['txt']) >= 3 for b in boxes):
            boxes = [b for b in boxes if not (len(b['txt']) <= 2 and b['txt'].isdigit() and b['bh'] < 0.45 * pad_h)]

        # Spatial Deduplication / Non-Maximum Suppression (NMS)
        if len(boxes) > 1:
            boxes.sort(key=lambda b: b['conf'], reverse=True)
            keep = []
            for b in boxes:
                # check overlap with already kept boxes
                is_dup = False
                for k in keep:
                    overlap_x = max(0.0, min(b['xmax'], k['xmax']) - max(b['xmin'], k['xmin']))
                    min_w = min(b['bw'], k['bw'])
                    vert_dist = abs(b['yc'] - k['yc'])
                    max_h = max(b['bh'], k['bh'])
                    if min_w > 0 and (overlap_x / min_w) >= 0.35 and vert_dist <= 0.60 * max_h:
                        is_dup = True
                        break
                if not is_dup:
                    keep.append(b)
            boxes = keep

        if not boxes:
            return "", 0.0

        # Group into columns (for multi-line plates like MH14 GN 9239)
        n_b = len(boxes)
        parent = list(range(n_b))
        def _find(i):
            if parent[i] == i: return i
            parent[i] = _find(parent[i])
            return parent[i]
        def _union(i, j):
            ri, rj = _find(i), _find(j)
            if ri != rj: parent[ri] = rj

        for i in range(n_b):
            for j in range(i + 1, n_b):
                bi, bj = boxes[i], boxes[j]
                overlap_x = max(0.0, min(bi['xmax'], bj['xmax']) - max(bi['xmin'], bj['xmin']))
                min_bw = min(bi['bw'], bj['bw'])
                # Only union if they are vertically separated (stacked lines)
                vert_gap = max(0.0, max(bi['ymin'], bj['ymin']) - min(bi['ymax'], bj['ymax']))
                if min_bw > 0 and (overlap_x / min_bw) >= 0.30 and vert_gap < 0.8 * max(bi['bh'], bj['bh']):
                    _union(i, j)

        groups = {}
        for i in range(n_b):
            root = _find(i)
            if root not in groups: groups[root] = []
            groups[root].append(boxes[i])

        group_list = []
        for root, items in groups.items():
            items.sort(key=lambda b: b['yc'])
            mean_xc = float(np.mean([b['xc'] for b in items]))
            group_list.append({
                'xc': mean_xc,
                'items': items,
                'text': "".join(b['txt'] for b in items),
                'confs': [b['conf'] for b in items]
            })

        group_list.sort(key=lambda g: g['xc'])
        stitched = "".join(g['text'] for g in group_list)
        all_confs = [c for g in group_list for c in g['confs']]
        mean_c = float(np.mean(all_confs)) if all_confs else 0.0
        return stitched, mean_c

    # Execute Pass 1
    stitched_plate, stitched_conf = _eval_pass(p1)

    # Hierarchical Fallback: only run Pass 2 if Pass 1 found insufficient characters
    if len(stitched_plate) < 4 or stitched_conf < 0.35:
        if len(ocr_input.shape) == 3:
            b_ch, g_ch, r_ch = cv2.split(ocr_input)
            max_c = np.maximum(np.maximum(r_ch, g_ch), b_ch)
            clahe_v = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            p2 = cv2.cvtColor(clahe_v.apply(max_c), cv2.COLOR_GRAY2BGR)
        else:
            clahe_v = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            p2 = cv2.cvtColor(clahe_v.apply(ocr_input), cv2.COLOR_GRAY2BGR)
        s2, c2 = _eval_pass(p2)
        if len(s2) >= len(stitched_plate) and c2 >= stitched_conf:
            stitched_plate, stitched_conf = s2, c2

    # Hierarchical Fallback: only test rotation if still < 4 chars
    if len(stitched_plate) < 4 or stitched_conf < 0.35:
        for ang in [-10, 10]:
            M = cv2.getRotationMatrix2D((ocr_input.shape[1] / 2.0, ocr_input.shape[0] / 2.0), ang, 1.0)
            rot = cv2.warpAffine(p1, M, (ocr_input.shape[1], ocr_input.shape[0]), borderValue=[255, 255, 255])
            s_rot, c_rot = _eval_pass(rot)
            if len(s_rot) >= len(stitched_plate) and c_rot >= stitched_conf:
                stitched_plate, stitched_conf = s_rot, c_rot
                if len(stitched_plate) >= 4 and stitched_conf >= 0.40:
                    break

    dt = (time.perf_counter() - t0) * 1000
    return stitched_plate, stitched_conf, dt

print("Loaded test harness.")
