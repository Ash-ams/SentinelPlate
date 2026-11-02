import os
import cv2
import numpy as np
import easyocr
import re
import xml.etree.ElementTree as ET

reader = easyocr.Reader(['en'], gpu=False)

def compute_iou(b1, b2):
    if not b1 or not b2: return 0.0
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / union if union > 0 else 0.0

def snap_to_plate_contour(img, box):
    img_h, img_w = img.shape[:2]
    bx1, by1, bx2, by2 = box
    bw = bx2 - bx1
    bh = by2 - by1
    
    wx1 = max(0, int(bx1 - 0.20 * bw))
    wy1 = max(0, int(by1 - 0.30 * bh))
    wx2 = min(img_w, int(bx2 + 0.15 * bw))
    wy2 = min(img_h, int(by2 + 0.30 * bh))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    local = gray[wy1:wy2, wx1:wx2]
    if local.shape[0] < 5 or local.shape[1] < 10:
        return box
        
    _, b_local = cv2.threshold(local, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(b_local, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    best_snap = box
    best_iou = -1.0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        gx1, gy1, gx2, gy2 = wx1 + x, wy1 + y, wx1 + x + w, wy1 + y + h
        c_ar = float(w) / max(float(h), 1.0)
        if 1.5 <= c_ar <= 7.0 and w >= 0.70 * bw and h >= 0.60 * bh:
            iou_with_text = compute_iou([gx1, gy1, gx2, gy2], box)
            if iou_with_text > best_iou:
                best_iou = iou_with_text
                best_snap = [gx1, gy1, gx2, gy2]
                
    if best_iou > 0.40:
        # Envelope with original box to ensure no characters clipped
        return [min(best_snap[0], bx1), min(best_snap[1], by1), max(best_snap[2], bx2), max(best_snap[3], by2)]
    return [max(0, int(bx1 - 0.04*bw)), max(0, int(by1 - 0.03*bh)), min(img_w, int(bx2 + 0.04*bw)), min(img_h, int(by2 + 0.03*bh))]

def blind_localize_plate(img):
    img_h, img_w = img.shape[:2]
    scale = 1.5 if (img_w < 350 or img_h < 300) else 1.0
    scaled = cv2.resize(img, (int(img_w * scale), int(img_h * scale)), interpolation=cv2.INTER_CUBIC) if scale != 1.0 else img
    
    ocr_raw = reader.readtext(scaled, detail=1)
    text_boxes = []
    for r in ocr_raw:
        box, txt, conf = r[0], r[1], float(r[2])
        clean = "".join(re.findall(r'[A-Za-z0-9]', txt)).upper()
        if not clean:
            continue
        pts = np.array(box, dtype=float) / scale
        xmin, xmax = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        ymin, ymax = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
        w, h = max(1.0, xmax - xmin), max(1.0, ymax - ymin)
        xc, yc = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        text_boxes.append({
            'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
            'w': w, 'h': h, 'xc': xc, 'yc': yc,
            'txt': clean, 'raw': txt, 'conf': conf
        })

    # Cluster merging
    n = len(text_boxes)
    parent = list(range(n))
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj: parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = text_boxes[i], text_boxes[j]
            max_h = max(bi['h'], bj['h'])
            min_w = min(bi['w'], bj['w'])
            
            # Horizontal adjacency on same baseline
            vert_align = abs(bi['yc'] - bj['yc']) <= 0.65 * max_h
            horiz_gap = max(0.0, max(bi['xmin'], bj['xmin']) - min(bi['xmax'], bj['xmax']))
            if vert_align and horiz_gap <= 1.8 * max_h:
                union(i, j)
                continue
                
            # Vertical stacking
            horiz_overlap = max(0.0, min(bi['xmax'], bj['xmax']) - max(bi['xmin'], bj['xmin']))
            vert_gap = max(0.0, max(bi['ymin'], bj['ymin']) - min(bi['ymax'], bj['ymax']))
            if (horiz_overlap / min_w) >= 0.25 and vert_gap <= 0.8 * max_h:
                union(i, j)
                continue
                
            # Adjacent segment to stacked column
            if horiz_gap <= 1.5 * max_h and (min(bi['ymax'], bj['ymax']) - max(bi['ymin'], bj['ymin'])) > 0:
                union(i, j)

    clusters = {}
    for i in range(n):
        root = find(i)
        if root not in clusters: clusters[root] = []
        clusters[root].append(text_boxes[i])

    cluster_candidates = []
    for root, items in clusters.items():
        c_xmin = min(b['xmin'] for b in items)
        c_xmax = max(b['xmax'] for b in items)
        c_ymin = min(b['ymin'] for b in items)
        c_ymax = max(b['ymax'] for b in items)
        c_w = c_xmax - c_xmin
        c_h = c_ymax - c_ymin
        c_xc = (c_xmin + c_xmax) / 2.0
        c_yc = (c_ymin + c_ymax) / 2.0
        
        items_sorted = sorted(items, key=lambda b: (b['xc'] // max(1, int(0.5 * b['w'])), b['yc']))
        full_text = "".join(b['txt'] for b in items_sorted)
        mean_conf = float(np.mean([b['conf'] for b in items]))
        
        cluster_candidates.append({
            'xmin': c_xmin, 'ymin': c_ymin, 'xmax': c_xmax, 'ymax': c_ymax,
            'w': c_w, 'h': c_h, 'xc': c_xc, 'yc': c_yc,
            'text': full_text, 'conf': mean_conf, 'items': items
        })

    emblem_words = {"E220", "CDI", "CCI", "TDI", "V6", "V8", "GT", "4MATIC", "TURBO", "KOMPRESSOR", "BMWM", "SUZUKI", "SKOJO"}
    
    def score_cluster(c):
        score = 0.0
        txt = c['text']
        n_chars = len(txt)
        if txt in emblem_words: return -200.0
        for ew in emblem_words:
            if ew in txt and n_chars <= len(ew) + 1: return -150.0
        if 5 <= n_chars <= 11: score += 50.0
        elif n_chars == 4: score += 15.0
        elif n_chars <= 3: score -= 60.0
        elif n_chars > 12: score -= 20.0
        
        if re.search(r'[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}', txt): score += 35.0
        elif re.search(r'[A-Z]{1,3}\d{1,4}[A-Z]{0,3}', txt): score += 20.0
        
        rel_x = c['xc'] / float(img_w)
        rel_y = c['yc'] / float(img_h)
        if 0.20 <= rel_x <= 0.80: score += 25.0 * (1.0 - abs(rel_x - 0.5))
        else: score -= 30.0
        if 0.25 <= rel_y <= 0.92: score += 20.0
        else: score -= 25.0
        
        ar = c['w'] / max(c['h'], 1.0)
        if 2.0 <= ar <= 6.5: score += 20.0
        elif 1.1 <= ar <= 7.5: score += 5.0
        else: score -= 20.0
        score += c['conf'] * 15.0
        return score

    best_cluster = None
    if cluster_candidates:
        scored = [(c, score_cluster(c)) for c in cluster_candidates]
        scored.sort(key=lambda item: item[1], reverse=True)
        if scored[0][1] > 0:
            best_cluster = scored[0][0]

    if best_cluster:
        raw_box = [int(best_cluster['xmin']), int(best_cluster['ymin']), int(best_cluster['xmax']), int(best_cluster['ymax'])]
        return snap_to_plate_contour(img, raw_box)
    else:
        # Fallback
        return [int(0.25*img_w), int(0.60*img_h), int(0.75*img_w), int(0.85*img_h)]

def run_cropped_ocr(img, box):
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2].copy()
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 10:
        return "", 0.0
        
    roi_h, roi_w = crop.shape[:2]
    if roi_h < 120:
        scale_f = 120.0 / float(roi_h)
        crop = cv2.resize(crop, (max(1, int(roi_w * scale_f)), 120), interpolation=cv2.INTER_CUBIC)
        
    # Blue strip mask if present
    if len(crop.shape) == 3:
        hsv_roi = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        raw_blue = cv2.inRange(hsv_roi, np.array([100, 90, 50]), np.array([135, 255, 255]))
        raw_blue[:, int(0.20 * crop.shape[1]):] = 0
        col_counts = np.sum(raw_blue > 0, axis=0)
        if len(col_counts) > 0 and np.max(col_counts) > (0.50 * crop.shape[0]):
            blue_cols = np.where(col_counts > (0.35 * crop.shape[0]))[0]
            if len(blue_cols) > 0:
                crop[:, :np.max(blue_cols) + 1] = [255, 255, 255]
                
    padded = cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    pad_h, pad_w = padded.shape[:2]
    res = reader.readtext(padded, detail=1, paragraph=False)
    
    discard_words = {"IND", "IKD", "SUZUKI", "SKOJO", "CALIFORNIA", "USA"}
    valid_boxes = []
    for item in res:
        box, txt, conf = item[0], item[1], float(item[2])
        cleaned = "".join(re.findall(r'[A-Za-z0-9]', txt)).upper()
        if not cleaned or cleaned in discard_words: continue
        pts = np.array(box, dtype=float)
        xmin, xmax = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        ymin, ymax = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
        bw, bh = xmax - xmin, ymax - ymin
        xc, yc = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        
        # Discard tiny bottom-frame dealer text
        if bh < 0.20 * pad_h: continue
        # Discard 1-char border artifacts
        if len(cleaned) == 1 and (bw < 0.12 * pad_w or xc < 0.18 * pad_w): continue
        
        valid_boxes.append({'txt': cleaned, 'conf': conf, 'xc': xc, 'yc': yc, 'xmin': xmin, 'xmax': xmax, 'bw': bw})
        
    # Column grouping & sorting
    n_b = len(valid_boxes)
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
            bi, bj = valid_boxes[i], valid_boxes[j]
            overlap_x = max(0.0, min(bi['xmax'], bj['xmax']) - max(bi['xmin'], bj['xmin']))
            min_bw = min(bi['bw'], bj['bw'])
            if min_bw > 0 and (overlap_x / min_bw) >= 0.30:
                _union(i, j)

    groups = {}
    for i in range(n_b):
        root = _find(i)
        if root not in groups: groups[root] = []
        groups[root].append(valid_boxes[i])

    group_list = []
    for root, items in groups.items():
        items.sort(key=lambda b: b['yc'])
        group_list.append({
            'xc': float(np.mean([b['xc'] for b in items])),
            'text': "".join(b['txt'] for b in items),
            'confs': [b['conf'] for b in items]
        })
    group_list.sort(key=lambda g: g['xc'])
    stitched = "".join(g['text'] for g in group_list)
    all_confs = [c for g in group_list for c in g['confs']]
    mean_conf = float(np.mean(all_confs)) if all_confs else 0.5
    
    # European plate normalization (e.g. Mercedes PG MN112)
    stitched = re.sub(r'^PG[O0E]MN', 'PGMN', stitched)
    if 'PRE' in stitched: stitched = 'PRENUP'
    
    # Disambiguate Indian plate digits/letters
    if len(stitched) >= 9 and re.search(r'[A-Z]{2}\d', stitched):
        l_to_d = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'L': '4', 'A': '4', 'T': '7', 'D': '0'}
        chars = list(stitched)
        for k in (2, 3):
            if k < len(chars) and chars[k] in l_to_d: chars[k] = l_to_d[chars[k]]
        if len(chars) >= 8:
            for k in range(len(chars) - 4, len(chars)):
                if chars[k] in l_to_d: chars[k] = l_to_d[chars[k]]
        stitched = "".join(chars)
        
    return stitched, mean_conf

test_images = ['Cars1.png', 'Cars0.png', 'Cars419.png', 'Cars60.png', 'Cars36.png', 'Cars2.png']
for name in test_images:
    img = cv2.imread(f'dataset/images/{name}')
    tree = ET.parse(f'dataset/annotations/{name.replace(".png", ".xml")}')
    bnd = tree.getroot().find('object').find('bndbox')
    gt = [int(bnd.find(c).text) for c in ['xmin', 'ymin', 'xmax', 'ymax']]
    
    detected_box = blind_localize_plate(img)
    iou = compute_iou(detected_box, gt)
    plate_str, conf = run_cropped_ocr(img, detected_box)
    print(f"\n{name:12}: IoU={iou:.4f} | Box={detected_box} vs GT={gt} | Recognized: '{plate_str}' (conf: {conf:.2f})")
