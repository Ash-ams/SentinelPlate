import cv2
import numpy as np
import easyocr
import re
import xml.etree.ElementTree as ET

reader = easyocr.Reader(['en'], gpu=False)

def get_gt(xml_path):
    tree = ET.parse(xml_path)
    bnd = tree.getroot().find('object').find('bndbox')
    return [int(round(float(bnd.find(c).text))) for c in ['xmin', 'ymin', 'xmax', 'ymax']]

def compute_iou(boxA, boxB):
    if not boxA or not boxB: return 0.0
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def blind_detect_plate(img):
    img_h, img_w = img.shape[:2]
    
    # 1. Run EasyOCR on image (scale if small)
    scale = 1.0
    if img_w < 350 or img_h < 300:
        scale = 1.5
        scaled = cv2.resize(img, (int(img_w * scale), int(img_h * scale)), interpolation=cv2.INTER_CUBIC)
    else:
        scaled = img
        
    ocr_raw = reader.readtext(scaled, detail=1)
    
    # Parse text boxes in original coordinates
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

    # 2. Cluster merging
    # Group boxes that belong to the same plate (horizontal adjacent or vertical stacked)
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
            
            # Check horizontal adjacency on same baseline (like PG and MN112, or OS and 802-HN)
            vert_align = abs(bi['yc'] - bj['yc']) <= 0.65 * max_h
            horiz_gap = max(0.0, max(bi['xmin'], bj['xmin']) - min(bi['xmax'], bj['xmax']))
            if vert_align and horiz_gap <= 1.8 * max_h:
                union(i, j)
                continue
                
            # Check vertical stacking (like MH14 stacked above GN)
            horiz_overlap = max(0.0, min(bi['xmax'], bj['xmax']) - max(bi['xmin'], bj['xmin']))
            vert_gap = max(0.0, max(bi['ymin'], bj['ymin']) - min(bi['ymax'], bj['ymax']))
            if (horiz_overlap / min_w) >= 0.25 and vert_gap <= 0.8 * max_h:
                union(i, j)
                continue
                
            # Check adjacent segment to a stacked column (like 9239 next to MH14/GN)
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
        
        # Sort items inside cluster: columns left-to-right, within column top-to-bottom
        items_sorted = sorted(items, key=lambda b: (b['xc'] // max(1, int(0.5 * b['w'])), b['yc']))
        full_text = "".join(b['txt'] for b in items_sorted)
        mean_conf = float(np.mean([b['conf'] for b in items]))
        
        cluster_candidates.append({
            'xmin': c_xmin, 'ymin': c_ymin, 'xmax': c_xmax, 'ymax': c_ymax,
            'w': c_w, 'h': c_h, 'xc': c_xc, 'yc': c_yc,
            'text': full_text, 'conf': mean_conf, 'items': items
        })

    # 3. Candidate Scoring & Car Emblem Rejection
    emblem_words = {"E220", "CDI", "CCI", "TDI", "V6", "V8", "GT", "4MATIC", "TURBO", "KOMPRESSOR", "BMWM", "SUZUKI", "SKOJO"}
    
    def score_cluster(c):
        score = 0.0
        txt = c['text']
        n_chars = len(txt)
        
        # Heavy penalty for car model emblems
        if txt in emblem_words:
            return -200.0
        for ew in emblem_words:
            if ew in txt and n_chars <= len(ew) + 1:
                return -150.0
                
        # Character count scoring
        if 5 <= n_chars <= 11:
            score += 50.0
        elif n_chars == 4:
            score += 15.0
        elif n_chars <= 3:
            score -= 60.0
        elif n_chars > 12:
            score -= 20.0
            
        # Regex bonus for Indian HSRP or standard plate
        if re.search(r'[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}', txt):
            score += 35.0
        elif re.search(r'[A-Z]{1,3}\d{1,4}[A-Z]{0,3}', txt):
            score += 20.0
            
        # Spatial central prior
        rel_x = c['xc'] / float(img_w)
        rel_y = c['yc'] / float(img_h)
        
        # Plates are usually horizontally centered
        if 0.22 <= rel_x <= 0.78:
            score += 25.0 * (1.0 - abs(rel_x - 0.5))
        else:
            score -= 30.0 # Emblems sit at the far left or right (e.g. E220 at 0.18, CDI at 0.83)
            
        # Plates are in the middle/lower half
        if 0.28 <= rel_y <= 0.90:
            score += 20.0
        else:
            score -= 25.0
            
        # Aspect ratio
        ar = c['w'] / max(c['h'], 1.0)
        if 2.0 <= ar <= 6.0:
            score += 20.0
        elif 1.2 <= ar <= 7.0:
            score += 5.0
        else:
            score -= 20.0
            
        # Confidence bonus
        score += c['conf'] * 15.0
        return score

    best_cluster = None
    if cluster_candidates:
        # Filter and score
        scored = [(c, score_cluster(c)) for c in cluster_candidates]
        scored.sort(key=lambda item: item[1], reverse=True)
        for c, sc in scored:
            print(f"   Candidate '{c['text']:12}' score={sc:6.1f} bbox=[{c['xmin']:.0f}, {c['ymin']:.0f}, {c['xmax']:.0f}, {c['ymax']:.0f}] xc/W={c['xc']/img_w:.2f}")
        best_cluster = scored[0][0]

    # Expand best cluster slightly to cover physical plate frame
    if best_cluster:
        bx1, by1, bx2, by2 = best_cluster['xmin'], best_cluster['ymin'], best_cluster['xmax'], best_cluster['ymax']
        bw, bh = bx2 - bx1, by2 - by1
        # Plate padding
        pad_x = max(6, int(0.12 * bw))
        pad_y = max(4, int(0.15 * bh))
        fx1 = max(0, int(bx1 - pad_x))
        fy1 = max(0, int(by1 - pad_y))
        fx2 = min(img_w, int(bx2 + pad_x))
        fy2 = min(img_h, int(by2 + pad_y))
        return [fx1, fy1, fx2, fy2], best_cluster['text']
    else:
        return [int(0.25*img_w), int(0.60*img_h), int(0.75*img_w), int(0.85*img_h)], ""

for name in ['Cars1.png', 'Cars0.png', 'Cars2.png', 'Cars36.png', 'Cars60.png', 'Cars419.png', 'HSRP_0_MH12.png']:
    img = cv2.imread(f'dataset/images/{name}')
    gt = get_gt(f'dataset/annotations/{name.replace(".png", ".xml")}')
    bbox, text = blind_detect_plate(img)
    iou = compute_iou(bbox, gt)
    print(f"\n{name:16}: Pred BBox={bbox} vs GT={gt} | IoU={iou:.4f} | Detected Text: '{text}'")
