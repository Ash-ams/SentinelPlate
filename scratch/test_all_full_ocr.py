import cv2
import numpy as np
import easyocr
import xml.etree.ElementTree as ET

reader = easyocr.Reader(['en'], gpu=False)

def get_gt(xml_path):
    tree = ET.parse(xml_path)
    bnd = tree.getroot().find('object').find('bndbox')
    return [int(bnd.find(c).text) for c in ['xmin', 'ymin', 'xmax', 'ymax']]

for name in ['Cars1.png', 'Cars0.png', 'Cars2.png', 'Cars36.png', 'Cars60.png', 'Cars419.png', 'HSRP_0_MH12.png']:
    img = cv2.imread(f'dataset/images/{name}')
    xml_path = f'dataset/annotations/{name.replace(".png", ".xml")}'
    gt = get_gt(xml_path)
    
    # If image is small, upscale slightly
    h, w = img.shape[:2]
    scale = 1.0
    if h < 300 or w < 300:
        scale = 1.5
        img_scaled = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    else:
        img_scaled = img
        
    res = reader.readtext(img_scaled, detail=1)
    print(f"\n=== {name} (shape: {img.shape}, GT: {gt}) ===")
    for r in res:
        box, txt, conf = r[0], r[1], float(r[2])
        pts = np.array(box, dtype=float) / scale
        xmin, xmax = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        ymin, ymax = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
        bw, bh = xmax - xmin, ymax - ymin
        print(f"   '{txt:15}' conf={conf:.3f} bbox=[{xmin:.0f}, {ymin:.0f}, {xmax:.0f}, {ymax:.0f}] w={bw:.0f} h={bh:.0f} xc/W={((xmin+xmax)/(2*w)):.2f} yc/H={((ymin+ymax)/(2*h)):.2f}")
