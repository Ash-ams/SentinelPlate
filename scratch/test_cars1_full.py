import cv2
import numpy as np
import easyocr
import xml.etree.ElementTree as ET

reader = easyocr.Reader(['en'], gpu=False)

img = cv2.imread('dataset/images/Cars1.png')
h, w = img.shape[:2]

tree = ET.parse('dataset/annotations/Cars1.xml')
bnd = tree.getroot().find('object').find('bndbox')
gt = [int(bnd.find(c).text) for c in ['xmin', 'ymin', 'xmax', 'ymax']]

print(f"Cars1.png shape: {img.shape}, GT: {gt}")

# Run EasyOCR on full image
res = reader.readtext(img, detail=1)
print(f"EasyOCR on full image ({len(res)} boxes):")
for r in res:
    box, txt, conf = r[0], r[1], float(r[2])
    pts = np.array(box)
    xmin, xmax = np.min(pts[:, 0]), np.max(pts[:, 0])
    ymin, ymax = np.min(pts[:, 1]), np.max(pts[:, 1])
    print(f"   '{txt}' (conf: {conf:.3f}) bbox: [{xmin:.0f}, {ymin:.0f}, {xmax:.0f}, {ymax:.0f}] h={ymax-ymin:.0f} w={xmax-xmin:.0f}")
