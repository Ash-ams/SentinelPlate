#!/usr/bin/env python3
"""Generate challenging vehicle plate test cases:
1. NightCar.png (Low-light night exposure, dark asphalt, headlight glow)
2. GlareCar.png (High-glare daytime overexposure with direct specular wash)
3. AngledCar.png (Angled perspective plate with realistic Gaussian sensor noise)

Output: dataset/images/ and dataset/annotations/
"""
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import xml.etree.ElementTree as ET
from xml.dom import minidom

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")

WIDTH = 640
HEIGHT = 480


def generate_voc_xml(filename, width, height, bndbox):
    root = ET.Element("annotation")
    folder = ET.SubElement(root, "folder")
    folder.text = "images"
    fname = ET.SubElement(root, "filename")
    fname.text = filename
    size = ET.SubElement(root, "size")
    w = ET.SubElement(size, "width")
    w.text = str(width)
    h = ET.SubElement(size, "height")
    h.text = str(height)
    d = ET.SubElement(size, "depth")
    d.text = "3"
    obj = ET.SubElement(root, "object")
    name = ET.SubElement(obj, "name")
    name.text = "licence"
    b_elem = ET.SubElement(obj, "bndbox")
    xmin = ET.SubElement(b_elem, "xmin")
    xmin.text = str(int(bndbox["xmin"]))
    ymin = ET.SubElement(b_elem, "ymin")
    ymin.text = str(int(bndbox["ymin"]))
    xmax = ET.SubElement(b_elem, "xmax")
    xmax.text = str(int(bndbox["xmax"]))
    ymax = ET.SubElement(b_elem, "ymax")
    ymax.text = str(int(bndbox["ymax"]))
    xml_str = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(xml_str)
    return parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def get_font(size=19):
    for f in ["C:\\Windows\\Fonts\\arialbd.ttf", "C:\\Windows\\Fonts\\arial.ttf", "arialbd.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def render_characters(draw, text, plate_x1, plate_y1, plate_w, plate_h, fg_color, ind_w=18):
    font = get_font(19)
    left_m = 8
    right_m = 8
    t_start = plate_x1 + ind_w + left_m
    t_end = plate_x1 + plate_w - right_m
    avail_w = t_end - t_start

    char_widths = []
    char_heights = []
    for ch in text:
        if hasattr(draw, "textbbox"):
            b = draw.textbbox((0, 0), ch, font=font)
            cw, ch_h = b[2] - b[0], b[3] - b[1]
        else:
            cw, ch_h = 10, 14
        char_widths.append(cw)
        char_heights.append(ch_h)

    total_w = sum(char_widths)
    gap = max(2.5, (avail_w - total_w) / max(len(text) - 1, 1))
    cur_x = t_start + max(0, (avail_w - (total_w + (len(text) - 1) * gap)) / 2.0)

    for i, ch in enumerate(text):
        cy = plate_y1 + (plate_h - char_heights[i]) // 2 - 1
        draw.text((int(round(cur_x)), cy), ch, fill=fg_color, font=font)
        cur_x += char_widths[i] + gap


def create_night_car():
    """Challenging low-light underexposed night scene."""
    img = Image.new("RGB", (WIDTH, HEIGHT), (12, 14, 18))
    draw = ImageDraw.Draw(img)

    # Road surface in darkness
    draw.rectangle([0, int(HEIGHT * 0.55), WIDTH, HEIGHT], fill=(22, 24, 28))

    # Headlight glow cones (soft additive lighting)
    glow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(glow)
    g_draw.polygon([(180, 270), (40, 480), (320, 480)], fill=(255, 250, 220, 45))
    g_draw.polygon([(460, 270), (320, 480), (600, 480)], fill=(255, 250, 220, 45))
    glow = glow.filter(ImageFilter.GaussianBlur(12))
    img.paste(glow, (0, 0), glow)

    # Car body (dark silhouette)
    draw = ImageDraw.Draw(img)
    car_x, car_y, car_w, car_h = 160, 240, 320, 150
    draw.rounded_rectangle([car_x, car_y + 40, car_x + car_w, car_y + car_h], radius=10, fill=(35, 38, 44), outline=(18, 20, 22), width=2)
    # Headlamps ON
    draw.rounded_rectangle([car_x + 15, car_y + 55, car_x + 55, car_y + 75], radius=4, fill=(255, 255, 230))
    draw.rounded_rectangle([car_x + car_w - 55, car_y + 55, car_x + car_w - 15, car_y + 75], radius=4, fill=(255, 255, 230))
    # Grille slats
    for gy in range(car_y + 55, car_y + 80, 5):
        draw.line([(car_x + 65, gy), (car_x + car_w - 65, gy)], fill=(75, 78, 85), width=1)

    # Plate (White private illuminated by bumper light)
    plate_w, plate_h = 172, 36
    px1 = car_x + (car_w - plate_w) // 2
    py1 = car_y + car_h - 40
    px2 = px1 + plate_w
    py2 = py1 + plate_h
    draw.rectangle([px1, py1, px2, py2], fill=(230, 232, 238), outline=(10, 10, 10), width=2)
    # IND strip
    draw.rectangle([px1 + 2, py1 + 2, px1 + 18, py2 - 2], fill=(20, 55, 140))
    render_characters(draw, "KA05MH9988", px1, py1, plate_w, plate_h, (15, 15, 20))

    # Sensor noise
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 4.5, arr.shape)
    img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    bndbox = {"xmin": px1, "ymin": py1, "xmax": px2, "ymax": py2}
    return img, bndbox, "KA05MH9988"


def create_glare_car():
    """Challenging high-glare daytime scene with specular wash on plate."""
    img = Image.new("RGB", (WIDTH, HEIGHT), (220, 235, 250))
    draw = ImageDraw.Draw(img)

    # Sky & bright asphalt
    draw.rectangle([0, int(HEIGHT * 0.55), WIDTH, HEIGHT], fill=(160, 165, 170))

    # Bright silver vehicle
    car_x, car_y, car_w, car_h = 145, 235, 330, 155
    draw.rounded_rectangle([car_x, car_y + 45, car_x + car_w, car_y + car_h], radius=12, fill=(215, 218, 225), outline=(110, 115, 120), width=2)
    # Headlights
    draw.rounded_rectangle([car_x + 12, car_y + 55, car_x + 55, car_y + 78], radius=4, fill=(245, 248, 255), outline=(150, 155, 160))
    draw.rounded_rectangle([car_x + car_w - 55, car_y + 55, car_x + car_w - 12, car_y + 78], radius=4, fill=(245, 248, 255), outline=(150, 155, 160))
    # Grille
    for gy in range(car_y + 56, car_y + 80, 5):
        draw.line([(car_x + 65, gy), (car_x + car_w - 65, gy)], fill=(130, 135, 140), width=1)

    # Yellow Commercial Plate under midday sun
    plate_w, plate_h = 172, 36
    px1 = car_x + (car_w - plate_w) // 2
    py1 = car_y + car_h - 42
    px2 = px1 + plate_w
    py2 = py1 + plate_h
    draw.rectangle([px1, py1, px2, py2], fill=(255, 215, 20), outline=(20, 20, 20), width=2)
    draw.rectangle([px1 + 2, py1 + 2, px1 + 18, py2 - 2], fill=(20, 60, 160))
    render_characters(draw, "HR26BC4411", px1, py1, plate_w, plate_h, (10, 10, 12))

    # Specular Sun Glare diagonal streak across plate
    glare = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(glare)
    g_draw.polygon([(px1 - 20, py2 + 10), (px1 + 40, py1 - 10), (px1 + 85, py1 - 10), (px1 + 25, py2 + 10)], fill=(255, 255, 255, 140))
    glare = glare.filter(ImageFilter.GaussianBlur(3))
    img.paste(glare, (0, 0), glare)

    bndbox = {"xmin": px1, "ymin": py1, "xmax": px2, "ymax": py2}
    return img, bndbox, "HR26BC4411"


def create_angled_car():
    """Slight perspective tilted vehicle plate with texture."""
    img = Image.new("RGB", (WIDTH, HEIGHT), (140, 175, 215))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, int(HEIGHT * 0.58), WIDTH, HEIGHT], fill=(62, 65, 70))

    # Red vehicle approaching from slight left angle
    car_x, car_y, car_w, car_h = 150, 240, 325, 155
    draw.rounded_rectangle([car_x, car_y + 40, car_x + car_w, car_y + car_h], radius=12, fill=(168, 30, 35), outline=(25, 25, 28), width=2)
    for gy in range(car_y + 54, car_y + 80, 5):
        draw.line([(car_x + 65, gy), (car_x + car_w - 65, gy)], fill=(40, 42, 45), width=1)

    # Green EV Plate
    plate_w, plate_h = 172, 36
    px1 = car_x + (car_w - plate_w) // 2 + 5
    py1 = car_y + car_h - 40
    px2 = px1 + plate_w
    py2 = py1 + plate_h
    draw.rectangle([px1, py1, px2, py2], fill=(16, 135, 45), outline=(10, 10, 10), width=2)
    draw.rectangle([px1 + 2, py1 + 2, px1 + 18, py2 - 2], fill=(22, 65, 168))
    render_characters(draw, "TS08EE2026", px1, py1, plate_w, plate_h, (250, 250, 255))

    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 3.2, arr.shape)
    img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    bndbox = {"xmin": px1, "ymin": py1, "xmax": px2, "ymax": py2}
    return img, bndbox, "TS08EE2026"


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

    generators = [
        ("NightCar.png", "NightCar.xml", create_night_car),
        ("GlareCar.png", "GlareCar.xml", create_glare_car),
        ("AngledCar.png", "AngledCar.xml", create_angled_car),
    ]

    for img_name, xml_name, gen_func in generators:
        img, bndbox, plate_str = gen_func()
        img_path = os.path.join(IMAGES_DIR, img_name)
        xml_path = os.path.join(ANNOTATIONS_DIR, xml_name)

        img.save(img_path, format="PNG")
        xml_str = generate_voc_xml(img_name, WIDTH, HEIGHT, bndbox)
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_str)
        print(f"Generated challenging case: {img_name} ({plate_str}) | BBox: {bndbox}")

    print("Challenging cases successfully added to dataset.")


if __name__ == "__main__":
    main()
