#!/usr/bin/env python3
"""Generate 5 synthetic car license plate test images with Pascal VOC XML annotations.

Realistic vehicle features:
- Radiator grille with high-frequency horizontal and vertical bars
- Multi-chamber headlights with lenses, reflectors, and turn indicators
- Realistic ambient underbody shadow gradient on asphalt road
- License plate with IND blue strip, guaranteed inter-character kerning,
  and ample border margins so characters never collide with edges.

Usage: python scripts/generate_sample_dataset.py
Output: dataset/images/Cars0.png ... Cars4.png
        dataset/annotations/Cars0.xml ... Cars4.xml
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "annotations")

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Sample plate numbers (Indian format - 10 characters)
PLATE_NUMBERS = [
    "MH12DE1433",
    "KA01AB1234",
    "DL3CAB5678",
    "TN09CD9012",
    "GJ05EF3456"
]

# Plate types with their background colors
PLATE_TYPES = [
    {"bg": (250, 250, 252), "fg": (12, 12, 14), "type": "white_private", "name": "White Private"},
    {"bg": (252, 210, 15), "fg": (12, 12, 14), "type": "yellow_commercial", "name": "Yellow Commercial"},
    {"bg": (18, 138, 48), "fg": (252, 252, 254), "type": "green_ev", "name": "Green EV"},
    {"bg": (250, 250, 252), "fg": (12, 12, 14), "type": "white_private", "name": "White Private"},
    {"bg": (252, 210, 15), "fg": (12, 12, 14), "type": "yellow_commercial", "name": "Yellow Commercial"},
]

# Distinct vehicle body colors
CAR_COLORS = [
    {"body": (36, 85, 160), "roof": (24, 60, 115), "glass": (160, 195, 225)},    # Royal Blue
    {"body": (175, 32, 32), "roof": (130, 22, 22), "glass": (170, 190, 210)},    # Crimson Red
    {"body": (168, 172, 180), "roof": (132, 136, 145), "glass": (185, 205, 220)},# Silver Metallic
    {"body": (32, 34, 38), "roof": (20, 22, 25), "glass": (135, 155, 178)},      # Charcoal Black
    {"body": (22, 105, 58), "roof": (15, 75, 42), "glass": (155, 190, 205)},     # Forest Green
]


def get_plate_font(target_size=20):
    """Load TrueType font suitable for license plate rendering with clean glyphs."""
    font_candidates = [
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\calibrib.ttf",
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "arialbd.ttf",
        "arial.ttf",
    ]
    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, target_size)
        except (IOError, OSError):
            continue
    try:
        return ImageFont.load_default(size=target_size)
    except TypeError:
        return ImageFont.load_default()


def get_text_dimensions(text, font, draw):
    """Measure text width and height."""
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    elif hasattr(font, "getsize"):
        return font.getsize(text)
    elif hasattr(draw, "textsize"):
        return draw.textsize(text, font=font)
    return len(text) * 10, 16


def create_background(width, height):
    """Draw gradient sky on top 58% and textured asphalt road on bottom 42%."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    sky_split = int(height * 0.58)

    # Sky gradient: light blue to horizon soft haze
    for y in range(sky_split):
        ratio = y / max(sky_split - 1, 1)
        r = int(115 * (1 - ratio) + 185 * ratio)
        g = int(170 * (1 - ratio) + 215 * ratio)
        b = int(225 * (1 - ratio) + 245 * ratio)
        draw.line([(0, y), (width - 1, y)], fill=(r, g, b))

    # Road: asphalt dark gray (58, 60, 64) with perspective texture
    road_color = (58, 60, 64)
    draw.rectangle([0, sky_split, width, height], fill=road_color)

    # Horizon curb line
    draw.line([(0, sky_split), (width - 1, sky_split)], fill=(110, 112, 115), width=2)
    draw.line([(0, sky_split + 2), (width - 1, sky_split + 2)], fill=(45, 47, 50), width=1)

    # Road lane markings
    dash_y = int(sky_split + (height - sky_split) * 0.72)
    for x in range(15, width, 65):
        draw.line([(x, dash_y), (x + 38, dash_y)], fill=(225, 225, 230), width=4)

    return img


def draw_car_with_plate(base_img, car_idx, plate_number, plate_info):
    """Draw a detailed vehicle front with radiator grille, headlights, shadow, and plate."""
    colors = CAR_COLORS[car_idx % len(CAR_COLORS)]

    car_w = 330
    car_h = 160

    horiz_offsets = [50, 175, 95, 235, 125]
    car_x = horiz_offsets[car_idx % len(horiz_offsets)]
    car_y = 238

    # 1. Underbody Ambient Shadow Gradient
    # Create separate shadow layer for realistic diffusion
    shadow_img = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(shadow_img)
    shadow_box = [
        car_x - 12,
        car_y + car_h + 4,
        car_x + car_w + 12,
        car_y + car_h + 30
    ]
    s_draw.ellipse(shadow_box, fill=(15, 16, 18, 185))
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=7))
    base_img.paste(shadow_img, (0, 0), shadow_img)

    draw = ImageDraw.Draw(base_img)

    # 2. Wheels (shadow + tire + alloy rim)
    wheel_radius = 24
    wheel_y = car_y + car_h + 6
    left_wheel_x = car_x + 52
    right_wheel_x = car_x + car_w - 52

    for wx in (left_wheel_x, right_wheel_x):
        # Outer tire
        draw.ellipse(
            [wx - wheel_radius, wheel_y - wheel_radius, wx + wheel_radius, wheel_y + wheel_radius],
            fill=(22, 23, 25),
            outline=(10, 10, 12),
            width=2,
        )
        # Inner rim
        rim_r = wheel_radius - 8
        draw.ellipse(
            [wx - rim_r, wheel_y - rim_r, wx + rim_r, wheel_y + rim_r],
            fill=(175, 180, 185),
            outline=(95, 100, 105),
            width=2,
        )
        # Center hub
        draw.ellipse(
            [wx - 4, wheel_y - 4, wx + 4, wheel_y + 4],
            fill=(45, 48, 52)
        )

    # 3. Cabin / Roof (Trapezoid with aerodynamic taper)
    roof_top_y = car_y
    roof_bottom_y = car_y + int(car_h * 0.44)
    cabin_points = [
        (car_x + 58, roof_bottom_y),
        (car_x + 92, roof_top_y),
        (car_x + car_w - 92, roof_top_y),
        (car_x + car_w - 58, roof_bottom_y),
    ]
    draw.polygon(cabin_points, fill=colors["roof"])

    # Cabin Windshield Glass
    window_points = [
        (car_x + 66, roof_bottom_y - 3),
        (car_x + 98, roof_top_y + 5),
        (car_x + car_w - 98, roof_top_y + 5),
        (car_x + car_w - 66, roof_bottom_y - 3),
    ]
    draw.polygon(window_points, fill=colors["glass"])

    # Center Windshield Rearview mirror & B-pillar divider
    draw.rectangle(
        [car_x + car_w // 2 - 3, roof_top_y + 5, car_x + car_w // 2 + 3, roof_bottom_y - 3],
        fill=colors["roof"]
    )
    draw.rectangle(
        [car_x + car_w // 2 - 8, roof_top_y + 7, car_x + car_w // 2 + 8, roof_top_y + 14],
        fill=(30, 32, 35)
    )

    # 4. Lower Vehicle Body & Hood
    body_y1 = car_y + int(car_h * 0.40)
    body_y2 = car_y + car_h
    draw.rounded_rectangle(
        [car_x, body_y1, car_x + car_w, body_y2],
        radius=12,
        fill=colors["body"],
        outline=(18, 18, 20),
        width=2,
    )

    # Hood crease lines (subtle perspective accents)
    draw.line([(car_x + 95, body_y1 + 4), (car_x + 105, body_y1 + 28)], fill=(255, 255, 255, 60), width=1)
    draw.line([(car_x + car_w - 95, body_y1 + 4), (car_x + car_w - 105, body_y1 + 28)], fill=(255, 255, 255, 60), width=1)

    # 5. Multi-Chamber Headlight Units
    hl_w = 42
    hl_h = 24
    hl_y = body_y1 + 10

    # Left Headlight
    hl_left_x = car_x + 10
    draw.rounded_rectangle([hl_left_x, hl_y, hl_left_x + hl_w, hl_y + hl_h], radius=5, fill=(235, 240, 245), outline=(30, 30, 35), width=2)
    # Reflector inner projector
    draw.ellipse([hl_left_x + 18, hl_y + 4, hl_left_x + 36, hl_y + 20], fill=(210, 225, 245), outline=(140, 160, 185), width=1)
    # Amber indicator lens
    draw.rectangle([hl_left_x + 4, hl_y + 5, hl_left_x + 14, hl_y + 19], fill=(245, 165, 20))

    # Right Headlight
    hl_right_x = car_x + car_w - 10 - hl_w
    draw.rounded_rectangle([hl_right_x, hl_y, hl_right_x + hl_w, hl_y + hl_h], radius=5, fill=(235, 240, 245), outline=(30, 30, 35), width=2)
    # Reflector inner projector
    draw.ellipse([hl_right_x + 6, hl_y + 4, hl_right_x + 24, hl_y + 20], fill=(210, 225, 245), outline=(140, 160, 185), width=1)
    # Amber indicator lens
    draw.rectangle([hl_right_x + hl_w - 14, hl_y + 5, hl_right_x + hl_w - 4, hl_y + 19], fill=(245, 165, 20))

    # 6. Radiator Grille (Key visual feature: tests vertical edge filtering & aspect ratio)
    grille_x1 = hl_left_x + hl_w + 12
    grille_x2 = hl_right_x - 12
    grille_y1 = hl_y + 2
    grille_y2 = grille_y1 + 26
    # Grille housing
    draw.rounded_rectangle([grille_x1, grille_y1, grille_x2, grille_y2], radius=4, fill=(18, 20, 22), outline=(140, 145, 150), width=1)
    # Horizontal grille slats (chrome bars)
    for gy in range(grille_y1 + 5, grille_y2 - 2, 5):
        draw.line([(grille_x1 + 4, gy), (grille_x2 - 4, gy)], fill=(180, 185, 190), width=1)
    # Vertical grille mesh bars (generates dense vertical Sobel gradients)
    for gx in range(grille_x1 + 10, grille_x2 - 6, 9):
        draw.line([(gx, grille_y1 + 3), (gx, grille_y2 - 3)], fill=(50, 52, 55), width=1)
    # Center Brand Emblem badge
    cx = (grille_x1 + grille_x2) // 2
    cy = (grille_y1 + grille_y2) // 2
    draw.ellipse([cx - 7, cy - 6, cx + 7, cy + 6], fill=(195, 200, 205), outline=(60, 62, 65), width=1)

    # 7. Lower Bumper Section
    bumper_y1 = body_y2 - 38
    bumper_y2 = body_y2 - 4
    draw.rounded_rectangle(
        [car_x + 14, bumper_y1, car_x + car_w - 14, bumper_y2],
        radius=5,
        fill=(38, 40, 44),
        outline=(18, 18, 20),
        width=2
    )

    # 8. License Plate with Guaranteed Kerning & Border Clearance
    # Standard dimensions: 172 x 36 px (Aspect ratio ~4.77, ideal for Indian HSRP)
    plate_w = 172
    plate_h = 36
    plate_x1 = car_x + (car_w - plate_w) // 2
    plate_y1 = bumper_y1 + (bumper_y2 - bumper_y1 - plate_h) // 2
    plate_x2 = plate_x1 + plate_w
    plate_y2 = plate_y1 + plate_h

    # Plate outer border (clean 2px black border)
    draw.rectangle([plate_x1, plate_y1, plate_x2, plate_y2], fill=plate_info["bg"], outline=(10, 10, 10), width=2)

    # Blue IND strip on left side (18 px wide)
    ind_width = 18
    draw.rectangle([plate_x1 + 2, plate_y1 + 2, plate_x1 + ind_width, plate_y2 - 2], fill=(22, 65, 168))
    # Small white "IND" text / emblem dot
    draw.ellipse([plate_x1 + 7, plate_y1 + 8, plate_x1 + 13, plate_y1 + 14], fill=(255, 255, 255))
    try:
        ind_font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 7)
        draw.text((plate_x1 + 3, plate_y1 + 18), "IND", fill=(255, 255, 255), font=ind_font)
    except Exception:
        pass

    # Character Rendering with Guaranteed Kerning:
    # Target 10 characters (e.g. MH12DE1433)
    # Available width for text = plate_w - ind_width - margins
    left_text_margin = 8
    right_text_margin = 8
    text_area_start = plate_x1 + ind_width + left_text_margin
    text_area_end = plate_x2 - right_text_margin
    avail_text_w = text_area_end - text_area_start  # 172 - 18 - 8 - 8 = 138 px

    font = get_plate_font(target_size=19)
    num_chars = len(plate_number)

    # Measure character widths and compute pitch
    char_widths = []
    char_heights = []
    for ch in plate_number:
        cw, ch_h = get_text_dimensions(ch, font, draw)
        char_widths.append(cw)
        char_heights.append(ch_h)

    total_glyph_w = sum(char_widths)
    # Remaining space distributed evenly between characters (9 gaps)
    remaining_space = avail_text_w - total_glyph_w
    inter_char_gap = max(2.5, remaining_space / max(num_chars - 1, 1))

    # Center the entire text block inside the text area
    actual_text_block_w = total_glyph_w + (num_chars - 1) * inter_char_gap
    start_x = text_area_start + max(0, (avail_text_w - actual_text_block_w) / 2.0)

    # Draw each character individually with explicit spacing
    curr_x = start_x
    for i, ch in enumerate(plate_number):
        ch_w = char_widths[i]
        ch_h = char_heights[i]
        char_y = plate_y1 + (plate_h - ch_h) // 2 - 1
        draw.text((int(round(curr_x)), char_y), ch, fill=plate_info["fg"], font=font)
        curr_x += ch_w + inter_char_gap

    # 9. Realistic Ambient Sensor Noise
    arr = np.array(base_img, dtype=np.float32)
    noise = np.random.normal(loc=0.0, scale=2.8, size=arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    final_img = Image.fromarray(noisy_arr)

    bndbox = {
        "xmin": int(plate_x1),
        "ymin": int(plate_y1),
        "xmax": int(plate_x2),
        "ymax": int(plate_y2),
    }

    return final_img, bndbox


def generate_voc_xml(filename, width, height, bndbox):
    """Generate clean Pascal VOC XML string with licence bndbox."""
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
    xmin.text = str(bndbox["xmin"])
    ymin = ET.SubElement(b_elem, "ymin")
    ymin.text = str(bndbox["ymin"])
    xmax = ET.SubElement(b_elem, "xmax")
    xmax.text = str(bndbox["xmax"])
    ymax = ET.SubElement(b_elem, "ymax")
    ymax.text = str(bndbox["ymax"])

    xml_str = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(xml_str)
    return parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

    print(f"Generating synthetic ALPR dataset into: {OUTPUT_DIR}")
    print(f"Target count: {len(PLATE_NUMBERS)} samples")
    print("-" * 60)

    np.random.seed(42)
    random.seed(42)

    generated_files = []

    for i in range(len(PLATE_NUMBERS)):
        plate_text = PLATE_NUMBERS[i]
        plate_type = PLATE_TYPES[i]
        base_name = f"Cars{i}"
        img_filename = f"{base_name}.png"
        xml_filename = f"{base_name}.xml"

        # 1. Generate base scene
        bg = create_background(IMAGE_WIDTH, IMAGE_HEIGHT)

        # 2. Draw car with plate and realistic features
        car_img, bndbox = draw_car_with_plate(bg, i, plate_text, plate_type)

        # 3. Save PNG image
        img_path = os.path.join(IMAGES_DIR, img_filename)
        car_img.save(img_path, format="PNG")

        # 4. Generate & Save VOC XML
        xml_content = generate_voc_xml(img_filename, IMAGE_WIDTH, IMAGE_HEIGHT, bndbox)
        xml_path = os.path.join(ANNOTATIONS_DIR, xml_filename)
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

        generated_files.append((img_filename, xml_filename, plate_text, plate_type["name"], bndbox))
        print(f"[{i+1}/5] Generated {img_filename} & {xml_filename} | Plate: {plate_text} ({plate_type['name']}) | BBox: {bndbox}")

    print("-" * 60)
    print(f"Successfully generated {len(generated_files)} synthetic samples with Pascal VOC XML ground truth.")


if __name__ == "__main__":
    main()
