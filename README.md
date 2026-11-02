# Automatic License Plate Recognition (ALPR) for Toll & Parking Automation

An academic and industrial-grade monorepo for Automatic License Plate Recognition (ALPR) built around a syllabus-aligned **Digital Image & Video Processing (IVP)** pipeline, paired with a FastAPI backend and a React + Vite + Tailwind + Recharts dashboard.

---

## 🏗 System Architecture

```
IVP/
├── backend/
│   ├── main.py                     # FastAPI application endpoints
│   ├── requirements.txt            # Python dependencies
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py              # Pydantic data contracts
│   └── services/
│       ├── __init__.py
│       ├── ivp_pipeline.py         # 7-Stage IVP Pipeline Engine
│       ├── dataset_service.py      # Pascal VOC XML parser & dataset browser
│       └── toll_service.py         # SQLite toll/parking automation & tariff engine
├── frontend/
│   ├── index.html
│   ├── vite.config.js              # Proxy setup to backend on port 8000
│   ├── tailwind.config.js
│   ├── package.json
│   └── src/
│       ├── App.jsx                 # Tab navigation & global state
│       ├── api/client.js           # Axios REST client
│       └── components/             # Reusable UI & visualization components
│           ├── Header.jsx          # Top bar with latency & plate status
│           ├── ImageUpload.jsx     # Drag-and-drop & dataset selector
│           ├── PipelineConfig.jsx  # Interactive filter & parameter controls
│           ├── PipelineResults.jsx # Summary metrics & StageCard container
│           ├── StageCard.jsx       # Expandable stage details with image zoom
│           ├── HistogramChart.jsx  # Recharts 256-bin area charts
│           ├── TollDashboard.jsx   # Barrier, occupancy, revenue, audit logs
│           └── DatasetBrowser.jsx  # 4-column responsive Pascal VOC browser
├── dataset/
│   ├── images/                     # Kaggle or synthetic PNG/JPG images
│   └── annotations/                # Pascal VOC XML annotations
└── scripts/
    └── generate_sample_dataset.py  # Out-of-the-box synthetic dataset generator
```

---

## 🧪 7-Stage IVP Pipeline Engine (`backend/services/ivp_pipeline.py`)

Every stage exports intermediate base64 PNG visualizations, 256-bin histograms, and exact mathematical metrics:

1. **Stage 1: Image Fundamentals & Color Space Transformation (Unit 1 & 7)**
   - Extracts resolution, bit-depth, file format, and RGB/Grayscale histograms.
   - Converts RGB to Grayscale, HSV, and YUV color spaces.
   - Performs **Color Slicing** in HSV space to classify Indian plate types:
     - 🟨 **Yellow Commercial Plate**: $H \in [20, 35], S > 100, V > 100$
     - ⬜ **White Private Plate**: $S < 40, V > 180$
     - 🟩 **Green EV Plate**: $H \in [35, 85], S > 50, V > 50$
   - Computes **Euclidean ($L_2$)** and **City-Block ($L_1$/$m$-path)** distance transforms rendered as JET heatmaps.

2. **Stage 2: Point Processing & Spatial Enhancement (Unit 2)**
   - Configurable Point Processing:
     - Digital Negative: $s = L - 1 - r$
     - Piecewise-Linear Contrast Stretching
     - Logarithmic Transformation: $s = c \cdot \log(1 + r)$
     - Power-Law (Gamma) Correction: $s = c \cdot r^\gamma$
   - Spatial Filtering: Gaussian Smoothing, Median Filtering (salt-and-pepper noise removal), and Laplacian 2nd-derivative sharpening.
   - Contrast Enhancement: Global Histogram Equalization vs. CLAHE (Contrast-Limited Adaptive Histogram Equalization).
   - Quantitative evaluation: **PSNR** (Peak Signal-to-Noise Ratio) and **MSE** (Mean Squared Error).

3. **Stage 3: Edge Detection Benchmark (Unit 5)**
   - Comparative benchmark comparing:
     1. **Sobel Operator** ($\vert G_x \vert$ vertical emphasis for character stroke density)
     2. **Prewitt Operator**
     3. **Laplacian of Gaussian (LoG)**
     4. **Canny Edge Detector** with dynamic hysteresis thresholds derived from median intensity.

4. **Stage 4: Morphological Processing for Plate ROI Localization (Unit 4)**
   - Uses rectangular Structuring Element ($17 \times 5$) to bridge horizontal character gaps.
   - Morphological Closing ($A \bullet B = (A \oplus B) \ominus B$).
   - Top-Hat ($A - (A \circ B)$) and Black-Hat ($ (A \bullet B) - A $) transforms.
   - Boundary Extraction ($A - (A \ominus B)$) and morphological thinning/skeletonization.

5. **Stage 5: Segmentation & Connected Component Analysis (Unit 6)**
   - 8-connectivity Connected Component Analysis filtering candidate contours by aspect ratio ($2.0 \le AR \le 5.5$), minimum area, and solidity.
   - Within cropped plate ROI, compares:
     - Global Fixed Thresholding
     - Adaptive Gaussian Thresholding
     - **Otsu's Binarization**: Computes optimal threshold $\tau^*$ and maximum between-class variance $\sigma_B^2$.
   - Character segmentation using connected components and vertical projection profiles.

6. **Stage 6: Frequency Domain & Image Compression for Archival (Unit 3)**
   - 2D Discrete Fourier Transform (DFT) magnitude spectrum $\log(1 + |F(u,v)|)$.
   - $8 \times 8$ block **Discrete Cosine Transform (DCT)** compression simulating toll plaza archival:
     - Calculates **Compression Ratio (CR)** and **Relative Data Redundancy ($R_D = 1 - \frac{1}{\text{CR}}$)**.
     - Measures reconstruction PSNR.

7. **Stage 7: Hybrid OCR & Ground-Truth Pascal VOC Evaluation**
   - Text extraction using `EasyOCR` with regex post-processing for Indian registration patterns.
   - Classical Template Matching (Normalized Cross-Correlation, NCC) fallback.
   - Automatic Pascal VOC XML ground truth parsing and **Intersection over Union (IoU)** benchmark:
     - Classical Morphological+Sobel Localization IoU vs. XML ground truth.
     - Enhanced Hybrid Pipeline IoU vs. XML ground truth.

---

## 📹 Unit 8: Digital Video & Motion Estimation Engine (`backend/services/video_motion.py`)

In real toll plaza surveillance, running continuous ALPR on 30 FPS video exhausts compute. The Unit 8 engine implements:

1. **Multi-Frame Ingestion & Sequence Synthesis**:
   - Synthesizes progressive affine approach trajectories ($t=0\dots 3$ approach at $>2.5$ px/frame, $t=4,5$ deceleration/halt at $<1.0$ px/frame).
2. **Temporal Correlation & Frame Classification**:
   - Frame-to-frame Mean Absolute Difference (MAD), Mean Squared Error (MSE), and Pearson correlation $\rho$.
   - **GOP Classification**: I-frame (Intra/Scene cut), P-frame (Forward predicted), B-frame (Bidirectionally interpolated).
3. **Block-Matching Benchmark ($16 \times 16$ Macroblocks, $p=7$)**:
   - **ESBM (Exhaustive Search)**: Evaluates $(2p+1)^2 = 225$ search points per block.
   - **TSS (Three-Step Search)**: Evaluates $9 + 8 + 8 = 25$ search points per block ($\approx 9.0\times$ speedup).
   - **2D-Logarithmic Search**: Evaluates cross patterns with step halving ($\approx 13.2\times$ speedup).
4. **Smart Toll Deceleration Trigger**:
   - Automatically detects when a vehicle enters the Toll Sensor Zone and decelerates ($\Vert \vec{v} \Vert < 1.0$ px/frame), triggering the full 7-stage ALPR pipeline on that optimal keyframe.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### 2. Setup Dataset
Generate the 5 synthetic test images with realistic radiator grilles, headlights, shadows, and Pascal VOC XML annotations:
```bash
python scripts/generate_sample_dataset.py
```
To fetch photographic real-world car samples or connect to Kaggle:
```bash
python scripts/fetch_real_samples.py
# Or for Kaggle: python scripts/fetch_real_samples.py --kaggle
```

### 3. Run Automated Phase 2 Verification Suite
```bash
python scripts/verify_phase2.py
```

### 4. Start Backend Server
```bash
# Run from repository root
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
The backend API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 5. Start Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 📡 API Endpoints

- `POST /api/process-image`: Runs the 7-stage IVP pipeline on an uploaded image or dataset file name.
- `POST /api/process-video-frames`: Ingests video frames, performs Block-Matching Motion Estimation (ESBM/TSS/2D-Log), and triggers ALPR on vehicle halt.
- `GET /api/dataset/list`: Returns all available dataset images, Pascal VOC bounding boxes, and base64 thumbnails.
- `POST /api/toll/simulate-entry`: Simulates gate entry/exit, assigns smart parking bays (`A-01` to `E-10`), and computes tariffs in INR (₹40 EV, ₹85 Private, ₹150 Commercial).
- `GET /api/toll/logs`: Returns toll surveillance logs, occupancy statistics, and revenue breakdown.
