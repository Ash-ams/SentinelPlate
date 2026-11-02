# SENTINELPLATE ALPR — EXPERIMENTAL BENCHMARK TABLES
*Empirical benchmark results computed across all annotated vehicle datasets.*

### TABLE I
### Quantitative Evaluation of Spatial Domain Point Processing & Enhancement Algorithms

| Enhancement Method | Mean MSE | Mean PSNR (dB) | Mean Shannon Entropy (bits) |
| :--- | :---: | :---: | :---: |
| Raw Unprocessed | 0.0 | inf (ref) | 5.097 |
| Gamma Correction (gamma=0.6) | 1221.81 | 17.47 dB | 4.813 |
| Global Histogram Equalization | 2572.55 | 16.28 dB | 4.988 |
| Contrast Limited Adaptive Hist Eq (CLAHE) | 151.98 | 30.11 dB | 5.938 |


### TABLE II
### Benchmark of First- & Second-Order Edge Operators for License Plate Localization

| Edge Operator + Morph (17x5) | Mean IoU | Precision | Recall | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| Prewitt Operator | 0.1539 | 0.157 | 0.724 | 9.63 ms |
| Sobel Vertical (|Gx|) | 0.6865 | 0.689 | 0.725 | 5.86 ms |
| Laplacian of Gaussian (LoG) | 0.4615 | 0.484 | 0.976 | 6.06 ms |
| Canny Edge Detector | 0.0611 | 0.061 | 0.727 | 4.55 ms |


### TABLE III
### Character Segmentation & OCR Performance Across Thresholding Paradigms

| Binarization Technique | Optimal Threshold $\tau^*$ | Character Seg. Rate (%) | Plate OCR Accuracy (%) |
| :--- | :---: | :---: | :---: |
| Global Fixed Threshold (tau=127) | 127.0 | 90.9% | 82.3% |
| Local Adaptive Gaussian Thresholding | 127.0 | 91.8% | 80.9% |
| Otsu's Method (with Polarity Auto-Inversion) | 132.8 | 90.9% | 82.3% |


### TABLE IV
### Block-Matching Motion Estimation Efficiency & Prediction Quality (Search Window p = 7)

| Motion Estimation Algorithm | Search Points / MB | Speedup Factor | Prediction PSNR (dB) | Execution Time (ms) |
| :--- | :---: | :---: | :---: | :---: |
| Exhaustive Search Block Matching (ESBM) | 225.0 | 1.00x | 21.56 dB | 68.07 ms |
| Three-Step Search (TSS) | 25.0 | 9.00x | 21.01 dB | 5.92 ms |
| 2D-Logarithmic Search (2D-Log) | 23.1 | 9.70x | 20.95 dB | 5.59 ms |
