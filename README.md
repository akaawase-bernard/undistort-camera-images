# Image Undistortion Toolbox

A powerful Python toolbox for undistorting images using camera calibration data. This tool simplifies the process of correcting lens distortion by providing configurable options for batch processing, ROI cropping, and visualization. Ideal for photogrammetry, computer vision, and image preprocessing tasks.

---

## Features

- Load camera intrinsics and distortion coefficients from XML files.
- Undistort single or batch images using OpenCV.
- Optional cropping of undistorted images based on the region of interest (ROI).
- Visualization: Side-by-side comparison of original and undistorted images.
- Support for multiple image formats (e.g., `bmp`, `jpg`, `png`).
- Highly configurable with various flags and parameters.

---

## Requirements

- Python 3.7+
- Libraries:
  - `opencv-python`
  - `numpy`
  - `matplotlib`

Install the required libraries via pip:
```bash
pip install opencv-python numpy matplotlib

