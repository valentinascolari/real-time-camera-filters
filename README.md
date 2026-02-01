# 📷 Real-Time Webcam Editor

> A Python tool to apply visual effects, filters and geometric transformations to a live webcam feed.

This project is a Computer Vision sandbox built with **Python** and **OpenCV**. It captures video from your webcam and allows you to apply various image processing algorithms in real-time, controlled via keyboard shortcuts and an adjustable intensity trackbar.

## 🚀 Features

* **Live Preview:** View the original feed and the processed feed side-by-side.
* **Smoothing & Edge Detection:** Gaussian Blur, Canny, and Sobel operators.
* **Photometric Adjustments:** Real-time control of Brightness, Contrast, and Negative mode.
* **Color Manipulation:** Variable Grayscale blending (interpolation between color and B&W).
* **Geometric Transformations:** Resize, Rotate, and Mirror/Flip the video feed.
* **Video Recording:** Save the processed output to an `.avi` file.
* **Dynamic Control:** Use the slide bar to adjust filter intensity (kernels, thresholds, alpha/beta values).

## 🛠️ Installation & Requirements

Ensure you have Python installed on your system. This project relies on `opencv-python` and `numpy`.

1.  **Clone the repository** (or download the script).
2.  **Install dependencies:**

    ```bash
    pip install opencv-python numpy
    ```

3.  **Run the script:**

    ```bash
    python main.py
    ```


## 🎮 Usage Guide

When the program starts, two windows will appear. Make sure to click on the **"Camera with Effects"** window to ensure keyboard commands are registered.

### Keyboard Shortcuts

| Key | Effect / Function | Description |
| :---: | :--- | :--- |
| **A** | **Gaussian Blur** | Smooths the image (Use trackbar to change kernel size). |
| **B** | **Canny Edge** | Detects edges using the Canny algorithm (Trackbar adjusts threshold). |
| **C** | **Sobel Edge** | Highlights edge gradients (Trackbar adjusts kernel size). |
| **D** | **Brightness** | Adjusts image luminosity. |
| **E** | **Contrast** | Adjusts the difference between light and dark areas. |
| **F** | **Negative** | Inverts the image colors. |
| **G** | **Grayscale** | Converts to B&W (Trackbar blends between Color and Gray). |
| **H** | **Resize** | Downscales the video to 50% size. |
| **I** | **Rotate** | Rotates the video 90° clockwise. |
| **J** | **Full Mirror** | Flips the image vertically and horizontally. |
| **K** | **Horizontal Mirror** | Flips the image horizontally (Selfie mode). |
| **L** | **Record (REC)** | Starts/Stops recording to `video.avi`. |
| **ESC** | **Exit** | Closes the application. |

### The Adjustment Bar (Trackbar)

The slider located at the top of the "Camera with Effects" window changes the parameters of the active filter:
* **For Blur/Sobel:** Changes the kernel size (sharpness/smoothness).
* **For Canny:** Changes the hysteresis threshold (sensitivity to lines).
* **For Brightness/Contrast:** Changes the Alpha/Beta values.
* **For Grayscale:** Changes the interpolation factor (0 = Color, Max = Gray).

Built with Python and OpenCV.