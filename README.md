# Low-Light-Image-Enhancement

# Image Enhancement System using OpenCV & Scikit-Image  
**Low-Light Image Enhancement with Quality Evaluation (SSIM & PSNR)**

This project improves low-light images by applying image enhancement techniques using **OpenCV**, and evaluates output quality with **SSIM (Structural Similarity)** and **PSNR (Peak Signal-to-Noise Ratio)** from **scikit-image**.

The system is optimized to handle large images and real-time frames without freezing by using a custom fast SSIM wrapper.

- **YUV Color Space Conversion** – Converts the image from BGR → YUV to enhance only the luminance (Y channel) while preserving natural colors.

- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** – Enhances contrast in dark regions without over-amplifying noise. Uses clipLimit=3.0 and tileGridSize=(8,8).

- **Gamma Correction (γ = 1.5)** – Brightens mid-tones using a lookup table for smoother and natural brightness enhancement.

### Core Enhancement Code
```python
# Convert to YUV and apply CLAHE
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
yuv[:, :, 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(yuv[:, :, 0])
img_clahe = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

# Gamma correction
gamma = 1.5
look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                          for i in range(256)]).astype("uint8")
img_gamma = cv2.LUT(img_clahe, look_up_table)


---

## Features

### ✔ Low-Light Image Enhancement  
- Brightness & contrast adjustment  

### ✔ Image Quality Metrics  
- **SSIM**: Structural Similarity Index  
- **PSNR**: Peak Signal-to-Noise Ratio  
- Safe fallback to PSNR if SSIM is slow  

### ✔ Real-Time Capability  
- Works with single images  
- Can be extended to webcam or video feed  
- Smooth performance with optimized SSIM  

---

## Technologies Used


| Library | Purpose |
|--------|---------|
| **Python 3.12** | Core programming language |
| **OpenCV** (`opencv-python`) | Image processing, reading, enhancement |
| **scikit-image** (`skimage`) | SSIM, PSNR calculation |
| **NumPy** | Numerical operations |
| **Matplotlib** (optional) | Visual comparison |


---

## Installation

### Install dependencies

```bash
pip install opencv-python
pip install scikit-image
pip install numpy
pip install matplotlib



