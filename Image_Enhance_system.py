import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math

def classical_enhancement(img):
    # Convert to YUV and apply CLAHE
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(yuv[:, :, 0])
    img_clahe = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # Gamma correction
    gamma = 1.5
    look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                              for i in range(256)]).astype("uint8")
    img_gamma = cv2.LUT(img_clahe, look_up_table)
    return img_gamma

def psnr(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received, exiting...")
        break

    frame = cv2.resize(frame, (320, 240))
    enhanced = classical_enhancement(frame)

    # Calculate metrics (optional, comment out if you want faster performance)
    psnr_value = psnr(frame, enhanced)
    ssim_value = ssim(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                      cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))

    # Combine both frames for display
    combined = np.hstack((frame, enhanced))
    cv2.putText(combined, f"PSNR: {psnr_value:.2f} dB | SSIM: {ssim_value:.3f}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow('Original (Left) vs Enhanced (Right)', combined)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()