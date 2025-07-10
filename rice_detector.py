
import cv2
import numpy as np
import os

def detect_rice(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological cleaning
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    broken = 0
    whole = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 50:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h != 0 else 0

        # Improved broken detection condition
        is_broken = (
            area < 200 or 
            (aspect_ratio < 0.25 or aspect_ratio > 4.5) or
            (circularity < 0.35 and area < 400) or
            solidity < 0.80
        )

        color = (0, 0, 255) if is_broken else (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        if is_broken:
            broken += 1
        else:
            whole += 1

    output_path = os.path.join('static/uploads', 'processed_' + os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    return output_path, broken, whole
