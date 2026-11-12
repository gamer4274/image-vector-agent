import numpy as np
import cv2

img = np.full((900, 1200, 3), 245, dtype=np.uint8)
black = (20, 20, 20)

cv2.rectangle(img, (60, 60), (1140, 840), black, 3)
cv2.rectangle(img, (120, 120), (620, 500), black, 2)
cv2.circle(img, (950, 300), 150, black, 2)
cv2.circle(img, (950, 300), 60, black, 2)
cv2.line(img, (120, 120), (620, 500), black, 2)
cv2.line(img, (620, 120), (120, 500), black, 2)

pts = np.array([[100, 700], [250, 600], [400, 750], [550, 620], [700, 760], [850, 640], [1100, 720]], np.int32)
cv2.polylines(img, [pts], False, black, 2)

cv2.ellipse(img, (300, 780), (80, 40), 25, 0, 360, black, 2)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "PIECE A-104", (380, 560), font, 1.4, black, 3, cv2.LINE_AA)
cv2.putText(img, "echelle 1:2", (380, 610), font, 0.7, black, 2, cv2.LINE_AA)

noise = np.random.default_rng(7).normal(0, 6, img.shape).astype(np.int16)
img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

cv2.imwrite("plan_test.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
print("OK")
