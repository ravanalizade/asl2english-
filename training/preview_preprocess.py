import os
import sys
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from preprocess import preprocess_steps

DATASET = "dataset"
OUT = "preprocess_samples"

sample = None
for c in sorted(os.listdir(DATASET)):
    d = os.path.join(DATASET, c)
    if not os.path.isdir(d):
        continue
    files = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
    if files:
        sample = os.path.join(d, files[0])
        break

img = cv2.imread(sample)
s = preprocess_steps(img)
os.makedirs(OUT, exist_ok=True)
cv2.imwrite(f"{OUT}/0_original.png", img)
cv2.imwrite(f"{OUT}/1_gray.png", s["gray"])
cv2.imwrite(f"{OUT}/2_blurred.png", s["blurred"])
cv2.imwrite(f"{OUT}/3_thresholded.png", s["thresholded"])
cv2.imwrite(f"{OUT}/4_resized.png", s["resized"])
print("saved 5 step images to", OUT, "from", sample)
