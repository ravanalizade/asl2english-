import os
import sys
import json
import cv2
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from preprocess import preprocess_crop
from net import CNN

DATASET = "dataset"
SPLIT = "training/split.json"
MODEL = "backend/model.pt"
CLASSES = "backend/classes.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

classes = json.load(open(CLASSES))
idx = {c: i for i, c in enumerate(classes)}
split = json.load(open(SPLIT))

model = CNN(len(classes)).to(DEVICE)
model.load_state_dict(torch.load(MODEL, map_location=DEVICE))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for c in classes:
        for f in split["test"][c]:
            img = cv2.imread(os.path.join(DATASET, c, f))
            x = preprocess_crop(img).astype("float32") / 255.0
            x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(DEVICE)
            y_true.append(idx[c])
            y_pred.append(int(model(x).argmax(1)))

print("top-1 accuracy:", round(accuracy_score(y_true, y_pred), 4))
print("macro-F1:", round(f1_score(y_true, y_pred, average="macro"), 4))
print()
print(classification_report(y_true, y_pred, target_names=classes, digits=3))

cm = confusion_matrix(y_true, y_pred)
np.savetxt("training/confusion_matrix.csv", cm, fmt="%d", delimiter=",")
print("confusion matrix saved to training/confusion_matrix.csv")
