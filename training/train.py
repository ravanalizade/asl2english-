import os
import sys
import json
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from preprocess import preprocess_crop, preprocess_steps
from net import CNN

DATASET = "dataset"
SPLIT = "training/split.json"
SAMPLES_DIR = "training/preprocess_samples"
MODEL_OUT = "backend/model.pt"

CLASSES_OUT = "backend/classes.json"
EPOCHS = 15
BATCH = 32
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

split = json.load(open(SPLIT))
classes = sorted(split["train"].keys())
idx = {c: i for i, c in enumerate(classes)}


def items(part):
    out = []
    for c in classes:
        for f in split[part][c]:
            out.append((os.path.join(DATASET, c, f), idx[c]))
    return out


class DS(Dataset):
    def __init__(self, part):
        self.data = items(part)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, y = self.data[i]
        x = preprocess_crop(cv2.imread(path)).astype("float32") / 255.0
        return torch.from_numpy(x).unsqueeze(0), y


def save_samples():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    path = items("train")[0][0]
    img = cv2.imread(path)
    s = preprocess_steps(img)
    cv2.imwrite(f"{SAMPLES_DIR}/0_original.png", img)
    cv2.imwrite(f"{SAMPLES_DIR}/1_gray.png", s["gray"])
    cv2.imwrite(f"{SAMPLES_DIR}/2_blurred.png", s["blurred"])
    cv2.imwrite(f"{SAMPLES_DIR}/3_thresholded.png", s["thresholded"])
    cv2.imwrite(f"{SAMPLES_DIR}/4_resized.png", s["resized"])
    print("saved preprocessing samples to", SAMPLES_DIR)


def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            p = model(x).argmax(1)
            correct += (p == y).sum().item()
            total += y.size(0)
    return correct / total


save_samples()
train_loader = DataLoader(DS("train"), batch_size=BATCH, shuffle=True)
val_loader = DataLoader(DS("val"), batch_size=BATCH)

model = CNN(len(classes)).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

best = 0.0
for e in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    acc = accuracy(model, val_loader)
    print(f"epoch {e + 1}/{EPOCHS} val_acc {acc:.3f}")
    if acc > best:
        best = acc
        torch.save(model.state_dict(), MODEL_OUT)
        json.dump(classes, open(CLASSES_OUT, "w"))

print("best val_acc", round(best, 3), "saved", MODEL_OUT)
