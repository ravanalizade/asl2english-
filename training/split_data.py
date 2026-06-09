import os
import json
import random

DATASET = "dataset"
OUT = "training/split.json"
TRAIN, VAL, TEST = 240, 30, 30
random.seed(42)

split = {"train": {}, "val": {}, "test": {}}
for cls in sorted(os.listdir(DATASET)):
    d = os.path.join(DATASET, cls)
    if not os.path.isdir(d):
        continue
    files = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
    random.shuffle(files)
    need = TRAIN + VAL + TEST
    if len(files) < need:
        print(f"{cls}: only {len(files)} images, need {need}")
    split["train"][cls] = files[:TRAIN]
    split["val"][cls] = files[TRAIN:TRAIN + VAL]
    split["test"][cls] = files[TRAIN + VAL:TRAIN + VAL + TEST]

os.makedirs("training", exist_ok=True)
json.dump(split, open(OUT, "w"))
print("wrote", OUT)
