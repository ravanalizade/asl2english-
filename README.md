# ASL Fingerspelling — Web App

Real-time ASL fingerspelling in the browser.
Pipeline: webcam -> MediaPipe finds & crops the hand -> backend processes the
crop (grayscale -> blur -> threshold -> resize) -> CNN predicts the letter ->
letters are assembled and sent to Gemini for a corrected English sentence.

Runs now with a placeholder model (random letters). After you train, the backend
auto-loads your model — no code to fill in.

## Structure

```
asl-project/
├─ collect_dataset.py     capture raw hand crops into dataset/<letter>/
├─ dataset/               your gathered images (you make this)
├─ backend/
│  ├─ main.py             FastAPI: /letter and /assemble
│  ├─ model.py            auto-loads model.pt + classes.json (placeholder if none)
│  ├─ preprocess.py       grayscale -> blur -> threshold -> resize (single source of truth)
│  ├─ requirements.txt
│  └─ .env.example        copy to .env, add Gemini key
├─ training/
│  ├─ split_data.py       split dataset 240/30/30 per letter (train/val/test)
│  ├─ train.py            train CNN, save backend/model.pt, save preprocess samples
│  ├─ evaluate.py         top-1, macro-F1, per-class recall, confusion matrix
│  └─ requirements.txt
└─ frontend/
   ├─ index.html, package.json, vite.config.js
   └─ src/  (App.jsx, App.css, api.js, useHandTracking.js, main.jsx)
```

## Order of work

1. Gather data:        `python collect_dataset.py`  (one run per letter)
2. Split:              `python training/split_data.py`
3. Train:              `python training/train.py`   (saves backend/model.pt)
4. Evaluate:           `python training/evaluate.py`
5. Run the app (backend + frontend) — it auto-loads your model.

Run the training commands from the project root (so `dataset/`, `backend/`,
`training/` are all visible).

## Run the app

Backend:
```
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Frontend:
```
cd frontend
npm install
npm run dev
```
Open the printed link (usually http://localhost:5173). Click Open camera, hold a
sign ~1s to capture a letter, use Delete / Delete all, then Send.

## Matching rules (keep accuracy honest)

- Crop in `useHandTracking.js` (CROP_SIZE 224, MARGIN 0.25, MIRROR on) must match
  `collect_dataset.py`.
- `preprocess.py` is the only copy of pixel processing; training and backend both
  use it, so they cannot drift.
- `CONFIDENCE_THRESHOLD` in `main.py` controls how sure the model must be before a
  letter is shown.

## Notes

- Single-signer, right-hand demo. J and Z use static stand-in shapes.
- Delete / Delete all / Send are buttons.
- Training: ~26 classes; runs on CPU slowly, faster on Colab GPU. Class order is
  taken from the folder names and saved to backend/classes.json automatically.
- Gemini: without a key the app still runs and returns the letters as text, you want to use gemini, add your api key to .env file
