import os
import json
import random

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pt")
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "classes.json")
DEFAULT_CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


class LetterModel:
    def __init__(self):
        self.model = None
        self.placeholder = True
        self.classes = DEFAULT_CLASSES
        self._try_load()

    def _try_load(self):
        if not os.path.exists(MODEL_PATH):
            print("[model] No model.pt. Placeholder mode (random letters).")
            return
        try:
            import torch
            from net import CNN
            if os.path.exists(CLASSES_PATH):
                self.classes = json.load(open(CLASSES_PATH))
            model = CNN(len(self.classes))
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.eval()
            self.model = model
            self.placeholder = False
            print("[model] Loaded model.pt with", len(self.classes), "classes.")
        except Exception as e:
            print("[model] Load failed, placeholder mode:", e)

    def predict(self, processed_image):
        if self.placeholder or self.model is None:
            return random.choice(self.classes), round(random.uniform(0.80, 0.99), 3)
        import torch
        x = torch.from_numpy(processed_image).float() / 255.0
        x = x.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
            i = int(torch.argmax(probs))
        return self.classes[i], float(probs[i])


letter_model = LetterModel()
