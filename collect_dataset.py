

import os
import time
import cv2
import mediapipe as mp

DATASET_DIR   = "dataset"  
TARGET_COUNT  = 600        
CROP_SIZE     = 224        
MARGIN        = 0.25       
CAPTURE_DELAY = 0.12       
COUNTDOWN     = 3          
CAMERA_INDEX  = 0          
MIRROR        = True       

mp_hands = mp.solutions.hands


def get_hand_crop(frame, landmarks):
    """
    Make a SQUARE crop around the hand.
      frame     : the webcam image (BGR)
      landmarks : MediaPipe hand landmarks (normalized 0..1)
    Returns a CROP_SIZE x CROP_SIZE image, or None if the crop is invalid.

    >>> REUSE THIS EXACT FUNCTION IN YOUR LIVE APP <<<
    """
    h, w = frame.shape[:2]

    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]

    x_min, x_max = min(xs) * w, max(xs) * w
    y_min, y_max = min(ys) * h, max(ys) * h

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    side = max(x_max - x_min, y_max - y_min) * (1 + MARGIN)
    half = side / 2

    left   = int(cx - half)
    right  = int(cx + half)
    top    = int(cy - half)
    bottom = int(cy + half)

    left   = max(0, left)
    top    = max(0, top)
    right  = min(w, right)
    bottom = min(h, bottom)

    if right - left < 10 or bottom - top < 10:
        return None

    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE))


def count_existing(folder):
    """How many jpg images are already in this folder (so we can continue)."""
    return len([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])


def main():
    sign = input("Sign / folder name (e.g. A, B, delete, send): ").strip()
    if not sign:
        print("No name given. Exiting.")
        return

    folder = os.path.join(DATASET_DIR, sign)
    os.makedirs(folder, exist_ok=True)   

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open the camera.")
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    saved = count_existing(folder)
    start_count = saved
    print(f"Folder '{folder}' already has {saved} images. Target is {TARGET_COUNT}.")
    print("Show the sign to the camera.  q = quit,  p = pause.")

    countdown_end = time.time() + COUNTDOWN
    paused = False
    last_save = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame failed.")
            break

        if MIRROR:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        view = frame.copy()
        hand_found = result.multi_hand_landmarks is not None
        crop = None

        if hand_found:
            landmarks = result.multi_hand_landmarks[0]
            crop = get_hand_crop(frame, landmarks)
            h, w = frame.shape[:2]
            xs = [lm.x for lm in landmarks.landmark]
            ys = [lm.y for lm in landmarks.landmark]
            cv2.rectangle(view,
                          (int(min(xs) * w), int(min(ys) * h)),
                          (int(max(xs) * w), int(max(ys) * h)),
                          (0, 255, 0), 2)

        now = time.time()
        in_countdown = now < countdown_end

        if in_countdown:
            status = f"Get ready... {int(countdown_end - now) + 1}"
        elif paused:
            status = "PAUSED (press p)"
        else:
            status = f"Saving: {saved}/{TARGET_COUNT}"
        cv2.putText(view, f"[{sign}] {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if not hand_found:
            cv2.putText(view, "No hand detected", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Collect dataset", view)

        if (not in_countdown) and (not paused) and (crop is not None):
            if now - last_save >= CAPTURE_DELAY:
                path = os.path.join(folder, f"{sign}_{saved:04d}.jpg")
                cv2.imwrite(path, crop)
                saved += 1
                last_save = now

        if saved >= TARGET_COUNT:
            print(f"Done. Saved {saved - start_count} new images in '{folder}'.")
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"Stopped early. {saved - start_count} new images saved.")
            break
        if key == ord('p'):
            paused = not paused

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
