import cv2
import time
import math
import numpy as np
from collections import deque, Counter
import mediapipe as mp

# Install: pip install opencv-python mediapipe numpy
# Run: python mudras.py

# ------------------ Camera ------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Error: Could not open camera.")
cap.set(3, 640)
cap.set(4, 480)

# ------------------ MediaPipe ------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
)

# ------------------ Helpers ------------------

def euclid(ax, ay, bx, by):
    return float(math.hypot(ax - bx, ay - by))


def fingers_up(lm, handedness_label: str):
    # Returns [thumb, index, middle, ring, pinky] (1 if up)
    up = []
    thumb_tip_x = lm[4][1]
    thumb_ip_x = lm[3][1]
    # Handedness-aware thumb extension
    if handedness_label == 'Right':
        up.append(1 if thumb_tip_x > thumb_ip_x + 10 else 0)
    else:
        up.append(1 if thumb_tip_x < thumb_ip_x - 10 else 0)
    # Other fingers: tip above PIP
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for t, p in zip(tips, pips):
        up.append(1 if lm[t][2] < lm[p][2] - 10 else 0)
    return up


def hand_span(lm):
    # Use wrist(0) to middle_tip(12)
    return euclid(lm[0][1], lm[0][2], lm[12][1], lm[12][2]) + 1e-6


def close(a_idx, b_idx, lm, thr_px):
    return euclid(lm[a_idx][1], lm[a_idx][2], lm[b_idx][1], lm[b_idx][2]) < thr_px


def roughly_aligned_y(indices, lm, tol):
    ys = [lm[i][2] for i in indices]
    return (max(ys) - min(ys)) < tol


def finger_gap_sum(indices, lm):
    # Sum of adj fingertip gaps as spread proxy
    total = 0.0
    for i in range(len(indices) - 1):
        a, b = indices[i], indices[i + 1]
        total += euclid(lm[a][1], lm[a][2], lm[b][1], lm[b][2])
    return total

# ------------------ Rule-based mudras (10) ------------------
# Names: Pataka, Tripataka, Ardhapataka, Mayura, Ardhachandra, Arala, Shukatunda,
# Mushthi, Shikhara, Suchi (10)


def classify_mudra(lm, handed):
    up = fingers_up(lm, handed)
    span = hand_span(lm)

    near = 0.20 * span
    near_small = 0.14 * span
    align_y_tol = 0.12 * span

    thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip = 4, 8, 12, 16, 20

    # Feature helpers
    index_pinky_gap = euclid(lm[index_tip][1], lm[index_tip][2], lm[pinky_tip][1], lm[pinky_tip][2])
    spread_total = finger_gap_sum([8, 12, 16, 20], lm)

    # 1) Mayura: thumb and ring touch, index+middle up
    if close(thumb_tip, ring_tip, lm, near) and up[1] == 1 and up[2] == 1:
        return "Mayura"

    # 2) Suchi: only index up
    if up == [0, 1, 0, 0, 0]:
        return "Suchi"

    # 3) Mushthi: all down or almost down
    if sum(up) <= 1:
        return "Mushthi"

    # 4) Shikhara: thumb up, others down
    if up[0] == 1 and sum(up[1:]) == 0:
        return "Shikhara"

    # 5) Shukatunda: thumb-index close, middle slightly up, ring/pinky down
    if close(thumb_tip, index_tip, lm, near_small) and up[1] == 1 and up[2] in (0, 1) and up[3] == 0 and up[4] == 0:
        return "Shukatunda"

    # 6) Arala: index slightly bent, others up; we approximate by index up & thumb up & small spread
    if up[0] == 1 and up[1] == 1 and up[2] == 1 and up[3] == 1 and up[4] in (0, 1):
        if spread_total < 1.3 * span and index_pinky_gap < 0.85 * span:
            return "Arala"

    # 7) Ardhachandra: all extended but thumb adducted near index, small spread
    if up[1] == 1 and up[2] == 1 and up[3] == 1 and up[4] == 1:
        if close(4, 8, lm, 0.28 * span) and roughly_aligned_y([8, 12, 16, 20], lm, align_y_tol) and spread_total < 1.5 * span:
            return "Ardhachandra"

    # 8) Ardhapataka: index+middle+ring up, pinky down (thumb free)
    if up[1] == 1 and up[2] == 1 and up[3] == 1 and up[4] == 0:
        return "Ardhapataka"

    # 9) Tripataka: index+middle+pinky up, ring down (thumb free)
    if up[1] == 1 and up[2] == 1 and up[3] == 0 and up[4] == 1:
        return "Tripataka"

    # 10) Pataka: all extended, aligned, not too spread
    if up == [1, 1, 1, 1, 1] or (up[1] and up[2] and up[3] and up[4]):
        if roughly_aligned_y([8, 12, 16, 20], lm, align_y_tol) and index_pinky_gap < 0.8 * span:
            return "Pataka"

    return "Unknown"

# ------------------ Temporal smoothing / hysteresis ------------------
SMOOTHING_WINDOW = 11
COOLDOWN_SEC = 0.35
CONFIRM_RATIO = 0.7  # proportion in window needed to confirm
history = deque(maxlen=SMOOTHING_WINDOW)
last_confirm_t = 0.0
last_label = "..."

# ------------------ Main loop ------------------
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        display = last_label
        hud = []

        if results.multi_hand_landmarks:
            hlm = results.multi_hand_landmarks[0]
            handed = results.multi_handedness[0].classification[0].label if results.multi_handedness else 'Right'

            lm = []
            for idx, p in enumerate(hlm.landmark):
                lm.append([idx, int(p.x * w), int(p.y * h)])

            mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

            pred = classify_mudra(lm, handed)
            history.append(pred)

            # Confirm only if class dominates the window
            if len(history) == history.maxlen:
                most, cnt = Counter(history).most_common(1)[0]
                now = time.time()
                if most != "Unknown" and cnt >= int(CONFIRM_RATIO * SMOOTHING_WINDOW) and (now - last_confirm_t) >= COOLDOWN_SEC:
                    display = most
                    last_label = display
                    last_confirm_t = now

            hud.append(f"Hand: {handed}")
        else:
            history.clear()

        cv2.putText(frame, f"Mudra: {display}", (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 180, 240), 2)
        y0 = 74
        for line in hud:
            cv2.putText(frame, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 220, 80), 2)
            y0 += 28

        cv2.imshow("Sattriya Mudra Recognition (Rule-based, 10 classes)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
