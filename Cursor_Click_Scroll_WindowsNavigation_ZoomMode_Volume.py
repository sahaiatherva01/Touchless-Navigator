import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np

# --- Configuration and Setup ---
DEBUG_MODE = True
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
cap.set(3, 640)
cap.set(4, 480)

# PyAutoGUI Configuration
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# MediaPipe Hands Initialization
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils

# --- Gesture Control Parameters ---
SMOOTHING = 0.2
FRAME_REDUCTION = 100
current_mode = "NONE"
last_mode_change_time = 0
prev_x, prev_y = 0, 0

# Mode-specific state
alt_tab_active = False
click_performed = False
is_dragging = False
zoom_active = False
last_scroll_time = 0
last_volume_time = 0

# Gesture Sensitivities
PINCH_THRESHOLD = 30
SCROLL_AMOUNT = 300
SCROLL_COOLDOWN = 0.5
VOLUME_COOLDOWN = 0.1
ALT_TAB_SENSITIVITY = 70
ZOOM_SENSITIVITY_RATIO = 0.07
FINGER_RAISED_MARGIN = 10

# --- Helper Function ---
def get_finger_status(lmList, handedness):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]
    # Thumb
    if handedness == "Right":
        if lmList[tip_ids[0]][1] < lmList[tip_ids[0] - 2][1]: fingers.append(1)
        else: fingers.append(0)
    else:  # Left Hand
        if lmList[tip_ids[0]][1] > lmList[tip_ids[0] - 2][1]: fingers.append(1)
        else: fingers.append(0)
    # Other four fingers
    for id in range(1, 5):
        if lmList[tip_ids[id]][2] < lmList[tip_ids[id] - 2][2] - FINGER_RAISED_MARGIN:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# --- Main Loop ---
try:
    while True:
        success, img = cap.read()
        if not success: continue

        img = cv2.flip(img, 1)
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        new_mode = "NONE"
        current_time = time.time()
        is_alt_tab_gesture_this_frame = False

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            hand_landmarks = results.multi_hand_landmarks[0]
            lmList = [[id, int(lm.x * w), int(lm.y * h)] for id, lm in enumerate(hand_landmarks.landmark)]

            if num_hands == 1:
                handedness = results.multi_handedness[0].classification[0].label
                
                # 1. Pinch Gesture (Highest Priority)
                thumb_tip, index_tip = lmList[4], lmList[8]
                pinch_dist = math.hypot(index_tip[1] - thumb_tip[1], index_tip[2] - thumb_tip[2])

                if pinch_dist < PINCH_THRESHOLD:
                    new_mode = "SELECTING"
                    if not is_dragging:
                        pyautogui.mouseDown()
                        is_dragging = True
                    
                    ix, iy = index_tip[1], index_tip[2]
                    screen_x = np.interp(ix, (FRAME_REDUCTION, w - FRAME_REDUCTION), (0, screen_width))
                    screen_y = np.interp(iy, (FRAME_REDUCTION, h - FRAME_REDUCTION), (0, screen_height))
                    smooth_x = prev_x + (screen_x - prev_x) * SMOOTHING
                    smooth_y = prev_y + (screen_y - prev_y) * SMOOTHING
                    pyautogui.moveTo(smooth_x, smooth_y)
                    prev_x, prev_y = smooth_x, smooth_y
                    if DEBUG_MODE: cv2.circle(img, (ix, iy), 15, (0, 0, 255), cv2.FILLED)
                
                else:
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False

                    # 2. Other Finger-Counting Gestures
                    fingers = get_finger_status(lmList, handedness)
                    
                    if fingers == [0, 1, 1, 1, 0]: # SCROLL DOWN (Index, Middle, Ring)
                        new_mode = "SCROLL DOWN"
                        if current_time - last_scroll_time > SCROLL_COOLDOWN:
                            pyautogui.scroll(-SCROLL_AMOUNT)
                            last_scroll_time = current_time
                        if DEBUG_MODE:
                            cv2.circle(img, (lmList[8][1], lmList[8][2]), 10, (0, 255, 255), cv2.FILLED)
                            cv2.circle(img, (lmList[12][1], lmList[12][2]), 10, (0, 255, 255), cv2.FILLED)
                            cv2.circle(img, (lmList[16][1], lmList[16][2]), 10, (0, 255, 255), cv2.FILLED)

                    # *** GESTURE CHANGED to Ring + Pinky ***
                    elif fingers == [0, 0, 0, 1, 1]: # SCROLL UP (Ring, Pinky)
                        new_mode = "SCROLL UP"
                        if current_time - last_scroll_time > SCROLL_COOLDOWN:
                            pyautogui.scroll(SCROLL_AMOUNT)
                            last_scroll_time = current_time
                        if DEBUG_MODE:
                            cv2.circle(img, (lmList[16][1], lmList[16][2]), 10, (255, 0, 255), cv2.FILLED)
                            cv2.circle(img, (lmList[20][1], lmList[20][2]), 10, (255, 0, 255), cv2.FILLED)

                    elif fingers == [1, 1, 1, 0, 0]: # VOLUME UP
                        new_mode = "VOLUME UP"
                        if current_time - last_volume_time > VOLUME_COOLDOWN:
                            pyautogui.press('volumeup')
                            last_volume_time = current_time
                    
                    elif fingers == [1, 1, 0, 0, 1]: # VOLUME DOWN
                        new_mode = "VOLUME DOWN"
                        if current_time - last_volume_time > VOLUME_COOLDOWN:
                            pyautogui.press('volumedown')
                            last_volume_time = current_time
                    
                    elif fingers == [0, 1, 1, 0, 0]: # CLICK
                        new_mode = "CLICK"
                        if not click_performed and time.time() - last_mode_change_time > 0.5:
                            pyautogui.click()
                            click_performed = True
                        if DEBUG_MODE:
                            cv2.circle(img, (lmList[8][1], lmList[8][2]), 10, (0, 255, 0), cv2.FILLED)
                            cv2.circle(img, (lmList[12][1], lmList[12][2]), 10, (0, 255, 0), cv2.FILLED)
                    
                    elif fingers == [0, 1, 0, 0, 0]: # CURSOR
                        new_mode = "CURSOR"
                        ix, iy = lmList[8][1], lmList[8][2]
                        screen_x = np.interp(ix, (FRAME_REDUCTION, w - FRAME_REDUCTION), (0, screen_width))
                        screen_y = np.interp(iy, (FRAME_REDUCTION, h - FRAME_REDUCTION), (0, screen_height))
                        smooth_x = prev_x + (screen_x - prev_x) * SMOOTHING
                        smooth_y = prev_y + (screen_y - prev_y) * SMOOTHING
                        pyautogui.moveTo(smooth_x, smooth_y)
                        prev_x, prev_y = smooth_x, smooth_y
                        if DEBUG_MODE: cv2.circle(img, (ix, iy), 10, (255, 0, 255), cv2.FILLED)
                    
                    elif fingers == [0, 0, 0, 0, 0]: # ALT-TAB
                        new_mode = "ALT-TAB"
                        is_alt_tab_gesture_this_frame = True
                        if not alt_tab_active:
                            alt_tab_active = True
                            pyautogui.keyDown('alt')
                            pyautogui.press('tab')
                            alt_tab_start_x = lmList[9][1]
                        else:
                            current_x = lmList[9][1]
                            if abs(current_x - alt_tab_start_x) > ALT_TAB_SENSITIVITY:
                                pyautogui.press('right' if current_x > alt_tab_start_x else 'left')
                                alt_tab_start_x = current_x

            elif num_hands == 2:
                # Zoom logic...
                pass

        # --- KEYBOARD SAFETY AND STATE MANAGEMENT ---
        if not is_alt_tab_gesture_this_frame and alt_tab_active:
            pyautogui.keyUp('alt')
            alt_tab_active = False

        if new_mode != current_mode:
            if current_mode == "CLICK":
                click_performed = False
            current_mode = new_mode
            last_mode_change_time = current_time
        
        if not results.multi_hand_landmarks:
            if alt_tab_active: pyautogui.keyUp('alt')
            if is_dragging: pyautogui.mouseUp()
            alt_tab_active, is_dragging, current_mode = False, False, "NONE"

        # --- Display Information & Drawing ---
        if DEBUG_MODE and results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, hand_lms, mpHands.HAND_CONNECTIONS)
        
        cv2.rectangle(img, (10, 10), (350, 60), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f"MODE: {current_mode}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Gesture Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

except KeyboardInterrupt:
    print("\nProgram terminated by user (Ctrl+C).")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # Final cleanup
    pyautogui.keyUp('alt')
    pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
