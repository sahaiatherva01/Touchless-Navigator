import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
cap.set(3, 640) # Set width
cap.set(4, 480) # Set height

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Screen size for mouse control
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

# --- Gesture Parameters ---
smooth_factor = 0.2
prev_x, prev_y = 0, 0
frameR = 100

# Click Parameters
start_hold_time = None
click_flag = False
REQUIRED_HOLD_DURATION = 1.0

# Scrolling Parameters
scroll_start_y = None
SCROLL_THRESHOLD = 25
SCROLL_AMOUNT = 100
is_scrolling = False # State for continuous scrolling

# Alt-Tab Gesture Parameters
alt_tab_mode = False
alt_tab_start_x = 0
ALT_TAB_SWIPE_THRESHOLD = 60
was_in_cursor_mode = False

# Refined Zoom Gesture Parameters
prev_zoom_dist, smoothed_zoom_dist = 0, 0
ZOOM_SENSITIVITY = 5.0
ZOOM_SMOOTH_FACTOR = 0.3
ZOOM_COOLDOWN = 0.4
last_zoom_time = 0

# Parameters for Toggling Zoom Mode
zoom_mode_active = False
ZOOM_TOGGLE_COOLDOWN = 1.5
last_zoom_toggle_time = 0

def fingers_up(lmList, handedness):
    """
    Determines which fingers are extended (up) using handedness for thumb accuracy.
    """
    fingers = []
    tips_ids = [4, 8, 12, 16, 20]
    # Thumb Check
    if handedness == "Right":
        if lmList[tips_ids[0]][1] < lmList[tips_ids[0] - 1][1]: fingers.append(1)
        else: fingers.append(0)
    else: # Left Hand
        if lmList[tips_ids[0]][1] > lmList[tips_ids[0] - 1][1]: fingers.append(1)
        else: fingers.append(0)
    # Other 4 fingers
    for id in range(1, 5):
        if lmList[tips_ids[id]][2] < lmList[tips_ids[id] - 2][2]: fingers.append(1)
        else: fingers.append(0)
    return fingers

def is_fist_detected(fingers):
    """
    Checks for a fist by confirming the four main fingers are down.
    """
    return fingers[1:] == [0, 0, 0, 0]

try:
    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        current_time = time.time()

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)

            # TWO-HAND LOGIC
            if num_hands == 2:
                if alt_tab_mode: pyautogui.keyUp('alt'); alt_tab_mode = False
                was_in_cursor_mode = False
                hand1_lms, hand2_lms = results.multi_hand_landmarks[0], results.multi_hand_landmarks[1]
                handedness1, handedness2 = results.multi_handedness[0].classification[0].label, results.multi_handedness[1].classification[0].label
                lmList1, lmList2 = [], []
                for id, lm in enumerate(hand1_lms.landmark): lmList1.append([id, int(lm.x * w), int(lm.y * h)])
                for id, lm in enumerate(hand2_lms.landmark): lmList2.append([id, int(lm.x * w), int(lm.y * h)])
                mpDraw.draw_landmarks(img, hand1_lms, mpHands.HAND_CONNECTIONS)
                mpDraw.draw_landmarks(img, hand2_lms, mpHands.HAND_CONNECTIONS)
                fingers1, fingers2 = fingers_up(lmList1, handedness1), fingers_up(lmList2, handedness2)
                if fingers1 == [1, 1, 1, 1, 1] and fingers2 == [1, 1, 1, 1, 1]:
                    if current_time - last_zoom_toggle_time > ZOOM_TOGGLE_COOLDOWN:
                        zoom_mode_active = not zoom_mode_active
                        last_zoom_toggle_time = current_time
                        print(f"ZOOM MODE {'ACTIVATED' if zoom_mode_active else 'DEACTIVATED'}")
                        prev_zoom_dist, smoothed_zoom_dist = 0, 0
                if zoom_mode_active:
                    x1, y1, x2, y2 = lmList1[8][1], lmList1[8][2], lmList2[8][1], lmList2[8][2]
                    current_dist = math.hypot(x2 - x1, y2 - y1)
                    if prev_zoom_dist == 0: prev_zoom_dist, smoothed_zoom_dist = current_dist, current_dist
                    else: smoothed_zoom_dist += (current_dist - smoothed_zoom_dist) * ZOOM_SMOOTH_FACTOR
                    if current_time - last_zoom_time > ZOOM_COOLDOWN:
                        delta_dist = smoothed_zoom_dist - prev_zoom_dist
                        if delta_dist > ZOOM_SENSITIVITY: pyautogui.hotkey('ctrl', '+'); print("Zoom In"); prev_zoom_dist, last_zoom_time = smoothed_zoom_dist, current_time
                        elif delta_dist < -ZOOM_SENSITIVITY: pyautogui.hotkey('ctrl', '-'); print("Zoom Out"); prev_zoom_dist, last_zoom_time = smoothed_zoom_dist, current_time
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3); cv2.putText(img, "ZOOM ACTIVE", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                else: cv2.putText(img, "Show open hands to toggle zoom", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            
            # ONE-HAND LOGIC
            elif num_hands == 1:
                zoom_mode_active = False
                handLms = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0].classification[0].label
                lmList = []
                for id, lm in enumerate(handLms.landmark): lmList.append([id, int(lm.x * w), int(lm.y * h)])
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                if lmList:
                    ix, iy = lmList[8][1], lmList[8][2]
                    fingers = fingers_up(lmList, handedness)
                    is_fist = is_fist_detected(fingers)
                    if is_fist and not alt_tab_mode and not was_in_cursor_mode:
                        pyautogui.keyDown('alt'); pyautogui.press('tab'); alt_tab_mode, alt_tab_start_x = True, lmList[0][1]
                    elif alt_tab_mode:
                        if is_fist:
                            delta_x = lmList[0][1] - alt_tab_start_x
                            if delta_x > ALT_TAB_SWIPE_THRESHOLD: pyautogui.press('tab'); alt_tab_start_x = lmList[0][1]
                            elif delta_x < -ALT_TAB_SWIPE_THRESHOLD: pyautogui.hotkey('shift', 'tab'); alt_tab_start_x = lmList[0][1]
                            cv2.putText(img, "ALT-TAB MODE", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
                        else: pyautogui.keyUp('alt'); alt_tab_mode = False
                    else:
                        if fingers[1] == 1 and fingers[2] == 0: # Cursor Mode
                            is_scrolling = False; was_in_cursor_mode = True
                            screen_x = np.interp(ix, (frameR, w - frameR), (0, screen_width)); screen_y = np.interp(iy, (frameR, h - frameR), (0, screen_height))
                            smooth_x = prev_x + (screen_x - prev_x) * smooth_factor; smooth_y = prev_y + (screen_y - prev_y) * smooth_factor
                            pyautogui.moveTo(smooth_x, smooth_y); prev_x, prev_y = smooth_x, smooth_y
                            cv2.circle(img, (ix, iy), 10, (255, 0, 255), cv2.FILLED)
                        
                        elif fingers[1] == 1 and fingers[2] == 1: # Scroll/Click Mode
                            was_in_cursor_mode = False
                            if start_hold_time is None: start_hold_time, scroll_start_y, is_scrolling, click_flag = current_time, iy, False, False
                            
                            delta_y = iy - scroll_start_y
                            if is_scrolling:
                                if abs(delta_y) > SCROLL_THRESHOLD:
                                    pyautogui.scroll(SCROLL_AMOUNT if delta_y < 0 else -SCROLL_AMOUNT)
                                    print(f"Scrolling {'Up' if delta_y < 0 else 'Down'}")
                                    scroll_start_y = iy # Update start_y for continuous motion
                            else:
                                if abs(delta_y) > SCROLL_THRESHOLD:
                                    is_scrolling = True; print("Scrolling Mode Activated")
                                elif current_time - start_hold_time > REQUIRED_HOLD_DURATION and not click_flag:
                                    pyautogui.click(); print("Click!")
                                    click_flag = True; start_hold_time = None
                            
                            if not is_scrolling and not click_flag and start_hold_time:
                                cv2.circle(img, (ix, iy), 15, (0, 165, 255), 2)
                        
                        else:
                            was_in_cursor_mode, click_flag, start_hold_time, is_scrolling = False, False, None, False
        else:
            if alt_tab_mode: pyautogui.keyUp('alt'); alt_tab_mode = False
            zoom_mode_active = False; is_scrolling = False
        
        cv2.imshow("NEXUS Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

except KeyboardInterrupt: print("Program interrupted by user.")
except pyautogui.FailSafeException: print("Fail-safe triggered by user. Exiting.")
except Exception as e: print(f"An error occurred: {e}")
finally:
    if alt_tab_mode: pyautogui.keyUp('alt')
    cap.release()
    cv2.destroyAllWindows()
