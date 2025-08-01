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
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.7)
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
REQUIRED_HOLD_DURATION = 0.5

# Scrolling Parameters
scroll_start_y = None
SCROLL_THRESHOLD = 30
SCROLL_AMOUNT = 120
last_action_time = 0
SCROLL_COOLDOWN = 0.7
POST_ACTION_CURSOR_DELAY = 0.5

# Alt-Tab Gesture Parameters
alt_tab_mode = False
alt_tab_start_x = 0
alt_tab_start_y = 0
ALT_TAB_SWIPE_THRESHOLD = 60
fist_hold_start_time = None
REQUIRED_FIST_DURATION = 1.0
was_in_cursor_mode = False

def fingers_up(lmList):
    """Determines which fingers are extended (up)."""
    fingers = []
    tips_ids = [4, 8, 12, 16, 20]
    # Thumb Check
    if lmList[tips_ids[0]][1] < lmList[tips_ids[0] - 2][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other 4 fingers
    for id in range(1, 5):
        if lmList[tips_ids[id]][2] < lmList[tips_ids[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

## --- MODIFIED: ROBUST FIST DETECTION FUNCTION --- ##
def is_fist_detected(lmList):
    """
    Checks for a fist that is facing the camera, avoiding conflict with scrolling.
    """
    # --- CONFLICT CHECK ---
    # If the index or middle finger is extended (for scrolling/clicking), it cannot be a fist.
    if lmList[8][2] < lmList[6][2] or lmList[12][2] < lmList[10][2]:
        return False

    # --- FIST DETECTION LOGIC ---
    # Condition 1: Check if the remaining fingers are curled
    fingers_curled = (
        lmList[16][2] > lmList[14][2] and # Ring
        lmList[20][2] > lmList[18][2]     # Pinky
    )
    if not fingers_curled:
        return False

    # Condition 2: Check for "facing" orientation
    knuckle_width = abs(lmList[5][1] - lmList[17][1])
    hand_height = abs(lmList[0][2] - lmList[9][2])
    
    if hand_height < 1: return False # Avoid division by zero

    orientation_ratio = knuckle_width / hand_height
    is_facing_camera = orientation_ratio > 0.6

    return is_facing_camera

try:
    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                if lmList:
                    fingers = fingers_up(lmList)
                    current_time = time.time()
                    ix, iy = lmList[8][1], lmList[8][2]
                    
                    is_fist = is_fist_detected(lmList)

                    # --- GESTURE LOGIC ---
                    if alt_tab_mode:
                        if is_fist:
                            # Navigation logic
                            current_x, current_y = lmList[0][1], lmList[0][2]
                            delta_x, delta_y = current_x - alt_tab_start_x, current_y - alt_tab_start_y

                            if abs(delta_x) > abs(delta_y):
                                if delta_x > ALT_TAB_SWIPE_THRESHOLD:
                                    pyautogui.press('tab')
                                    alt_tab_start_x = current_x
                                elif delta_x < -ALT_TAB_SWIPE_THRESHOLD:
                                    pyautogui.hotkey('shift', 'tab')
                                    alt_tab_start_x = current_x
                            else:
                                if delta_y > ALT_TAB_SWIPE_THRESHOLD:
                                    pyautogui.press('down')
                                    alt_tab_start_y = current_y
                                elif delta_y < -ALT_TAB_SWIPE_THRESHOLD:
                                    pyautogui.press('up')
                                    alt_tab_start_y = current_y
                            cv2.putText(img, "ALT-TAB MODE", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
                        else:
                            print("Tab Deactivated")
                            pyautogui.keyUp('alt')
                            alt_tab_mode = False
                    
                    else:
                        # 1. CURSOR MOVEMENT
                        if fingers[1] == 1 and fingers[2] == 0:
                            was_in_cursor_mode = True
                            fist_hold_start_time = None
                            start_hold_time, click_flag, scroll_start_y = None, False, None
                            
                            if current_time - last_action_time > POST_ACTION_CURSOR_DELAY:
                                smooth_x = prev_x + (ix - prev_x) * smooth_factor
                                smooth_y = prev_y + (iy - prev_y) * smooth_factor
                                screen_x = np.interp(smooth_x, (frameR, w - frameR), (0, screen_width))
                                screen_y = np.interp(smooth_y, (frameR, h - frameR), (0, screen_height))
                                
                                final_x, final_y = np.clip(screen_x, 1, screen_width - 1), np.clip(screen_y, 1, screen_height - 1)
                                pyautogui.moveTo(final_x, final_y)
                                prev_x, prev_y = smooth_x, smooth_y
                            else:
                                prev_x, prev_y = ix, iy
                            cv2.circle(img, (ix, iy), 10, (255, 0, 255), cv2.FILLED)

                        # 2. CLICK & SCROLL
                        elif fingers[1] == 1 and fingers[2] == 1:
                            was_in_cursor_mode = False
                            fist_hold_start_time = None
                            if start_hold_time is None:
                                start_hold_time, scroll_start_y, click_flag = current_time, iy, False
                            delta_y = iy - scroll_start_y
                            if abs(delta_y) > SCROLL_THRESHOLD:
                                if delta_y < 0 and (current_time - last_action_time > SCROLL_COOLDOWN):
                                    pyautogui.scroll(-SCROLL_AMOUNT)
                                    print("Scrolled Down")
                                    last_action_time = current_time
                                start_hold_time = None 
                            elif start_hold_time and (current_time - start_hold_time > REQUIRED_HOLD_DURATION):
                                if not click_flag:
                                    pyautogui.click()
                                    print("Click!")
                                    click_flag, last_action_time = True, current_time
                                    cv2.circle(img, (ix, iy), 15, (0, 255, 0), cv2.FILLED)
                            if start_hold_time and not click_flag:
                                cv2.circle(img, (ix, iy), 15, (0, 165, 255), 2)
                        
                        # 3. ALT-TAB ACTIVATION (Hold Fist)
                        elif is_fist and not was_in_cursor_mode:
                            if fist_hold_start_time is None:
                                fist_hold_start_time = current_time
                            elif current_time - fist_hold_start_time > REQUIRED_FIST_DURATION:
                                print("Tab Activated")
                                pyautogui.keyDown('alt')
                                pyautogui.press('tab')
                                alt_tab_mode = True
                                alt_tab_start_x, alt_tab_start_y = lmList[0][1], lmList[0][2]
                                fist_hold_start_time = None
                        
                        # 4. RESET STATE
                        else:
                            was_in_cursor_mode = False
                            fist_hold_start_time = None
                            start_hold_time, click_flag, scroll_start_y = None, False, None

        cv2.imshow("NEXUS Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")
except pyautogui.FailSafeException:
    print("Fail-safe triggered by user. Exiting.")
except Exception as e:
    print(f"An error occurred: {e}")

finally:
    print("Closing... Releasing any pressed keys.")
    pyautogui.keyUp('alt')
    cap.release()
    cv2.destroyAllWindows()
