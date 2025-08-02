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
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Screen size for mouse control
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False # Disable failsafe for edge movement
pyautogui.PAUSE = 0 # No pause between PyAutoGUI calls

# --- Gesture Parameters ---
smooth_factor = 0.2
prev_x, prev_y = 0, 0
frameR = 100

# NEW: Pinch-to-Drag/Select Parameters
PINCH_THRESHOLD = 30 # Pixel distance between thumb and index to trigger pinch
is_dragging = False  # State for pinch-drag

# Click Parameters
start_hold_time = None
click_flag = False
REQUIRED_HOLD_DURATION = 1.0

# Scrolling Parameters
scroll_start_y = None
SCROLL_THRESHOLD = 25
SCROLL_AMOUNT = 100
is_scrolling = False

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
zoom_mode_active = False
ZOOM_TOGGLE_COOLDOWN = 1.5
last_zoom_toggle_time = 0

# Volume Control Parameters
VOLUME_COOLDOWN = 0.3
last_volume_change_time = 0

def fingers_up(lmList, handedness):
    """Determines which fingers are extended (up)."""
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
    """Checks for a fist."""
    return fingers[1:] == [0, 0, 0, 0]

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        current_time = time.time()

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)

            # TWO-HAND LOGIC
            if num_hands == 2:
                # Reset one-hand modes if two hands are detected
                if alt_tab_mode:
                    pyautogui.keyUp('alt')
                    alt_tab_mode = False
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                was_in_cursor_mode = False

                # Get landmarks and handedness for both hands
                hand1_lms, hand2_lms = results.multi_hand_landmarks[0], results.multi_hand_landmarks[1]
                handedness1 = results.multi_handedness[0].classification[0].label
                handedness2 = results.multi_handedness[1].classification[0].label
                
                lmList1, lmList2 = [], []
                for id, lm in enumerate(hand1_lms.landmark):
                    lmList1.append([id, int(lm.x * w), int(lm.y * h)])
                for id, lm in enumerate(hand2_lms.landmark):
                    lmList2.append([id, int(lm.x * w), int(lm.y * h)])
                
                mpDraw.draw_landmarks(img, hand1_lms, mpHands.HAND_CONNECTIONS)
                mpDraw.draw_landmarks(img, hand2_lms, mpHands.HAND_CONNECTIONS)
                
                fingers1, fingers2 = fingers_up(lmList1, handedness1), fingers_up(lmList2, handedness2)
                
                # Toggle Zoom Mode (both hands open)
                if fingers1 == [1, 1, 1, 1, 1] and fingers2 == [1, 1, 1, 1, 1]:
                    if current_time - last_zoom_toggle_time > ZOOM_TOGGLE_COOLDOWN:
                        zoom_mode_active = not zoom_mode_active
                        last_zoom_toggle_time = current_time
                        prev_zoom_dist, smoothed_zoom_dist = 0, 0 # Reset on toggle
                
                if zoom_mode_active:
                    x1, y1 = lmList1[8][1], lmList1[8][2]
                    x2, y2 = lmList2[8][1], lmList2[8][2]
                    current_dist = math.hypot(x2 - x1, y2 - y1)
                    
                    if prev_zoom_dist == 0:
                        prev_zoom_dist, smoothed_zoom_dist = current_dist, current_dist
                    else:
                        smoothed_zoom_dist += (current_dist - smoothed_zoom_dist) * ZOOM_SMOOTH_FACTOR
                    
                    if current_time - last_zoom_time > ZOOM_COOLDOWN:
                        delta_dist = smoothed_zoom_dist - prev_zoom_dist
                        if delta_dist > ZOOM_SENSITIVITY:
                            pyautogui.hotkey('ctrl', '+')
                            prev_zoom_dist, last_zoom_time = smoothed_zoom_dist, current_time
                        elif delta_dist < -ZOOM_SENSITIVITY:
                            pyautogui.hotkey('ctrl', '-')
                            prev_zoom_dist, last_zoom_time = smoothed_zoom_dist, current_time
                    
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(img, "ZOOM ACTIVE", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                else:
                    cv2.putText(img, "Show open hands to toggle zoom", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            
            # ONE-HAND LOGIC
            elif num_hands == 1:
                zoom_mode_active = False # Deactivate zoom with one hand
                
                handLms = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0].classification[0].label
                
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    lmList.append([id, int(lm.x * w), int(lm.y * h)])
                
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                
                if lmList:
                    thumb_tip = lmList[4]
                    index_tip = lmList[8]
                    ix, iy = index_tip[1], index_tip[2]

                    # --- 1. NEW: Pinch Gesture for Dragging (Highest Priority) ---
                    distance = math.hypot(index_tip[1] - thumb_tip[1], index_tip[2] - thumb_tip[2])
                    
                    if distance < PINCH_THRESHOLD:
                        # Reset other modes to prevent conflicts
                        was_in_cursor_mode = False
                        is_scrolling = False
                        start_hold_time = None
                        
                        if not is_dragging:
                            is_dragging = True
                            pyautogui.mouseDown()
                            
                        # Move cursor while dragging
                        screen_x = np.interp(ix, (frameR, w - frameR), (0, screen_width))
                        screen_y = np.interp(iy, (frameR, h - frameR), (0, screen_height))
                        smooth_x = prev_x + (screen_x - prev_x) * smooth_factor
                        smooth_y = prev_y + (screen_y - prev_y) * smooth_factor
                        pyautogui.moveTo(smooth_x, smooth_y)
                        prev_x, prev_y = smooth_x, smooth_y

                        # Visual Feedback for dragging
                        cv2.circle(img, (ix, iy), 15, (0, 0, 255), cv2.FILLED)
                        cv2.putText(img, "DRAGGING / SELECTING", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

                    else:
                        if is_dragging:
                            pyautogui.mouseUp()
                            is_dragging = False

                        # --- 2. Other Gestures (if not dragging) ---
                        fingers = fingers_up(lmList, handedness)
                        is_fist = is_fist_detected(fingers)
                        
                        # Alt-Tab (Fist)
                        if is_fist and not alt_tab_mode and not was_in_cursor_mode:
                            pyautogui.keyDown('alt')
                            pyautogui.press('tab')
                            alt_tab_mode = True
                            alt_tab_start_x = lmList[0][1]
                        elif alt_tab_mode:
                            if is_fist:
                                delta_x = lmList[0][1] - alt_tab_start_x
                                if delta_x > ALT_TAB_SWIPE_THRESHOLD:
                                    pyautogui.press('tab')
                                    alt_tab_start_x = lmList[0][1]
                                elif delta_x < -ALT_TAB_SWIPE_THRESHOLD:
                                    pyautogui.hotkey('shift', 'tab')
                                    alt_tab_start_x = lmList[0][1]
                                cv2.putText(img, "ALT-TAB MODE", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
                            else:
                                pyautogui.keyUp('alt')
                                alt_tab_mode = False
                        else:
                            # Cursor Mode (Index up)
                            if fingers[1] == 1 and fingers[2:] == [0, 0, 0]:
                                is_scrolling = False
                                was_in_cursor_mode = True
                                screen_x = np.interp(ix, (frameR, w - frameR), (0, screen_width))
                                screen_y = np.interp(iy, (frameR, h - frameR), (0, screen_height))
                                smooth_x = prev_x + (screen_x - prev_x) * smooth_factor
                                smooth_y = prev_y + (screen_y - prev_y) * smooth_factor
                                pyautogui.moveTo(smooth_x, smooth_y)
                                prev_x, prev_y = smooth_x, smooth_y
                                cv2.circle(img, (ix, iy), 10, (255, 0, 255), cv2.FILLED)
                                cv2.putText(img, "CURSOR MODE", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

                            # Volume Up (Thumb, Index, Middle up)
                            elif fingers[0:3] == [1, 1, 1] and fingers[3:] == [0, 0]:
                                if current_time - last_volume_change_time > VOLUME_COOLDOWN:
                                    pyautogui.press('volumeup')
                                    last_volume_change_time = current_time
                                cv2.putText(img, "VOLUME UP", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)

                            # Click/Scroll Mode (Index, Middle up)
                            elif fingers[1] == 1 and fingers[2] == 1 and fingers[3:] == [0, 0]:
                                was_in_cursor_mode = False
                                if start_hold_time is None:
                                    start_hold_time = current_time
                                    scroll_start_y = iy
                                    is_scrolling, click_flag = False, False
                                
                                delta_y = iy - scroll_start_y
                                if is_scrolling or abs(delta_y) > SCROLL_THRESHOLD:
                                    is_scrolling = True
                                    pyautogui.scroll(SCROLL_AMOUNT if delta_y < 0 else -SCROLL_AMOUNT)
                                    scroll_start_y = iy
                                    cv2.putText(img, "SCROLLING", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)
                                elif current_time - start_hold_time > REQUIRED_HOLD_DURATION and not click_flag:
                                    pyautogui.click()
                                    click_flag, start_hold_time = True, None
                                    cv2.putText(img, "CLICKED!", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                                else:
                                    cv2.putText(img, "HOLD FOR CLICK / MOVE FOR SCROLL", (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)


                            # Volume Down (Thumb, Index, Pinky up)
                            elif fingers == [1, 1, 0, 0, 1]:
                                if current_time - last_volume_change_time > VOLUME_COOLDOWN:
                                    pyautogui.press('volumedown')
                                    last_volume_change_time = current_time
                                cv2.putText(img, "VOLUME DOWN", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)

                            else:
                                was_in_cursor_mode, click_flag, start_hold_time, is_scrolling = False, False, None, False
        else:
            # Reset all states if no hands are detected
            if alt_tab_mode: pyautogui.keyUp('alt')
            if is_dragging: pyautogui.mouseUp()
            alt_tab_mode, zoom_mode_active, is_scrolling, was_in_cursor_mode, is_dragging = False, False, False, False, False

        cv2.imshow("Gesture Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
except KeyboardInterrupt:
    print("Program interrupted by user.")
except pyautogui.FailSafeException:
    print("Fail-safe triggered by user. Exiting.")
except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Ensure keys/mouse are released on exit
    if alt_tab_mode: pyautogui.keyUp('alt')
    if is_dragging: pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
