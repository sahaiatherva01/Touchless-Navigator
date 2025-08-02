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

# NEW: Volume Control Parameters
VOLUME_COOLDOWN = 0.3 # Cooldown to prevent rapid volume changes
last_volume_change_time = 0

def fingers_up(lmList, handedness):
    """
    Determines which fingers are extended (up) using handedness for thumb accuracy.
    """
    fingers = []
    tips_ids = [4, 8, 12, 16, 20]
    # Thumb Check
    if handedness == "Right":
        # For right hand, thumb tip (4) x-coordinate should be less than the joint below it (3)
        if lmList[tips_ids[0]][1] < lmList[tips_ids[0] - 1][1]: fingers.append(1)
        else: fingers.append(0)
    else: # Left Hand
        # For left hand, thumb tip (4) x-coordinate should be greater than the joint below it (3)
        if lmList[tips_ids[0]][1] > lmList[tips_ids[0] - 1][1]: fingers.append(1)
        else: fingers.append(0)
    # Other 4 fingers (Index, Middle, Ring, Pinky)
    # Check if the tip of the finger is above the joint below it (y-coordinate)
    for id in range(1, 5):
        if lmList[tips_ids[id]][2] < lmList[tips_ids[id] - 2][2]: fingers.append(1)
        else: fingers.append(0)
    return fingers

def is_fist_detected(fingers):
    """
    Checks for a fist by confirming the four main fingers are down.
    """
    # Fist is detected if index, middle, ring, and pinky fingers are all down (0)
    # Thumb can be up or down for a fist, so we only check fingers[1:]
    return fingers[1:] == [0, 0, 0, 0]

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        img = cv2.flip(img, 1) # Flip horizontally for mirror effect
        h, w, _ = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        current_time = time.time()

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)

            # TWO-HAND LOGIC
            if num_hands == 2:
                # If alt-tab mode was active, release alt key
                if alt_tab_mode:
                    pyautogui.keyUp('alt')
                    alt_tab_mode = False
                was_in_cursor_mode = False # Reset cursor mode state
                
                # Get landmarks and handedness for both hands
                hand1_lms, hand2_lms = results.multi_hand_landmarks[0], results.multi_hand_landmarks[1]
                handedness1, handedness2 = results.multi_handedness[0].classification[0].label, results.multi_handedness[1].classification[0].label
                
                lmList1, lmList2 = [], []
                for id, lm in enumerate(hand1_lms.landmark):
                    lmList1.append([id, int(lm.x * w), int(lm.y * h)])
                for id, lm in enumerate(hand2_lms.landmark):
                    lmList2.append([id, int(lm.x * w), int(lm.y * h)])
                
                # Draw landmarks on the image
                mpDraw.draw_landmarks(img, hand1_lms, mpHands.HAND_CONNECTIONS)
                mpDraw.draw_landmarks(img, hand2_lms, mpHands.HAND_CONNECTIONS)
                
                # Determine extended fingers for both hands
                fingers1, fingers2 = fingers_up(lmList1, handedness1), fingers_up(lmList2, handedness2)
                
                # Gesture to toggle Zoom Mode (both hands open)
                if fingers1 == [1, 1, 1, 1, 1] and fingers2 == [1, 1, 1, 1, 1]:
                    if current_time - last_zoom_toggle_time > ZOOM_TOGGLE_COOLDOWN:
                        zoom_mode_active = not zoom_mode_active
                        last_zoom_toggle_time = current_time
                        print(f"ZOOM MODE {'ACTIVATED' if zoom_mode_active else 'DEACTIVATED'}")
                        prev_zoom_dist, smoothed_zoom_dist = 0, 0 # Reset zoom distance when toggling
                
                # If Zoom Mode is active, perform zoom actions
                if zoom_mode_active:
                    # Get index finger coordinates for both hands
                    x1, y1 = lmList1[8][1], lmList1[8][2]
                    x2, y2 = lmList2[8][1], lmList2[8][2]
                    
                    # Calculate current distance between index fingers
                    current_dist = math.hypot(x2 - x1, y2 - y1)
                    
                    # Initialize or smooth the zoom distance
                    if prev_zoom_dist == 0:
                        prev_zoom_dist, smoothed_zoom_dist = current_dist, current_dist
                    else:
                        smoothed_zoom_dist += (current_dist - smoothed_zoom_dist) * ZOOM_SMOOTH_FACTOR
                    
                    # Apply zoom action with cooldown
                    if current_time - last_zoom_time > ZOOM_COOLDOWN:
                        delta_dist = smoothed_zoom_dist - prev_zoom_dist
                        if delta_dist > ZOOM_SENSITIVITY:
                            pyautogui.hotkey('ctrl', '+') # Zoom In
                            print("Zoom In")
                            prev_zoom_dist, last_zoom_time = smoothed_zoom_dist, current_time
                        elif delta_dist < -ZOOM_SENSITIVITY:
                            pyautogui.hotkey('ctrl', '-') # Zoom Out
                            print("Zoom Out")
                            prev_zoom_dist, last_zoom_time = smoothed_zoom_dist, current_time
                    
                    # Visual feedback for zoom mode
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(img, "ZOOM ACTIVE", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                else:
                    cv2.putText(img, "Show open hands to toggle zoom", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            
            # ONE-HAND LOGIC
            elif num_hands == 1:
                zoom_mode_active = False # Deactivate zoom mode if only one hand is detected
                
                handLms = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0].classification[0].label
                
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    lmList.append([id, int(lm.x * w), int(lm.y * h)])
                
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                
                if lmList:
                    ix, iy = lmList[8][1], lmList[8][2] # Index finger tip coordinates
                    fingers = fingers_up(lmList, handedness)
                    is_fist = is_fist_detected(fingers)

                    # Alt-Tab Mode (Fist gesture)
                    if is_fist and not alt_tab_mode and not was_in_cursor_mode:
                        pyautogui.keyDown('alt')
                        pyautogui.press('tab')
                        alt_tab_mode = True
                        alt_tab_start_x = lmList[0][1] # Store wrist x for swipe detection
                    elif alt_tab_mode:
                        if is_fist:
                            # Swipe detection for Alt-Tab navigation
                            delta_x = lmList[0][1] - alt_tab_start_x
                            if delta_x > ALT_TAB_SWIPE_THRESHOLD:
                                pyautogui.press('tab') # Move to next application
                                alt_tab_start_x = lmList[0][1] # Reset start_x for next swipe
                            elif delta_x < -ALT_TAB_SWIPE_THRESHOLD:
                                pyautogui.hotkey('shift', 'tab') # Move to previous application
                                alt_tab_start_x = lmList[0][1] # Reset start_x for next swipe
                            cv2.putText(img, "ALT-TAB MODE", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
                        else:
                            # Release alt key if fist is no longer detected
                            pyautogui.keyUp('alt')
                            alt_tab_mode = False
                    else:
                        # Cursor Mode (Index finger up)
                        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                            is_scrolling = False # Exit scrolling mode
                            was_in_cursor_mode = True # Indicate cursor mode is active

                            # Map hand position to screen coordinates
                            screen_x = np.interp(ix, (frameR, w - frameR), (0, screen_width))
                            screen_y = np.interp(iy, (frameR, h - frameR), (0, screen_height))
                            
                            # Smooth cursor movement
                            smooth_x = prev_x + (screen_x - prev_x) * smooth_factor
                            smooth_y = prev_y + (screen_y - prev_y) * smooth_factor
                            
                            pyautogui.moveTo(smooth_x, smooth_y)
                            prev_x, prev_y = smooth_x, smooth_y
                            
                            cv2.circle(img, (ix, iy), 10, (255, 0, 255), cv2.FILLED)
                            cv2.putText(img, "CURSOR MODE", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

                        # Volume Up Gesture (Thumb, Index, Middle fingers up) - CHECK THIS FIRST!
                        elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                            was_in_cursor_mode = False # Exit cursor mode
                            click_flag = False # Reset click flag
                            start_hold_time = None # Reset hold time
                            is_scrolling = False # Reset scrolling mode

                            if current_time - last_volume_change_time > VOLUME_COOLDOWN:
                                pyautogui.press('volumeup')
                                print("Volume Up!")
                                last_volume_change_time = current_time
                            cv2.putText(img, "VOLUME UP", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)

                        # Click/Scroll Mode (Index and Middle fingers up) - NOW SECOND
                        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                            was_in_cursor_mode = False # Exit cursor mode
                            
                            if start_hold_time is None:
                                # Initialize for click/scroll detection
                                start_hold_time = current_time
                                scroll_start_y = iy
                                is_scrolling = False
                                click_flag = False
                            
                            delta_y = iy - scroll_start_y # Vertical movement for scrolling
                            
                            if is_scrolling:
                                # Continuous scrolling if threshold is met
                                if abs(delta_y) > SCROLL_THRESHOLD:
                                    pyautogui.scroll(SCROLL_AMOUNT if delta_y < 0 else -SCROLL_AMOUNT)
                                    print(f"Scrolling {'Up' if delta_y < 0 else 'Down'}")
                                    scroll_start_y = iy # Update start_y for continuous motion
                                cv2.putText(img, "SCROLLING", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)
                            else:
                                # Check if movement exceeds scroll threshold
                                if abs(delta_y) > SCROLL_THRESHOLD:
                                    is_scrolling = True
                                    print("Scrolling Mode Activated")
                                # If not scrolling and hold duration met, perform click
                                elif current_time - start_hold_time > REQUIRED_HOLD_DURATION and not click_flag:
                                    pyautogui.click()
                                    print("Click!")
                                    click_flag = True # Prevent multiple clicks for one hold
                                    start_hold_time = None # Reset hold time
                                cv2.putText(img, "CLICK/SCROLL MODE", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)
                            
                            # Visual feedback for click hold
                            if not is_scrolling and not click_flag and start_hold_time:
                                cv2.circle(img, (ix, iy), 15, (0, 165, 255), 2) # Orange circle for hold

                        # Volume Down Gesture (Thumb, Index, Pinky fingers up)
                        elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:
                            was_in_cursor_mode = False # Exit cursor mode
                            click_flag = False # Reset click flag
                            start_hold_time = None # Reset hold time
                            is_scrolling = False # Reset scrolling mode

                            if current_time - last_volume_change_time > VOLUME_COOLDOWN:
                                pyautogui.press('volumedown')
                                print("Volume Down!")
                                last_volume_change_time = current_time
                            cv2.putText(img, "VOLUME DOWN", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)

                        else:
                            # Reset states if no specific gesture is detected
                            was_in_cursor_mode, click_flag, start_hold_time, is_scrolling = False, False, None, False
        else:
            # If no hands are detected, reset all modes
            if alt_tab_mode:
                pyautogui.keyUp('alt')
                alt_tab_mode = False
            zoom_mode_active = False
            is_scrolling = False
            was_in_cursor_mode = False # Reset cursor mode state

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
    # Ensure alt key is released on exit if it was held down
    if alt_tab_mode:
        pyautogui.keyUp('alt')
    cap.release()
    cv2.destroyAllWindows()
