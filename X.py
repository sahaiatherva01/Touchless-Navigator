import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np

# --- System and PyAutoGUI Setup ---
# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Get screen size for mapping
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = True  # Allows cursor to move to screen edges
pyautogui.PAUSE = 0.01         # No pause between PyAutoGUI calls

# --- MediaPipe Hands Initialization ---
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils

# --- Gesture Control Parameters ---
# General
SMOOTHING = 0.2
FRAME_REDUCTION = 100  # Reduces the active area for cursor control
last_action_time = 0.0  # Last time an action was performed
ACTION_COOLDOWN = 0.5  # Cooldown between different gesture actions

# State variables
prev_x, prev_y = 0, 0
current_mode = "NONE"

# Cursor Mode
cursor_active = False

# Click Mode
# We need to track if a click was already performed for the current gesture
click_performed = False

# Scroll Mode
scroll_start_y = 0
SCROLL_SENSITIVITY = 40  # Pixels hand must move to trigger a scroll
SCROLL_AMOUNT = 100      # How much to scroll
is_scrolling = False     # State for continuous scrolling

# Volume Control
volume_cooldown_active = False
VOLUME_COOLDOWN = 0.3 # Cooldown to prevent rapid volume changes
last_volume_change_time = 0

# Alt-Tab Mode
alt_tab_active = False
alt_tab_start_x = 0
ALT_TAB_SWIPE_THRESHOLD = 60
was_in_cursor_mode = False
ALT_TAB_SENSITIVITY = 70 # Pixels fist must move to switch tabs

# Zoom Mode (Two Hands)
zoom_mode_active = False
ZOOM_TOGGLE_COOLDOWN = 1.5
last_zoom_toggle_time = 0
zoom_initial_dist = 0
prev_zoom_dist, smoothed_zoom_dist = 0, 0
ZOOM_SENSITIVITY = 5.0
ZOOM_SMOOTH_FACTOR = 0.3
ZOOM_COOLDOWN = 0.4
last_zoom_time = 0

# --- Helper Function ---
def get_finger_status(lmList, handedness):
    """
    Determines which fingers are up or down.
    Returns a list of 5 booleans (thumb, index, middle, ring, pinky).
    """
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]
    
    # Thumb check (based on x-coordinate relative to wrist)
    # This is a more robust check than comparing to the previous landmark
    if handedness == "Right":
        if lmList[tip_ids[0]][1] < lmList[tip_ids[0] - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # Left Hand
        if lmList[tip_ids[0]][1] > lmList[tip_ids[0] - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
    # Four other fingers (based on y-coordinate)
    for id in range(1, 5):
        if lmList[tip_ids[id]][2] < lmList[tip_ids[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
            
    return fingers

def is_fist_detected(fingers):
    """
    Checks for a fist by confirming the four main fingers are down.
    """
    # Fist is detected if index, middle, ring, and pinky fingers are all down (0)
    # Thumb can be up or down for a fist, so we only check fingers[1:]
    return fingers[1:] == [0, 0, 0, 0]

# --- Main Loop ---
try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame.")
            continue
        
        # Flip the image horizontally for a later selfie-view display
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        
        # Process the image and find hands
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        current_time = time.time()
        
        # Default mode if no hands or gestures are detected
        new_mode = "NONE"

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            
            # 🖐️🖐️ TWO-HAND GESTURE: ZOOM
            if num_hands == 2:
                # If alt-tab mode was active, release alt key
                if alt_tab_mode:
                    pyautogui.keyUp('alt')
                    alt_tab_mode = False
                was_in_cursor_mode = False # Reset cursor mode state
                
                # Get landmarks for both hands
                hand1_lms = results.multi_hand_landmarks[0]
                hand2_lms = results.multi_hand_landmarks[1]
                
                # Draw landmarks for visualization
                mpDraw.draw_landmarks(img, hand1_lms, mpHands.HAND_CONNECTIONS)
                mpDraw.draw_landmarks(img, hand2_lms, mpHands.HAND_CONNECTIONS)
                
                lmList1 = [[id, int(lm.x * w), int(lm.y * h)] for id, lm in enumerate(hand1_lms.landmark)]
                lmList2 = [[id, int(lm.x * w), int(lm.y * h)] for id, lm in enumerate(hand2_lms.landmark)]

                fingers1 = get_finger_status(lmList1, results.multi_handedness[0].classification[0].label)
                fingers2 = get_finger_status(lmList2, results.multi_handedness[1].classification[0].label)

                # Both hands must be open to activate/use zoom
                if fingers1 == [1, 1, 1, 1, 1] and fingers2 == [1, 1, 1, 1, 1]:
                    new_mode = "ZOOM"
                    if not zoom_active:
                        # Initialize zoom when gesture starts
                        zoom_active = True
                        ix1, iy1 = lmList1[8][1], lmList1[8][2]
                        ix2, iy2 = lmList2[8][1], lmList2[8][2]
                        zoom_initial_dist = math.hypot(ix2 - ix1, iy2 - iy1)
                    else:
                        # Continue zoom
                        ix1, iy1 = lmList1[8][1], lmList1[8][2]
                        ix2, iy2 = lmList2[8][1], lmList2[8][2]
                        current_dist = math.hypot(ix2 - ix1, iy2 - iy1)
                        
                        cv2.line(img, (ix1, iy1), (ix2, iy2), (255, 0, 255), 3)

                        # Calculate zoom factor
                        zoom_factor = current_dist / zoom_initial_dist

                        if zoom_factor > 1 + ZOOM_SENSITIVITY:
                            pyautogui.hotkey('ctrl', '+')
                            zoom_initial_dist = current_dist # Reset base distance
                        elif zoom_factor < 1 - ZOOM_SENSITIVITY:
                            pyautogui.hotkey('ctrl', '-')
                            zoom_initial_dist = current_dist # Reset base distance
                else:
                    zoom_active = False

            # 🖐️ ONE-HAND GESTURES
            elif num_hands == 1:
                zoom_active = False # Deactivate zoom if only one hand is visible
                hand_landmarks = results.multi_hand_landmarks[0]
                mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)
                
                lmList = [[id, int(lm.x * w), int(lm.y * h)] for id, lm in enumerate(hand_landmarks.landmark)]
                handedness = results.multi_handedness[0].classification[0].label
                fingers = get_finger_status(lmList, handedness)
                
                # Get coordinates of the index finger tip for cursor/scroll
                ix, iy = lmList[8][1], lmList[8][2]
                
                # --- Gesture Recognition (Strict Hierarchy) ---
                
                # 📜 SCROLL MODE: Index, Middle, Ring up
                if fingers == [0, 1, 1, 1, 0]:
                    new_mode = "SCROLL"
                    if current_mode != "SCROLL":
                        scroll_start_y = iy # Initialize scroll position
                    else:
                        delta_y = iy - scroll_start_y
                        if abs(delta_y) > SCROLL_SENSITIVITY:
                            # Move hand UP to scroll DOWN (natural)
                            pyautogui.scroll(-SCROLL_AMOUNT if delta_y < 0 else SCROLL_AMOUNT)
                            scroll_start_y = iy # Reset position for continuous scroll
                
                # 👍 VOLUME UP: Thumb, Index, Middle up
                elif fingers == [1, 1, 1, 0, 0]:
                    new_mode = "VOLUME_UP"
                    if current_mode != "VOLUME_UP": # Trigger once on gesture change
                         pyautogui.press('volumeup')
                
                # 🤘 VOLUME DOWN: Thumb, Index, Pinky up
                elif fingers == [1, 1, 0, 0, 1]:
                    new_mode = "VOLUME_DOWN"
                    if current_mode != "VOLUME_DOWN": # Trigger once on gesture change
                        pyautogui.press('volumedown')

                # 🖱️ CLICK MODE: Index, Middle up
                elif fingers == [0, 1, 1, 0, 0]:
                    new_mode = "CLICK"
                    if not click_performed and current_time - last_action_time > ACTION_COOLDOWN:
                        pyautogui.click()
                        click_performed = True # Prevent multiple clicks
                        last_action_time = current_time

                # 👆 CURSOR MODE: Index up
                elif fingers == [0, 1, 0, 0, 0]:
                    new_mode = "CURSOR"
                    cursor_active = True
                    # Convert coordinates to screen space
                    screen_x = np.interp(ix, (FRAME_REDUCTION, w - FRAME_REDUCTION), (0, screen_width))
                    screen_y = np.interp(iy, (FRAME_REDUCTION, h - FRAME_REDUCTION), (0, screen_height))
                    
                    # Smooth the movement
                    smooth_x = prev_x + (screen_x - prev_x) * SMOOTHING
                    smooth_y = prev_y + (screen_y - prev_y) * SMOOTHING
                    
                    pyautogui.moveTo(smooth_x, smooth_y)
                    prev_x, prev_y = smooth_x, smooth_y
                    cv2.circle(img, (ix, iy), 10, (255, 0, 255), cv2.FILLED)
                
                # ✊ ALT-TAB MODE: Fist
                elif fingers == [0, 0, 0, 0, 0]:
                    new_mode = "ALT_TAB"
                    if not alt_tab_active:
                        # Enter alt-tab mode
                        alt_tab_active = True
                        pyautogui.keyDown('alt')
                        pyautogui.press('tab')
                        alt_tab_start_x = lmList[9][1] # Use middle knuckle for stability
                    else:
                        # Navigate while in alt-tab mode
                        current_x = lmList[9][1]
                        delta_x = current_x - alt_tab_start_x
                        if delta_x > ALT_TAB_SENSITIVITY:
                            pyautogui.press('tab')
                            alt_tab_start_x = current_x
                        elif delta_x < -ALT_TAB_SENSITIVITY:
                            pyautogui.hotkey('shift', 'tab')
                            alt_tab_start_x = current_x

        # --- State Management & Cleanup ---
        # If the detected gesture is different from the previous frame's
        if new_mode != current_mode:
            # Release Alt key when exiting Alt-Tab mode
            if current_mode == "ALT_TAB":
                pyautogui.keyUp('alt')
                alt_tab_active = False
            
            # Reset click flag when leaving click mode
            if current_mode == "CLICK":
                click_performed = False

            # Reset cursor active flag
            cursor_active = (new_mode == "CURSOR")
            
            # Update the current mode for the next frame
            current_mode = new_mode
            last_action_time = current_time

        # If no hands are detected, reset everything
        if not results.multi_hand_landmarks:
             if alt_tab_active:
                pyautogui.keyUp('alt')
                alt_tab_active = False
             current_mode = "NONE"
             zoom_active = False
             click_performed = False
             cursor_active = False

        # --- Display Information on Screen ---
        cv2.rectangle(img, (10, 10), (350, 60), (0, 0, 0), cv2.FILLED)
        display_text = f"MODE: {current_mode}"
        cv2.putText(img, display_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Gesture Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # --- Cleanup ---
    print("Cleaning up and closing...")
    # Ensure any pressed keys are released on exit
    if alt_tab_active:
        pyautogui.keyUp('alt')
    
    cap.release()
    cv2.destroyAllWindows()
