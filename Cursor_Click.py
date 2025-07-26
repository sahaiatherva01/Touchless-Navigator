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
# max_num_hands=1: Detects only one hand to avoid confusion
# min_detection_confidence: Minimum confidence for hand detection
# min_tracking_confidence: Minimum confidence for hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Screen size for mouse control
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = True  # Move mouse to corner to stop program (useful for debugging)
pyautogui.PAUSE = 0.01     # Small delay for smoother control across all pyautogui actions

# Smoothing parameters for cursor movement
# Increased smooth_factor for more cursor stability.
# A higher value means the cursor will move more smoothly and be less sensitive to small jitters.
smooth_factor = 0.9 # Adjusted from 0.75 to 0.9 for significantly increased stability
prev_x, prev_y = 0, 0

# Click gesture parameters
click_flag = False # Flag to prevent multiple clicks for a single gesture
start_hold_time = None # Stores the time when the hold gesture started
hold_position = (0, 0) # Stores the position where the hold gesture started
required_hold_duration = 0.7 # Time in seconds to hold still for a click
stillness_threshold = 30   # Max pixel movement for "stillness" within the camera feed

def fingers_up(lmList):
    """
    Determines which fingers are extended (up) based on landmark positions.
    Args:
        lmList (list): A list of hand landmarks, where each landmark is [id, x, y].
    Returns:
        list: A list of 0s and 1s, where 1 means the finger is up and 0 means it's down.
              [Thumb, Index, Middle, Ring, Pinky]
    """
    fingers = []
    tips_ids = [4, 8, 12, 16, 20]  # Landmark IDs for finger tips

    # Thumb check: Compare x-coordinate of thumb tip (4) with its base (3)
    # This accounts for hand orientation (left vs. right hand)
    # For right hand, thumb tip x > thumb base x if extended. For left hand, it's the opposite.
    # A more robust check for thumb extension relative to its base.
    if lmList[tips_ids[0]][1] > lmList[tips_ids[0] - 1][1]: # Assuming right hand, thumb extends right
        fingers.append(1)
    else: # Assuming left hand or thumb is bent
        fingers.append(0)

    # Other fingers (Index, Middle, Ring, Pinky): Compare y-coordinate of tip with the knuckle below it
    # If tip y is less than knuckle y, the finger is extended upwards.
    for id in range(1, 5):
        if lmList[tips_ids[id]][2] < lmList[tips_ids[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def get_hand_size(lmList):
    """
    Estimates the size of the hand to dynamically adjust click thresholds.
    Calculated as the distance between the wrist (landmark 0) and the MCP of the middle finger (landmark 9).
    Args:
        lmList (list): A list of hand landmarks.
    Returns:
        float: The estimated hand size.
    """
    return math.hypot(lmList[0][1] - lmList[9][1], lmList[0][2] - lmList[9][2])

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image.")
            break
        img = cv2.flip(img, 1)  # Flip horizontally for a natural mirror effect
        h, w, _ = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for MediaPipe
        results = hands.process(imgRGB) # Process the image to find hand landmarks

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lmList = []
                # Extract landmark coordinates and store them in lmList
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                # Draw landmarks and connections on the image
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                if lmList:
                    fingers = fingers_up(lmList) # Get the state of all fingers
                    current_time = time.time() # Get current time for cooldowns

                    # Cursor movement: ONLY if the index finger is up (fingers[1] == 1)
                    if fingers[1] == 1: # Check if index finger is extended
                        x, y = lmList[8][1], lmList[8][2] # Coordinates of index finger tip (landmark 8)
                        
                        # Smooth cursor movement using linear interpolation
                        smooth_x = prev_x + (x - prev_x) * smooth_factor
                        smooth_y = prev_y + (y - prev_y) * smooth_factor
                        prev_x, prev_y = smooth_x, smooth_y # Update previous coordinates

                        # Map camera coordinates to screen coordinates and clip to screen boundaries
                        screen_x = np.clip(screen_width / w * smooth_x, 0, screen_width)
                        screen_y = np.clip(screen_height / h * smooth_y, 0, screen_height)
                        pyautogui.moveTo(screen_x, screen_y) # Move the mouse cursor

                        # Visual feedback for index finger tracking (Magenta circle)
                        cv2.circle(img, (lmList[8][1], lmList[8][2]), 10, (255, 0, 255), cv2.FILLED)
                    else:
                        # If index finger is not up, reset smoothing to current position
                        # This prevents sudden jumps when the index finger becomes visible again
                        if lmList[8][1] != 0 or lmList[8][2] != 0: # Check if landmark is valid
                            prev_x, prev_y = lmList[8][1], lmList[8][2]
                        # If index finger is not up, reset click hold state as movement is not active
                        start_hold_time = None
                        click_flag = False


                    # Click gesture: Index and Middle finger up AND held still
                    if fingers[1] == 1 and fingers[2] == 1: # Both index (1) and middle (2) fingers are up
                        current_x, current_y = lmList[8][1], lmList[8][2] # Use index finger tip for stillness check

                        if start_hold_time is None:
                            # Start holding timer if not already started
                            start_hold_time = time.time()
                            hold_position = (current_x, current_y)
                            # Visual feedback: Orange outline when starting to hold
                            cv2.circle(img, (current_x, current_y), 15, (0, 165, 255), 2) 
                        else:
                            # Calculate distance from initial hold position to check for stillness
                            distance_from_hold = math.hypot(current_x - hold_position[0], current_y - hold_position[1])

                            if distance_from_hold < stillness_threshold:
                                # If still within threshold, check if hold duration is met
                                if time.time() - start_hold_time > required_hold_duration:
                                    if not click_flag: # Only click if the flag is not set (prevents continuous clicks)
                                        pyautogui.click() # Perform a left click
                                        print("Click! (Index & Middle Finger Hold)")
                                        click_flag = True # Set flag to prevent further clicks until gesture changes
                                        # Visual feedback: Green filled circle when click occurs
                                        cv2.circle(img, (current_x, current_y), 15, (0, 255, 0), cv2.FILLED) 
                                else:
                                    # Still holding, show progress (optional: could draw a shrinking circle)
                                    # Visual feedback: Orange outline while holding
                                    cv2.circle(img, (current_x, current_y), 15, (0, 165, 255), 2) 
                            else:
                                # Moved too much, reset hold timer and click flag
                                start_hold_time = None
                                click_flag = False
                                # Visual feedback: Red outline if movement detected during hold
                                cv2.circle(img, (current_x, current_y), 15, (0, 0, 255), 2) 
                    else:
                        # If fingers are not in click gesture, reset hold state
                        start_hold_time = None
                        click_flag = False

        # Display the video feed
        cv2.imshow("NEXUS Control", img)

        # Press 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release the camera and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
