import cv2
import mediapipe as mp
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
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# --- CORRECTED Mudra Identification Logic ---

def fingers_up(lmList, hand_handedness):
    """
    Determines which fingers are extended (up).
    FIX: Now correctly uses handedness (Left vs. Right) for accurate thumb detection.
    """
    fingers = []
    tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky

    # --- Thumb ---
    # A more robust check based on the thumb's x-coordinate relative to its base.
    if hand_handedness == 'Right':
        if lmList[tip_ids[0]][1] < lmList[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    elif hand_handedness == 'Left':
        if lmList[tip_ids[0]][1] > lmList[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    # --- Other 4 Fingers ---
    # Check if the tip's y-coordinate is above the joint's y-coordinate.
    for id in range(1, 5):
        if lmList[tip_ids[id]][2] < lmList[tip_ids[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
            
    return fingers

def get_mudra_name(lmList, hand_handedness):
    """
    Identifies the mudra based on landmarks.
    FIX: Corrected logical errors and structure.
    """
    if not lmList:
        return "Unknown"

    # CRASH FIX: All distance calculations are moved to the top
    # to prevent NameError.
    dist_thumb_index = math.hypot(lmList[4][1] - lmList[8][1], lmList[4][2] - lmList[8][2])
    dist_thumb_ring = math.hypot(lmList[4][1] - lmList[16][1], lmList[4][2] - lmList[16][2])
    
    fingers = fingers_up(lmList, hand_handedness)

    # LOGIC FIX: Restructured this block with if/elif/else to correctly
    # identify mutually exclusive mudras that all have fingers up.
    if fingers == [1, 1, 1, 1, 1]:
        thumb_index_dist_x = abs(lmList[4][1] - lmList[5][1])
        if thumb_index_dist_x > 60:
            return "Ardhachandra"
        else:
            # You can add logic for Hamsapaksha here if needed
            return "Pataka"

    # Kapittha (Elephant Apple)
    if dist_thumb_index < 45 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
        return "Kapittha"
    

    #Check for Hamsasya (the most specific case)
    # Condition: Thumb and Index tips are close, AND Middle, Ring, and Pinky are straight up.
    if dist_thumb_index < 45 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
        return "Hamsasya"

    if fingers == [1, 1, 1, 0, 1]:
        return "Tripataka"

    if dist_thumb_ring < 45:
        return "Mayura"

    if fingers == [0, 0, 0, 0, 0]:
        return "Mushti"
    if fingers == [0, 1, 0, 0, 0]:
        return "Suchimukha"
    if fingers == [1, 0, 0, 0, 0]:
        return "Shikara"
    if fingers == [0, 1, 1, 1, 0]:
        return "Trishula"
    if fingers == [0, 1, 1, 0, 0]:
        return "Ardhapataka"
    if fingers == [1, 0, 0, 0, 1]:
        return "Dhanu"
    if fingers == [1, 0, 1, 1, 1]:
        return "Arala"
    if fingers == [1, 0, 1, 0, 0]:  
        return "Bhramara"
    
    return "Unknown Mudra"

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        
        img = cv2.flip(img, 1) # Flip horizontally for mirror effect
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        mudra_name = "No Hand Detected"

        if results.multi_hand_landmarks:
            # Loop through detected hands (even though we set max_num_hands=1)
            for handLms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, px, py])
                
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                
                # Get hand orientation (Left or Right)
                hand_label = handedness.classification[0].label
                
                if lmList:
                    # Pass both landmarks and handedness to the function
                    mudra_name = get_mudra_name(lmList, hand_label)

        # Display the identified mudra name on the screen
        cv2.putText(img, mudra_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        cv2.imshow("Mudra Identifier", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
