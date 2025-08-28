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

# --- Mudra Identification Logic ---

def fingers_up(lmList):
    """
    Determines which fingers are extended (up) based on MediaPipe landmarks.
    """
    fingers = []
    # Thumb tip landmark is 4, base of thumb is 2.
    # Check if thumb tip is to the right of its base (for right hand)
    # or left of its base (for left hand).
    if lmList[4][1] > lmList[3][1]:  # Assuming right hand by default
        if lmList[4][1] > lmList[4][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    else: # Assuming left hand
        if lmList[4][1] < lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    # For other 4 fingers, check if tip (8, 12, 16, 20) is above the first joint (6, 10, 14, 18)
    tips_ids = [8, 12, 16, 20]
    for id in tips_ids:
        if lmList[id][2] < lmList[id - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def get_mudra_name(lmList):
    """
    Identifies the mudra based on the hand landmarks and finger positions.
    Returns the mudra name or 'Unknown'.
    """
    fingers = fingers_up(lmList)

    # A simple, rule-based approach to identify common mudras.
    # This can be expanded with more complex logic or a trained model.

    # Pataka Mudra: All fingers straight, close together.
    # We check if all fingers are up.
    if fingers == [1, 1, 1, 1, 1]:
            return "Pataka"

    # Tripataka Mudra: Ring finger bent down.
    # Logic: Index, Middle, Pinky fingers up, Ring finger down.
    if fingers == [1, 1, 1, 0, 1]:
        return "Tripataka"

    # Mayura Mudra: Thumb and ring finger touch.
    # We check the distance between the tips of the thumb and ring finger.
    dist_thumb_ring = math.hypot(lmList[4][1] - lmList[16][1], lmList[4][2] - lmList[16][2])
    if dist_thumb_ring < 50: # Small distance means they are touching
        return "Mayura"
    
    # Sukhaputaka Mudra: Index finger and thumb form a circle.
    # We check the distance between the tips of the index finger and thumb.
    dist_thumb_index = math.hypot(lmList[4][1] - lmList[8][1], lmList[4][2] - lmList[8][2])
    if dist_thumb_index < 50:
        return "Sukhaputaka"

    # Other common gestures from the original code that are not mudras
    if fingers == [0, 0, 0, 0, 0]:
        return "Mushti"
    if fingers == [0, 1, 0, 0, 0]:
        return "Suchi"
    if fingers == [1, 0, 0, 0, 0]:
        return "Shikara"
    if fingers == [0, 1, 1, 1, 0]:
        return "Trishula"
    
    return "Unknown Mudra"

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
        
        mudra_name = "No Hand Detected"

        if results.multi_hand_landmarks:
            handLms = results.multi_hand_landmarks[0]
            
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                lmList.append([id, int(lm.x * w), int(lm.y * h)])
            
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
            # Identify the mudra
            if lmList:
                mudra_name = get_mudra_name(lmList)

        # Display the identified mudra name on the screen
        cv2.putText(img, f"Mudra: {mudra_name}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)
        
        cv2.imshow("Mudra Identifier", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
