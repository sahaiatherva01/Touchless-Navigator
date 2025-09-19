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
    """
    fingers = []
    tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky

    # Thumb detection
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

    # Other 4 fingers
    for id in range(1, 5):
        if lmList[tip_ids[id]][2] < lmList[tip_ids[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
            
    return fingers

def get_mudra_name(lmList, hand_handedness):
    """
    Identifies a specific hand mudra from the Sattriya classical dance tradition
    based on the positions of hand landmarks detected by MediaPipe.
    """
    if not lmList:
        return "Unknown"

    # Calculate required distances between landmarks
    dist_thumb_index = math.hypot(lmList[4][1] - lmList[8][1], lmList[4][2] - lmList[8][2])
    dist_thumb_middle = math.hypot(lmList[4][1] - lmList[12][1], lmList[4][2] - lmList[12][2])
    dist_thumb_ring = math.hypot(lmList[4][1] - lmList[16][1], lmList[4][2] - lmList[16][2])
    dist_thumb_pinky = math.hypot(lmList[4][1] - lmList[20][1], lmList[4][2] - lmList[20][2])
    dist_index_middle = math.hypot(lmList[8][1] - lmList[12][1], lmList[8][2] - lmList[12][2])
    dist_index_pinky = math.hypot(lmList[8][1] - lmList[20][1], lmList[8][2] - lmList[20][2])
    
    fingers = fingers_up(lmList, hand_handedness)

    # --- High Priority & Complex Mudras ---

    # Mukula (Bud)
    # Represents a bud, particularly a lotus bud or a water lily.
    # Condition: All fingertips are brought close to the thumb tip.
    if (dist_thumb_index < 35 and dist_thumb_middle < 35 and 
        dist_thumb_ring < 35 and dist_thumb_pinky < 35):
        return "Mukula"

    # Kataka Mukha (Opening of a bracelet)
    # Often used to hold flowers, play the flute, or represent a bird's head.
    # Condition: Ring and Pinky fingers are up; Index and Middle fingers touch the thumb.
    if (fingers[3] == 1 and fingers[4] == 1 and fingers[1] == 0 and fingers[2] == 0 and
        dist_thumb_index < 50 and dist_thumb_middle < 50):
        return "Kataka Mukha"

    # Alapadma (Full-blown lotus)
    # Represents a fully bloomed lotus, fruits, beauty, or a circular shape.
    # Condition: All fingers are up and fanned out widely.
    if fingers == [1, 1, 1, 1, 1] and dist_index_pinky > 140:
        return "Alapadma"

    # Simhamukha (Lion's Face)
    # Represents a lion's face, often used to show courage or power.
    # Condition: Index and Pinky are up; Middle and Ring fingers touch the thumb.
    if (fingers[1] == 1 and fingers[4] == 1 and fingers[2] == 0 and fingers[3] == 0 and
        dist_thumb_middle < 45 and dist_thumb_ring < 45):
        return "Simhamukha"
        
    # Sandamsa (Pincers)
    # Represents grasping, pincers, tweezers, or drawing something out.
    # Condition: Thumb and index tips touch, while other fingers are closed.
    if dist_thumb_index < 25 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
        return "Sandamsa"

    # --- Standard Mudras ---

    # Chandrakala (Digit of the Moon)
    # Represents the crescent moon, often seen on Lord Shiva's head.
    # Condition: Thumb and Index finger are up, forming an 'L' shape.
    if fingers == [1, 1, 0, 0, 0]:
        return "Chandrakala"
        
    # Chatura (Clever)
    # Represents cleverness, musk, a small quantity, or breaking things.
    # Condition: All fingers are up except for the pinky.
    if fingers == [1, 1, 1, 1, 0]:
        return "Chatura"

    # Shukatunda (Parrot's Beak)
    # Represents a parrot's beak or the act of shooting an arrow.
    # Condition: A specific finger combination where Thumb, Middle, and Pinky are up.
    if fingers == [1, 0, 1, 0, 1]:
        return "Shukatunda"

    # Kartarimukha (Scissor's Face)
    # Represents scissors, separation, the corner of an eye, or two different things.
    # Condition: Index and Middle fingers are up and spread apart like scissors.
    if fingers == [0, 1, 1, 0, 0] and dist_index_middle > 40:
        return "Kartarimukha"
        
    # Mrigashirsha (Deer's Head)
    # Represents the head of a deer, often used to show animals or costumes.
    # Condition: Thumb and Pinky finger are up.
    if fingers == [1, 0, 0, 0, 1]:
        return "Mrigashirsha"

    # Ardhachandra (Half Moon) and Pataka (Flag)
    # Pataka is the foundational "flag" gesture. Ardhachandra is a variation.
    # Condition: All fingers are up. They are differentiated by how far the thumb is spread out.
    if fingers == [1, 1, 1, 1, 1]:
        thumb_index_dist_x = abs(lmList[4][1] - lmList[5][1])
        if thumb_index_dist_x > 60:
            return "Ardhachandra"
        else:
            return "Pataka"

    # Hamsasya (Swan Beak)
    # Represents a swan's beak, softness, tying a knot, or giving instruction.
    # Condition: Thumb and Index tips are close, while Middle, Ring, and Pinky are straight up.
    if dist_thumb_index < 45 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
        return "Hamsasya"
    
    # Tripataka (Three Parts of a Flag)
    # A variation of Pataka, often used for crowns, trees, or drawing lines.
    # Condition: All fingers are up except for the Ring finger.
    if fingers == [1, 1, 1, 0, 1]:
        return "Tripataka"
    
    # Mayura (Peacock)
    # Represents a peacock's beak or neck, or the act of writing.
    # Condition: Thumb and Ring finger tips are close together.
    if dist_thumb_ring < 45:
        return "Mayura"
    
    # Suchimukha (Needle Face)
    # Represents a needle, the number one, pointing, or demonstration.
    # Condition: Index finger is pointing up, while others are in a fist.
    if fingers == [0, 1, 0, 0, 0]:
        return "Suchimukha"
        
    # Shikara (Peak)
    # Represents a peak, a spire, a "thumbs-up", or the act of holding a bow.
    # Condition: Thumb is pointing up, while others are in a fist.
    if fingers == [1, 0, 0, 0, 0]:
        return "Shikara"
        
    # Trishula (Trident)
    # Represents a trident, a symbol of Lord Shiva, or the number three.
    # Condition: Index, Middle, and Ring fingers are up, resembling a trident.
    if fingers == [0, 1, 1, 1, 0]:
        return "Trishula"
        
    # Ardhapataka (Half Flag)
    # Represents a knife, dagger, tower, or the number two.
    # Condition: Index and Middle fingers are up together.
    if fingers == [0, 1, 1, 0, 0]:
        return "Ardhapataka"
        
    # Arala (Bent)
    # Represents drinking poison or nectar, or a violent wind.
    # Condition: All fingers are up except for the Index finger, which is bent.
    if fingers == [1, 0, 1, 1, 1]:
        return "Arala"
        
    # Bhramara (Bee)
    # Represents a bee, yoga, a wing, or the act of plucking flowers.
    # Condition: Thumb and Middle finger tips are close, with the index finger curled.
    if fingers == [1, 0, 1, 0, 0]:
        return "Bhramara"

    # --- CONSOLIDATED FIST LOGIC FOR ALL VARIATIONS ---
    # This block handles all fist-like gestures ('fingers' array is [0,0,0,0,0]).
    # It checks for specific variations first, then defaults to the standard fist.
    if fingers == [0, 0, 0, 0, 0]:
        # Padmakosha (Lotus Bud)
        # Represents a lotus bud, fruit, or a ball-like object.
        # Condition: A cupped hand where fingertips are further from the wrist than in a tight fist.
        if abs(lmList[0][2] - lmList[8][2]) > 100:
            return "Padmakosha"
            
        # Kapittha (Elephant Apple)
        # Represents the elephant apple fruit, holding cymbals, or milking cows.
        # Condition: A fist where the thumb is covered by the index finger.
        if dist_thumb_index < 45 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            return "Kapittha"
            
        # Mushti (Fist)
        # The standard fist, representing steadiness, grasping, or combat.
        # Condition: The default "all fingers down" gesture if no other variation is detected.
        return "Mushti"
    
    return "Unknown Mudra"

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        
        img = cv2.flip(img, 1) # Flip horizontally for mirror effect
        h, w, c = img.shape
        # --- THIS IS THE CORRECTED LINE ---
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        mudra_name = "No Hand Detected"

        if results.multi_hand_landmarks:
            for handLms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, px, py])
                
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                
                hand_label = handedness.classification[0].label
                
                if lmList:
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
