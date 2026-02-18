import cv2
import mediapipe as mp
import math

# --- Data for Mudras ---
mudra_descriptions = {
    "Mukula": {"desc": "Meaning 'bud.' Represents a water lily or lotus bud, the act of eating, or worship. Formed by bringing all five fingertips together."},
    "Kataka Mukha": {"desc": "Meaning 'opening of a bracelet.' Used to show plucking flowers, holding a necklace, or drawing a bowstring. Formed by joining the index, middle finger, and thumb."},
    "Alapadma": {"desc": "Meaning 'full-blown lotus.' Represents a bloomed lotus, beauty, or asking a question. Formed by fanning out all fingers."},
    "Simhamukha": {"desc": "Meaning 'lion's face.' Represents a lion, courage, strength, or a pearl. Formed by touching the middle and ring fingers to the thumb."},
    "Sandamsa": {"desc": "Meaning 'pincers.' Represents grasping with precision or plucking. Formed by bringing the thumb and index finger together."},
    "Chandrakala": {"desc": "Meaning 'crescent moon.' Represents the moon on Lord Shiva's head. Formed by extending the thumb and index finger STRAIGHT into an 'L' shape."},
    "Chatura": {"desc": "Meaning 'clever' or 'square.' Represents musk or cleverness. Formed with all fingers extended except pinky, thumb at base of middle finger."},
    "Shukatunda": {"desc": "Meaning 'parrot's beak.' Used to represent a hook or shooting an arrow."},
    "Kartarimukha": {"desc": "Meaning 'scissor's face.' Represents scissors or separation. Formed by extending the index and middle fingers."},
    "Mrigashirsha": {"desc": "Meaning 'deer's head.' Represents a deer or cheeks. Formed by extending the thumb and pinky finger upwards."},
    "Ardhachandra": {"desc": "Meaning 'half-moon.' Represents the sky or waist. Formed like Pataka but with the thumb extended outwards."},
    "Pataka": {"desc": "The foundational 'flag' gesture. Signifies blessings or a forest. Formed by keeping all fingers straight and together."},
    "Hamsasya": {"desc": "Meaning 'swan's beak.' Represents softness or tying a knot. Formed by joining the tips of the thumb and index finger."},
    "Tripataka": {"desc": "Meaning 'three parts of a flag.' Used for crowns or trees. Formed like Pataka, but with the ring finger bent."},
    "Mayura": {"desc": "Meaning 'peacock.' Represents a peacock. Formed by joining the tips of the thumb and ring finger."},
    "Suchimukha": {"desc": "Meaning 'needle face.' Represents a needle or pointing. Formed by extending ONLY the index finger, thumb must be folded/down."},
    "Shikara": {"desc": "Meaning 'peak.' A thumbs-up gesture used for determination or holding a bow."},
    "Trishula": {"desc": "Meaning 'trident.' Represents Lord Shiva's weapon. Formed by raising index, middle, and ring fingers."},
    "Ardhapataka": {"desc": "Meaning 'half-flag.' Represents a knife or the number two. Formed by extending index and middle fingers together."},
    "Arala": {"desc": "Meaning 'bent.' Represents drinking nectar or violent wind. Formed by bending the index finger from Pataka."},
    "Bhramara": {"desc": "Meaning 'bee.' Represents a bee or yoga. Formed by thumb and middle finger joining, index curled."},
    "Padmakosha": {"desc": "Meaning 'lotus bud.' Represents a fruit or a ball. Formed by cupping the hand with all fingers bent."},
    "Mushti": {"desc": "The standard fist. Represents steadiness or combat. ALL fingers and thumb tightly closed into a fist."},
    "Sarpashirsha": {"desc": "Meaning 'serpent's head.' Represents a cobra hood. Formed by fingers held tightly together and slightly cupped."},
    "Hamsapaksha": {"desc": "Meaning 'swan's wing.' Represents a bridge or veil. Formed by 4 fingers straight and together with the thumb tucked into the palm."},
    "Kapittha": {"desc": "Representing Lakshmi, Sarasvati, grasping the cymbals, milking the cows, collyrium, holding the flowers during amorous sport."},
    "Kangula": {"desc": "To represent Lakuca fruit, bell, Cakora bird, betel-nut tree, the bosoms of young maiden, white lily flower, coconut and Caataka bird."},
    "Tamrachuda": {"desc": "Meaning 'red-crested rooster.' A Sattriya folk dance gesture from Assam. Represents a rooster or auspiciousness. Formed by hooking the index finger upward with thumb extended out, while middle, ring, and pinky are folded into the palm."},
}

def wrap_text(image, text, pos, font, font_scale, color, thickness, max_width):
    x, y = pos
    words = text.split(' ')
    line = ""
    for word in words:
        test_line = line + word + " "
        (text_width, text_height), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if text_width > max_width:
            cv2.putText(image, line, (x, y), font, font_scale, color, thickness)
            y += text_height + 5
            line = word + " "
        else:
            line = test_line
    cv2.putText(image, line, (x, y), font, font_scale, color, thickness)

# --- Core Logic ---
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

def fingers_up(lmList, hand_handedness):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]
    if hand_handedness == 'Right':
        fingers.append(1 if lmList[tip_ids[0]][1] < lmList[tip_ids[0] - 1][1] else 0)
    else:
        fingers.append(1 if lmList[tip_ids[0]][1] > lmList[tip_ids[0] - 1][1] else 0)
    for id in range(1, 5):
        fingers.append(1 if lmList[tip_ids[id]][2] < lmList[tip_ids[id] - 2][2] else 0)
    return fingers

def angle_3pt(lmList, a, b, c):
    """Returns angle in degrees at landmark b, between landmarks a-b-c."""
    v1 = (lmList[a][1]-lmList[b][1], lmList[a][2]-lmList[b][2])
    v2 = (lmList[c][1]-lmList[b][1], lmList[c][2]-lmList[b][2])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag = (math.hypot(*v1)+1e-9) * (math.hypot(*v2)+1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot/mag))))

def get_mudra_info(lmList, hand_handedness):
    if not lmList:
        return "Unknown", 0.0

    # --- Measurements ---
    hand_size           = math.hypot(lmList[9][1]-lmList[0][1],   lmList[9][2]-lmList[0][2])
    dist_thumb_index    = math.hypot(lmList[4][1]-lmList[8][1],   lmList[4][2]-lmList[8][2])
    dist_thumb_middle   = math.hypot(lmList[4][1]-lmList[12][1],  lmList[4][2]-lmList[12][2])
    dist_thumb_ring     = math.hypot(lmList[4][1]-lmList[16][1],  lmList[4][2]-lmList[16][2])
    dist_index_pinky    = math.hypot(lmList[8][1]-lmList[20][1],  lmList[8][2]-lmList[20][2])
    dist_pink_ring      = math.hypot(lmList[20][1]-lmList[16][1], lmList[20][2]-lmList[16][2])
    thumb_to_index_base = math.hypot(lmList[4][1]-lmList[5][1],   lmList[4][2]-lmList[5][2])
    dist_hood           = math.hypot(lmList[6][1]-lmList[8][1],   lmList[6][2]-lmList[8][2])
    dist_kagula         = math.hypot(lmList[2][1]-lmList[9][1],   lmList[2][2]-lmList[9][2])
    dist_tripatka       = math.hypot(lmList[14][1]-lmList[16][1], lmList[14][2]-lmList[16][2])

    fingers = fingers_up(lmList, hand_handedness)

    if (fingers[0] == 1 and fingers[1] == 1 and
            fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0):
        # Measure how bent the index finger is at PIP joint (lm 5-6-8)
        index_pip_angle = angle_3pt(lmList, 5, 6, 8)
        # Also confirm middle/ring are genuinely folded (tips below MCP)
        mid_folded  = lmList[12][2] > lmList[9][2]
        ring_folded = lmList[16][2] > lmList[13][2]
        # Tamrachuda: index is bent/hooked (angle < 155°)
        if index_pip_angle < 155 and mid_folded and ring_folded:
            return "Tamrachuda", 1.0
        # index is straight (angle >= 155°) → falls through to Chandrakala below

    # --- Pataka / Hamsapaksha / Sarpashirsha / Ardhachandra ---
    if fingers == [1, 1, 1, 1, 1] or (fingers[1:] == [1, 1, 1, 1]):
        if (thumb_to_index_base / hand_size) < 0.4 and dist_pink_ring > 50:
            return "Hamsapaksha", 0.95
        if dist_hood < 20 and dist_pink_ring < 30:
            return "Sarpashirsha", 0.92
        if fingers[0] == 1:
            thumb_index_dist_x = abs(lmList[4][1] - lmList[5][1])
            if thumb_index_dist_x > 60:
                return "Ardhachandra", 0.9
        return "Pataka", 0.85

    # --- All other mudras (untouched from original) ---
    if dist_thumb_index < 35 and dist_thumb_middle < 35 and dist_thumb_ring < 35:
        return "Mukula", 1.0

    if fingers == [1, 1, 1, 1, 1] and dist_index_pinky > 140:
        return "Alapadma", 1.0

    # Chandrakala: [1,1,0,0,0] with index STRAIGHT (already passed Tamrachuda check above)
    if fingers == [1, 1, 0, 0, 0]:
        return "Chandrakala", 1.0

    if fingers == [1, 1, 1, 1, 0]:
        return "Chatura", 1.0

    # Suchimukha: ONLY index up, thumb DOWN (fingers[0]==0)
    if fingers == [0, 1, 0, 0, 0] and dist_thumb_index > 60:
        return "Suchimukha", 1.0

    if fingers == [1, 0, 0, 0, 0]:
        return "Shikara", 1.0

    # Mushti: ALL fingers including thumb folded (fingers[0]==0, fingers[1]==0)
    if fingers == [0, 0, 0, 0, 0]:
        return "Mushti", 1.0

    if fingers == [0, 1, 1, 1, 0]:
        return "Trishula", 1.0

    if fingers == [1, 1, 1, 0, 1] and dist_tripatka < 20:
        return "Tripataka", 1.0

    if fingers == [1, 0, 1, 1, 1] and dist_thumb_index > 30:
        return "Arala", 1.0

    if fingers == [1, 0, 0, 0, 1]:
        return "Mrigashirsha", 1.0

    if fingers == [1, 1, 1, 0, 1] and dist_kagula < 50:
        return "Kangula", 1.0

    if fingers == [0, 1, 0, 0, 0] and dist_thumb_index < 20:
        return "Kapittha", 1.0

    return "Unknown Mudra", 0.0

# --- Main Runtime ---
try:
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        mudra_name, display_confidence, description = "No Hand Detected", 0.0, ""

        if results.multi_hand_landmarks:
            for handLms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                lmList = [[id, int(lm.x * w), int(lm.y * h)] for id, lm in enumerate(handLms.landmark)]
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                label = handedness.classification[0].label
                mudra_name, raw_conf = get_mudra_info(lmList, label)
                description = mudra_descriptions.get(mudra_name, {}).get('desc', '')
                display_confidence = 0.92 + (raw_conf * 0.08) if mudra_name != "Unknown Mudra" else 0

        # UI Drawing
        cv2.putText(img, mudra_name, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        if display_confidence > 0:
            cv2.putText(img, f"Confidence: {int(display_confidence * 100)}%", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if description and mudra_name != "Unknown Mudra":
            overlay = img.copy()
            cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
            wrap_text(img, description, (10, h - 70),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, w - 20)

        cv2.imshow("Mudra Identifier", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()

    cv2.destroyAllWindows()

