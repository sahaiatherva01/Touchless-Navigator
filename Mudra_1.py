import cv2
import mediapipe as mp
import numpy as np
import math

# --- Sattriya Mudra Definitions & Descriptions (Based on the provided image) ---
sattriya_mudras = {
    "Alpadma": "Full-blown lotus: All fingers separated and curved.",
    "Ankusha": "Hook: Index finger bent like a hook, thumb near palm, others extended.",
    "Ardhasuchi": "Half Needle: Index finger extended, others forming a fist.",
    "Ardhachandra": "Half Moon: Thumb is stretched out, other fingers are straight and together.",
    "Ban": "Arrow: Index and middle fingers extended straight, thumb pressed, others bent.",
    "Bhramara": "Bee: Thumb and middle finger touch, index finger is straight, and others are bent.",
    "Chatur": "Four: Four fingers together, thumb held apart.",
    "Dhanu": "Bow: Index and middle fingers bent like a bow, thumb and others extended.",
    "Granika": "Knot: Fingers loosely bent, palm slightly cupped, thumb slightly out.",
    "Hangshamukha": "Swan-face: Tip of index finger joins the tip of the thumb, others extended.",
    "Kartarimukha": "Scissor-face: Index and middle fingers extended, others bent, like scissors.",
    "Khatkhamukha": "Opening: All fingers bent in, but not touching the palm, thumb slightly off.",
    "Kopittha": "Elephant-apple: Fist with thumb covering the other fingers.",
    "Krishnasarmukha": "Krishna's Head: Ring finger bent towards palm, thumb and pinky extended, others bent.",
    "Mukula": "Bud: All fingertips are brought together to form a bud shape.",
    "Mustika": "Fist: Thumb over the other four fingers.",
    "Padmokosha": "Lotus-bud: Fingers loosely bent, palm hollowed.",
    "Pataka": "Flag: All fingers straight and held together.",
    "Sandangsha": "Pliers: Like Padmakosha, but with the palm repeatedly opened and closed.",
    "Sarpashira": "Serpent-head: Fingers bent together like a snake's hood.",
    "Sashaka": "Rabbit: Index and middle fingers extended, others bent, often with a slight curve.",
    "Sighamukha": "Lion-face: Thumb and ring finger touch at their tips, others are extended.",
    "Sikhara": "Peak/Spire: Fist with the thumb extended upwards.",
    "Suchimukha": "Needle-face: Index finger points up, others in a fist.",
    "Tamrachuda": "Cock: Ring finger extended, thumb touching middle finger, others bent.",
    "Tantrimukha": "String-face: Pinky and index finger extended, middle and ring bent, thumb to side.",
    "Tripataka": "Three parts of a flag: Pataka mudra with the ring finger bent.",
    "Trishula": "Trident: Thumb, index, and middle fingers extended, others bent.",
    "Urnanav": "Spider: All fingers curved and spread out, like a spider's legs.",
    "Unknown Mudra": "Gesture not recognized.",
    "No Hand Detected": "Show your hand to the camera."
}

# --- Canonical Feature Vectors for Each Mudra (Based on visual interpretation of the image) ---
# The vector consists of: [finger_is_up(x5), normalized_distances(x4)]
canonical_vectors = {
    "Alpadma":      np.array([1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0]),
    "Ankusha":      np.array([0, 0, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]),
    "Ardhasuchi":   np.array([0, 1, 0, 0, 0, 1.0, 0.5, 0.5, 0.5]),
    "Ardhachandra": np.array([1, 1, 1, 1, 1, 1.0, 0.2, 0.2, 0.2]),
    "Ban":          np.array([0, 1, 1, 0, 0, 0.5, 0.2, 0.5, 0.5]),
    "Bhramara":     np.array([0, 1, 0, 0, 0, 0.5, 0.1, 0.8, 0.8]),
    "Chatur":       np.array([1, 1, 1, 1, 1, 0.8, 0.2, 0.2, 0.2]),
    "Dhanu":        np.array([1, 0, 0, 1, 1, 1.0, 0.8, 0.5, 0.5]),
    "Granika":      np.array([0, 0, 0, 0, 0, 0.6, 0.6, 0.6, 0.6]),
    "Hangshamukha": np.array([1, 1, 1, 1, 1, 0.1, 1.0, 1.0, 1.0]),
    "Kartarimukha": np.array([0, 1, 1, 0, 0, 1.5, 0.5, 1.0, 0.5]),
    "Khatkhamukha": np.array([0, 0, 0, 0, 0, 0.7, 0.7, 0.7, 0.7]),
    "Kopittha":     np.array([0, 0, 0, 0, 0, 0.4, 0.4, 0.4, 0.4]),
    "Krishnasarmukha": np.array([1, 0, 0, 1, 0, 0.8, 0.8, 0.1, 0.8]),
    "Mukula":       np.array([0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1]),
    "Mustika":      np.array([0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5]),
    "Padmokosha":   np.array([0, 0, 0, 0, 0, 0.8, 0.8, 0.8, 0.8]),
    "Pataka":       np.array([1, 1, 1, 1, 1, 0.2, 0.2, 0.2, 0.2]),
    "Sandangsha":   np.array([0, 0, 0, 0, 0, 0.6, 0.6, 0.6, 0.6]),
    "Sarpashira":   np.array([0, 0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3]),
    "Sashaka":      np.array([0, 1, 1, 0, 0, 0.8, 0.4, 0.8, 0.5]),
    "Sighamukha":   np.array([0, 1, 1, 0, 0, 0.8, 0.5, 0.1, 0.8]),
    "Sikhara":      np.array([1, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5]),
    "Suchimukha":   np.array([0, 1, 0, 0, 0, 1.0, 0.5, 0.5, 0.5]),
    "Tamrachuda":   np.array([0, 0, 1, 1, 0, 0.5, 0.1, 0.8, 0.5]),
    "Tantrimukha":  np.array([1, 1, 0, 0, 1, 0.8, 1.0, 0.8, 0.8]),
    "Tripataka":    np.array([1, 1, 1, 0, 1, 0.2, 0.2, 1.0, 0.3]),
    "Trishula":     np.array([1, 1, 1, 0, 0, 0.8, 0.2, 0.8, 0.8]),
    "Urnanav":      np.array([0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0]),
}

def get_feature_vector(lmList):
    """
    Generates a feature vector from hand landmarks.
    The vector includes finger states and normalized distances between fingertips.
    """
    if not lmList:
        return None

    # Landmark IDs for the tips of the fingers
    tip_ids = [4, 8, 12, 16, 20]
    
    # 1. Finger States (Up/Down)
    fingers_up = []
    # Thumb (checks horizontal position relative to its base for right hand, might need adjustment for left)
    if lmList[tip_ids[0]][1] > lmList[tip_ids[0]-1][1]:
        fingers_up.append(1)
    else:
        fingers_up.append(0)

    # Other 4 fingers (checks vertical position)
    for i in range(1, 5):
        fingers_up.append(1 if lmList[tip_ids[i]][2] < lmList[tip_ids[i] - 2][2] else 0)

    # 2. Normalized Distances between Fingertips
    # Use palm width (landmarks 5 to 17) as a reference for normalization
    palm_width = np.linalg.norm(np.array(lmList[5][1:]) - np.array(lmList[17][1:]))
    if palm_width == 0: palm_width = 1

    distances = []
    # Thumb-Index, Index-Mid, Mid-Ring, Ring-Pinky
    for i in range(4):
        dist = np.linalg.norm(np.array(lmList[tip_ids[i]][1:]) - np.array(lmList[tip_ids[i+1]][1:]))
        distances.append(dist / palm_width)

    # Combine into a single feature vector
    return np.array(fingers_up + distances)

def identify_mudra(live_vector, confidence_threshold=0.80):
    """
    Identifies the mudra by comparing the live vector to canonical vectors
    using cosine similarity.
    """
    if live_vector is None:
        return "Unknown Mudra", 0.0

    best_match = "Unknown Mudra"
    max_similarity = -1

    for mudra_name, canonical_vec in canonical_vectors.items():
        # Cosine Similarity Calculation
        dot_product = np.dot(live_vector, canonical_vec)
        norm_live = np.linalg.norm(live_vector)
        norm_canonical = np.linalg.norm(canonical_vec)
        
        if norm_live == 0 or norm_canonical == 0: continue
        
        similarity = dot_product / (norm_live * norm_canonical)

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = mudra_name

    if max_similarity < confidence_threshold:
        return "Unknown Mudra", max_similarity

    return best_match, max_similarity

# --- Main Application Logic ---
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    cap.set(3, 1280)
    cap.set(4, 720)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    try:
        while True:
            success, img = cap.read()
            if not success:
                break
            
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            mudra_name = "No Hand Detected"
            confidence = 0.0

            if results.multi_hand_landmarks:
                handLms = results.multi_hand_landmarks[0]
                lmList = [[id, int(lm.x * w), int(lm.y * h)] for id, lm in enumerate(handLms.landmark)]
                
                live_feature_vector = get_feature_vector(lmList)
                mudra_name, confidence = identify_mudra(live_feature_vector)
                
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
            # --- Display Information ---
            description = sattriya_mudras.get(mudra_name, "")
            cv2.rectangle(img, (10, 10), (900, 110), (0, 0, 0), cv2.FILLED)
            cv2.putText(img, f"Mudra: {mudra_name} ({confidence:.2f})", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            cv2.putText(img, f"Means: {description}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Sattriya Mudra Identifier", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Program finished.")

if __name__ == "__main__":
    main()
