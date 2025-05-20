import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Fungsi hitung jari
def count_fingers(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Jempol (horizontal)
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Empat jari lainnya (vertikal)
    for tip in tips_ids[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

# Buka Kamera
cap = cv2.VideoCapture(0)

# Muat gambar jika ingin visual khusus
smiley = cv2.imread('smiley.png')        # Ukuran: 640x480
hand_icon = cv2.imread('hand.png')       # Ukuran: 640x480

# Resize agar sama dengan kamera
def resize_img(img):
    return cv2.resize(img, (640, 480)) if img is not None else None

smiley = resize_img(smiley)
hand_icon = resize_img(hand_icon)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    finger_count = -1

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_count = count_fingers(hand_landmarks)

    # Ganti visual sesuai jumlah jari
    if finger_count == 0:
        vis = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(vis, "No Fingers", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)

    elif finger_count == 1:
        vis = np.zeros((480, 640, 3), np.uint8)
        vis[:] = (255, 0, 0)  # Biru
        cv2.putText(vis, "One Finger", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)

    elif finger_count == 2:
        vis = smiley.copy() if smiley is not None else frame.copy()
        cv2.putText(vis, "Smile!", (220, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    elif finger_count == 3:
        vis = np.zeros((480, 640, 3), np.uint8)
        vis[:] = (0, 255, 0)  # Hijau
        cv2.putText(vis, "Three Fingers", (130, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)

    elif finger_count == 4:
        vis = np.zeros((480, 640, 3), np.uint8)
        vis[:] = (0, 255, 255)  # Kuning
        cv2.putText(vis, "Four Fingers", (130, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)

    elif finger_count == 5:
        vis = hand_icon.copy() if hand_icon is not None else frame.copy()
        cv2.putText(vis, "High Five!", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    else:
        vis = frame

    cv2.imshow("Gesture Visual", vis)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
