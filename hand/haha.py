import cv2
import mediapipe as mp
import pygame
import os
from PIL import Image, ImageSequence
import threading
import time

# --- Fungsi untuk Menampilkan GIF di OpenCV ---
# def show_gif_opencv(gif_path, duration=3):
#     gif = Image.open(gif_path)
#     start_time = time.time()
#     for frame in ImageSequence.Iterator(gif):
#         if time.time() - start_time > duration:
#             break
#         rgb_frame = frame.convert('RGB')
#         open_cv_image = cv2.cvtColor(np.array(rgb_frame), cv2.COLOR_RGB2BGR)
#         cv2.imshow("GIF Animasi", open_cv_image)
#         if cv2.waitKey(100) & 0xFF == 27:  # ESC untuk keluar
#             break
#     cv2.destroyWindow("GIF Animasi")

# def tampilkan_gif_async(nomor):
#     gif_path = f"gesture_images/{nomor}.gif"
#     if os.path.exists(gif_path):
#         threading.Thread(target=show_gif_opencv, args=(gif_path,), daemon=True).start()

# --- Setup Lagu dan Gesture ---
pygame.mixer.init()
folder_lagu = "lagu"
playlist = [os.path.join(folder_lagu, f) for f in os.listdir(folder_lagu) if f.endswith(".mp3")]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def count_fingers(hand_landmarks, hand_label):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []
    if hand_label == "Right":
        fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
    else:
        fingers.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)
    for i in range(1, 5):
        fingers.append(1 if hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y else 0)
    return sum(fingers)

cap = cv2.VideoCapture(0)
last_gesture = -1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_count = count_fingers(hand_landmarks, label)
            cv2.putText(frame, f"Jari: {finger_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            if finger_count != last_gesture:
                pygame.mixer.music.stop()
                if 1 <= finger_count <= len(playlist):
                    pygame.mixer.music.load(playlist[finger_count - 1])
                    pygame.mixer.music.play()
                    print(f" Memutar lagu ke-{finger_count}")
                    # tampilkan_gif_async(finger_count)
                    last_gesture = finger_count

    cv2.imshow("Gesture Music Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
