import cv2
import mediapipe as mp
import pygame
import os

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Inisialisasi pygame mixer
pygame.mixer.init()

# Ambil semua lagu
folder_lagu = "lagu"
playlist = [os.path.join(folder_lagu, f) for f in os.listdir(folder_lagu) if f.endswith(".mp3")]
current_index = 0

def play_music(index):
    pygame.mixer.music.load(playlist[index])
    pygame.mixer.music.play()
    print(f"▶️ Memutar: {os.path.basename(playlist[index])}")

def pause_music():
    pygame.mixer.music.pause()
    print("⏸ Musik dijeda")

def unpause_music():
    pygame.mixer.music.unpause()
    print("▶️ Musik dilanjut")

def next_music():
    global current_index
    current_index = (current_index + 1) % len(playlist)
    play_music(current_index)

def prev_music():
    global current_index
    current_index = (current_index - 1) % len(playlist)
    play_music(current_index)

# Fungsi hitung jari
def count_fingers(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for i in range(1, 5):
        if hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

# Inisialisasi Kamera
cam = cv2.VideoCapture(0)

last_action = ""

while True:
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        handLms = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        total_fingers = count_fingers(handLms)

        if total_fingers == 5 and last_action != "play":
            unpause_music()
            last_action = "play"
        elif total_fingers == 0 and last_action != "pause":
            pause_music()
            last_action = "pause"
        elif total_fingers == 2 and last_action != "next":
            next_music()
            last_action = "next"
        elif total_fingers == 1 and last_action != "prev":
            prev_music()
            last_action = "prev"

        cv2.putText(frame, f"Aksi: {last_action}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    else:
        last_action = ""

    cv2.imshow("Kontrol Musik dengan Gestur", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
