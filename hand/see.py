# import cv2
# import numpy as np
# import json

# # Inisialisasi kamera

# cam = cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# def get_dominant_color(hsv_area):
#     """
#     Fungsi untuk menentukan warna dominan dari area HSV.
#     Mengambil nilai rata-rata hue, saturation, dan value dari area.
#     """
#     h_mean = int(np.mean(hsv_area[:, :, 0]))
#     s_mean = int(np.mean(hsv_area[:, :, 1]))
#     v_mean = int(np.mean(hsv_area[:, :, 2]))

#     # Klasifikasi warna berdasarkan rentang HSV
#     if v_mean < 50:
#             return "HITAM"
#     elif s_mean < 50 and v_mean > 200:
#             return "PUTIH"
#     elif s_mean < 50:
#             return "ABU-ABU"
#     else:
#             if (h_mean >= 0 and h_mean <= 10) or (h_mean >= 160 and h_mean <= 180):
#                 return "MERAH"
#             elif h_mean > 10 and h_mean <= 25:
#                 return "ORANGE"
#             elif h_mean > 25 and h_mean <= 35:
#                 return "KUNING"
#             elif h_mean > 35 and h_mean <= 85:
#                 return "HIJAU"
#             elif h_mean > 85 and h_mean <= 125:
#                 return "BIRU"
#             elif h_mean > 125 and h_mean <= 145:
#                 return "UNGU"
#             elif h_mean > 145 and h_mean < 160:
#                 return "PINK"

# while True:
#     _, frame = cam.read()
#     frame = cv2.flip(frame, 1)
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     height, width, _ = frame.shape

#     # Area deteksi warna (kotak 50x50 di tengah layar)
#     box_size = 100
#     cx, cy = int(0.75 * width), height // 2
#     # cx, cy = width // 2, height // 2
#     x1, y1 = cx - box_size // 2, cy - box_size // 2
#     x2, y2 = cx + box_size // 2, cy + box_size // 2

#     # Ambil area dari gambar HSV
#     hsv_area = hsv_frame[y1:y2, x1:x2]
#     detected_color = get_dominant_color(hsv_area)

#     # Gambar kotak area deteksi
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
#     cv2.putText(frame, detected_color, (x1, y1 - 10), 0, 1.5, (255, 255, 255), 2)

#     cv2.imshow("Deteksi Warna Area", frame)

#     key = cv2.waitKey(1)
#     if key == ord('s'):
#         # Simpan data warna ke file saat tekan 's'
#         result = {
#             "warna_terdeteksi": detected_color,
#             "hsv_rata_rata": {
#                 "h": int(np.mean(hsv_area[:, :, 0])),
#                 "s": int(np.mean(hsv_area[:, :, 1])),
#                 "v": int(np.mean(hsv_area[:, :, 2]))
#             }
#         }
#         with open("hasil_deteksi_warna.json", "w") as f:
#             json.dump(result, f, indent=4)
#         print("Warna disimpan ke hasil_deteksi_warna.json")

#     elif key == 27:  # ESC
#         break

# cam.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import mediapipe as mp

# # Inisialisasi kamera
# cam = cv2.VideoCapture(2)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# # Inisialisasi MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False,
#                        max_num_hands=2,
#                        min_detection_confidence=0.7,
#                        min_tracking_confidence=0.7)
# mp_draw = mp.solutions.drawing_utils

# # Fungsi menghitung jari yang terangkat
# def count_fingers(hand_landmarks, hand_label):
#     finger_tips = [8, 12, 16, 20]
#     finger_base = [6, 10, 14, 18]
#     count = 0

#     # Ibu jari
#     if hand_label == "Right":
#         if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
#             count += 1
#     else:
#         if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
#             count += 1

#     # Jari lainnya
#     for tip, base in zip(finger_tips, finger_base):
#         if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
#             count += 1
#     return count

# # Variabel operasi
# operation = "+"
# result = None

# while True:
#     ret, frame = cam.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(rgb_frame)
#     height, width, _ = frame.shape

#     left_fingers = 0
#     right_fingers = 0

#     if results.multi_hand_landmarks and results.multi_handedness:
#         for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
#             hand_label = results.multi_handedness[i].classification[0].label  # 'Right' or 'Left'
#             fingers = count_fingers(hand_landmarks, hand_label)

#             if hand_label == "Left":
#                 left_fingers = fingers
#             else:
#                 right_fingers = fingers

#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#         # Tampilkan jumlah jari
#         cv2.putText(frame, f"Tangan Kiri: {left_fingers}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#         cv2.putText(frame, f"Tangan Kanan: {right_fingers}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#         # Hitung hasil operasi
#         if operation == '+':
#             result = left_fingers + right_fingers
#         elif operation == '-':
#             result = left_fingers - right_fingers
#         elif operation == '*':
#             result = left_fingers * right_fingers
#         elif operation == '/' and right_fingers != 0:
#             result = round(left_fingers / right_fingers, 2)
#         elif operation == '/' and right_fingers == 0:
#             result = "ERR"

#         # Tampilkan operasi
#         cv2.putText(frame, f"{left_fingers} {operation} {right_fingers} = {result}", (10, 100),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

#     # Tampilkan frame
#     cv2.imshow("Deteksi Jari - Kalkulator Tangan", frame)

#     # Keyboard control
#     key = cv2.waitKey(1)
#     if key == ord('+'):
#         operation = '+'
#     elif key == ord('-'):
#         operation = '-'
#     elif key == ord('*'):
#         operation = '*'
#     elif key == ord('/'):
#         operation = '/'
#     elif key == 27:
#         break

# cam.release()
# cv2.destroyAllWindows()
#####################################

import cv2
import numpy as np
import mediapipe as mp

# Inisialisasi Kamera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Fungsi menghitung jarak antar 2 titik
def euclidean_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# Variabel untuk zoom dan tracking
zoom_factor = 1.0
target_zoom = 1.0
max_zoom = 9.5
min_zoom = 1.0

center_x, center_y = 320, 240
target_center_x, target_center_y = center_x, center_y

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Ambil posisi ibu jari (4) dan telunjuk (8)
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]

        x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
        x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

        # Hitung jarak untuk kontrol zoom
        distance = euclidean_distance((x1, y1), (x2, y2))
        target_zoom = np.clip(1 + (distance / 100), min_zoom, max_zoom)

        # Hitung posisi tengah tangan untuk tracking
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        target_center_x = cx
        target_center_y = cy

        # Gambar lingkaran di jari
        cv2.circle(frame, (x1, y1), 8, (255, 0, 0), -1)
        cv2.circle(frame, (x2, y2), 8, (0, 255, 0), -1)

    # Smooth zoom & tracking
    zoom_factor = 0.9 * zoom_factor + 0.1 * target_zoom
    center_x = int(0.9 * center_x + 0.1 * target_center_x)
    center_y = int(0.9 * center_y + 0.1 * target_center_y)

    # Crop area zoom
    zoom_w = int(w / zoom_factor)
    zoom_h = int(h / zoom_factor)
    x1 = max(center_x - zoom_w // 2, 0)
    y1 = max(center_y - zoom_h // 2, 0)
    x2 = min(center_x + zoom_w // 2, w)
    y2 = min(center_y + zoom_h // 2, h)

    # Ambil area, resize jadi full screen
    cropped = frame[y1:y2, x1:x2]
    zoomed_frame = cv2.resize(cropped, (w, h))

    cv2.imshow("Zoom & Tracking Tangan", zoomed_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
