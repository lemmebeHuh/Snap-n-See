import cv2
import numpy as np
import json

# Inisialisasi kamera

cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def get_dominant_color(hsv_area):
    """
    Fungsi untuk menentukan warna dominan dari area HSV.
    Mengambil nilai rata-rata hue, saturation, dan value dari area.
    """
    h_mean = int(np.mean(hsv_area[:, :, 0]))
    s_mean = int(np.mean(hsv_area[:, :, 1]))
    v_mean = int(np.mean(hsv_area[:, :, 2]))

    # Klasifikasi warna berdasarkan rentang HSV
    if v_mean < 50:
        return "HITAM"
    elif s_mean < 50:
        return "ABU-ABU"
    elif h_mean == 0 or s_mean == 0:
        return "PUTIH"
    elif h_mean < 5:
        return "MERAH"
    elif h_mean < 20:
        return "ORANGE"
    elif h_mean < 30:
        return "KUNING"
    elif h_mean < 70:
        return "HIJAU"
    elif h_mean < 125:
        return "BIRU"
    elif h_mean < 145:
        return "UNGU"
    elif h_mean < 170:
        return "PINK"
    else:
        return "MERAH"

while True:
    _, frame = cam.read()
    # frame = cv2.flip(frame, 1)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width, _ = frame.shape

    # Area deteksi warna (kotak 50x50 di tengah layar)
    box_size = 25
    # cx, cy = int(0.75 * width), height // 2
    cx, cy = width // 2, height // 2
    x1, y1 = cx - box_size // 2, cy - box_size // 2
    x2, y2 = cx + box_size // 2, cy + box_size // 2

    # Ambil area dari gambar HSV
    hsv_area = hsv_frame[y1:y2, x1:x2]
    detected_color = get_dominant_color(hsv_area)

    # Gambar kotak area deteksi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(frame, detected_color, (x1, y1 - 10), 0, 1.5, (255, 255, 255), 2)

    cv2.imshow("Deteksi Warna Area", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        # Simpan data warna ke file saat tekan 's'
        result = {
            "warna_terdeteksi": detected_color,
            "hsv_rata_rata": {
                "h": int(np.mean(hsv_area[:, :, 0])),
                "s": int(np.mean(hsv_area[:, :, 1])),
                "v": int(np.mean(hsv_area[:, :, 2]))
            }
        }
        with open("hasil_deteksi_warna.json", "w") as f:
            json.dump(result, f, indent=4)
        print("Warna disimpan ke hasil_deteksi_warna.json")

    elif key == 27:  # ESC
        break

cam.release()
cv2.destroyAllWindows()
