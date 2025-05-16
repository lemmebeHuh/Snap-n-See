import os
import cv2
import numpy as np

# Daftar warna HSV dan nama label
colors = {
    "merah": (0, 255, 255),
    "hijau": (60, 255, 255),
    "biru": (120, 255, 255),
    "kuning": (30, 255, 255),
    "ungu": (140, 255, 255),
    "pink": (170, 150, 255),
    "orange": (15, 255, 255),
    "hitam": (0, 0, 10),
    "putih": (0, 0, 255),
    "abu_abu": (0, 0, 127)
}

# Parameter gambar
img_size = 50
jumlah_gambar_per_warna = 30

# Buat folder dataset
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

for warna, hsv in colors.items():
    folder = os.path.join(dataset_path, warna)
    os.makedirs(folder, exist_ok=True)
    
    for i in range(jumlah_gambar_per_warna):
        # Tambah variasi kecil pada HSV
        h = np.clip(hsv[0] + np.random.randint(-5, 5), 0, 179)
        s = np.clip(hsv[1] + np.random.randint(-30, 30), 0, 255)
        v = np.clip(hsv[2] + np.random.randint(-30, 30), 0, 255)
        
        color_img = np.full((img_size, img_size, 3), (h, s, v), dtype=np.uint8)
        bgr_img = cv2.cvtColor(color_img, cv2.COLOR_HSV2BGR)
        
        filename = f"{warna}_{i}.png"
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, bgr_img)

print("✅ Dataset warna berhasil dibuat!")
