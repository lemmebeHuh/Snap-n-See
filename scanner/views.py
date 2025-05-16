from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import cv2
import numpy as np

# Create your views here.

def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        file_url = fs.url(filename)

        image_path = fs.path(filename)
        img = cv2.imread(image_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h, w, _ = img.shape
        cx, cy = int(w / 2), int(h / 2)
        box_size = 25
        x1, y1 = cx - box_size // 2, cy - box_size // 2
        x2, y2 = cx + box_size // 2, cy + box_size // 2

        hsv_area = hsv[y1:y2, x1:x2]
        h_mean = int(np.mean(hsv_area[:, :, 0]))
        s_mean = int(np.mean(hsv_area[:, :, 1]))
        v_mean = int(np.mean(hsv_area[:, :, 2]))

        # Logika warna lebih akurat
        color = "Tak Terdeteksi"

        if v_mean < 50:
            color = "HITAM"
        elif s_mean < 50 and v_mean > 200:
            color = "PUTIH"
        elif s_mean < 50:
            color = "ABU-ABU"
        else:
            if (h_mean >= 0 and h_mean <= 10) or (h_mean >= 160 and h_mean <= 180):
                color = "MERAH"
            elif h_mean > 10 and h_mean <= 25:
                color = "ORANGE"
            elif h_mean > 25 and h_mean <= 35:
                color = "KUNING"
            elif h_mean > 35 and h_mean <= 85:
                color = "HIJAU"
            elif h_mean > 85 and h_mean <= 125:
                color = "BIRU"
            elif h_mean > 125 and h_mean <= 145:
                color = "UNGU"
            elif h_mean > 145 and h_mean < 160:
                color = "PINK"

        return render(request, 'scanner/result.html', {
            'color': color,
            'hsv': (h_mean, s_mean, v_mean),
            'image_url': file_url
        })

    return render(request, 'scanner/upload.html')
