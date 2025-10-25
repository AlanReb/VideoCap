import cv2
import numpy as np
import os

# === Cargar el clasificador Haar ===
haar_path = cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(
    os.path.join(haar_path, 'haarcascade_frontalface_alt.xml')
)

# === Cargar máscara con canal alfa (transparencia) ===
face_mask = cv2.imread('mask.png', cv2.IMREAD_UNCHANGED)  # <-- mantiene canal alfa
if face_mask is None:
    raise IOError("No se pudo cargar 'mask.png'")

if face_cascade.empty():
    raise IOError('No se pudo cargar el clasificador Haar')

h_mask, w_mask = face_mask.shape[:2]

# === Inicializar cámara ===
cap = cv2.VideoCapture(0)
scaling_factor = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor,
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face_rects:
        if h > 0 and w > 0:
            # Ajuste de proporciones y posición de la máscara
            h = int(1.4 * h)
            w = int(1.0 * w)
            y = int(y - 0.1 * h)
            x = int(x)

            # Asegurar límites válidos dentro del frame
            y = max(0, y)
            x = max(0, x)
            y2 = min(y + h, frame.shape[0])
            x2 = min(x + w, frame.shape[1])

            # Redimensionar máscara para tener transparencia
            mask_resized = cv2.resize(face_mask, (x2 - x, y2 - y), interpolation=cv2.INTER_AREA)

            # Separar canales
            if mask_resized.shape[2] == 4:
                b, g, r, a = cv2.split(mask_resized)
                mask_rgb = cv2.merge((b, g, r))
                alpha = a / 255.0  # Normalizar a rango 0–1
            else:
                # Si la máscara no tiene alfa, usar todo opaco
                mask_rgb = mask_resized
                alpha = np.ones(mask_rgb.shape[:2], dtype=float)

            # Región del rostro
            roi = frame[y:y2, x:x2]

            # Mezclar usando el canal alfa
            for c in range(3):
                roi[:, :, c] = (alpha * mask_rgb[:, :, c] +
                                (1 - alpha) * roi[:, :, c])

            frame[y:y2, x:x2] = roi

    # Mostrar resultado
    cv2.imshow('Face Mask Detector', frame)

    if cv2.waitKey(1) == 27:  #Usar ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
