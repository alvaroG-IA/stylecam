import os
import cv2
import time
import numpy as np

# =========================
# 1. Load deep learning model
# =========================
# Ruta del directorio base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, 'dnn_models', 'candy.t7')

net = cv2.dnn.readNet(MODEL_PATH)

backend = cv2.dnn.DNN_BACKEND_OPENCV
target = cv2.dnn.DNN_TARGET_CPU

net.setPreferableBackend(backend)
net.setPreferableTarget(target)

# =========================
# 2. Open camera
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Unable to open video source")

print('[INFO] Press ESC to quit')

# =========================
# 3. Parameters
# =========================
INPUT_W = 512
INPUT_H = 320

alpha = 1

MEAN = (103.939, 116.779, 123.68)

# =========================
# 4. Real-time loop
# =========================
while True:
    loop_start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # ---- Resize (KEY for FPS) ----
    frame = cv2.resize(frame, (INPUT_W, INPUT_H))

    # ---- Pre-processing ----
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0,
        size=(INPUT_W, INPUT_H),
        mean=MEAN,
        swapRB=False,
        crop=False
    )

    net.setInput(blob)

    # ---- Inference ----
    stylized_frame = net.forward()

    # ---- Post-processing ----
    stylized_frame = stylized_frame.reshape(3, stylized_frame.shape[2], stylized_frame.shape[3])
    stylized_frame[0] += MEAN[0]
    stylized_frame[1] += MEAN[1]
    stylized_frame[2] += MEAN[2]

    stylized_frame = stylized_frame.transpose(1, 2, 0)
    stylized_frame = stylized_frame.clip(0, 255).astype("uint8")
    stylized_frame = np.ascontiguousarray(stylized_frame)

    out = cv2.addWeighted(frame, 1-alpha, stylized_frame, alpha, 0)

    # ---- FPS ----
    fps = 1.0 / (time.time() - loop_start)

    cv2.putText(
        out,
        f"FPS: {fps:.2f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    cv2.putText(out, f"Style: La Muse", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(out, f"Intensity: {alpha:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ---- Display ----
    cv2.imshow("Style Transfer - La Muse", out)

    if cv2.waitKey(1) == 27:
        break

# =========================
# 5. Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()
