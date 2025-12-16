import os
import cv2
import time
from style_transfer import set_initial_model, select_style, preprocess_frame, apply_style, postprocess_frame, blend_frames
from utils import save_img, generate_output
# =========================
# 1. Load deep learning model
# =========================1
# Ruta del directorio base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_style_name = 'Starry Night'
current_model_path = os.path.join(BASE_DIR, 'dnn_models', 'starry_night_eccv.t7')

net = set_initial_model(current_model_path)

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

# =========================
# 4. Real-time loop
# =========================
while True:
    loop_start = time.time()

    frame = cap.read()[1]

    frame, blob = preprocess_frame(frame, (INPUT_W, INPUT_H))
    stylized = apply_style(net, blob)
    stylized = postprocess_frame(stylized)
    final = blend_frames(frame, stylized, alpha)

    # ---- FPS ----
    fps = 1.0 / (time.time() - loop_start)

    # ---- Output ---
    out = generate_output(final, fps, current_style_name, alpha)

    # ---- Display ----
    cv2.imshow("Style Transfer - La Muse", out)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('+'):
        alpha = min(1, alpha + 0.05)
    elif key == ord('-'):
        alpha = max(0, alpha - 0.05)
    elif key == ord('s'):
        save_dir = os.path.join(BASE_DIR, 'images')
        save_img(save_dir, final)
    else:
        style_info = select_style(key)
        if style_info is not None:
            new_style_name, new_model_path = style_info
            if new_style_name != current_style_name:

                current_style_name = new_style_name
                current_model_path = new_model_path

                net = cv2.dnn.readNet(current_model_path)
                alpha = 1

                print(f'[INFO] Changed style to {current_style_name}')

# =========================
# 5. Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()
