import cv2
import os


def save_img(path, img, name='save.jpg'):
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(os.path.join(path, name), img)
    print(f'[INFO] Saved image in {path}')


def generate_output(img, fps, style_name, alpha):
    output = img.copy()

    cv2.putText(
        output,
        f"FPS: {fps:.2f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    cv2.putText(output, f"Style: {style_name}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(output, f"Intensity: {alpha:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return output
