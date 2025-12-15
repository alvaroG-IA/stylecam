import os
import cv2
import numpy as np

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STYLES = {
    ord('1'): ("La Muse", os.path.join(BASE_PATH, 'dnn_models', 'la_muse.t7')),
    ord('2'): ("Starry Night", os.path.join(BASE_PATH, 'dnn_models', 'starry_night.t7')),
    ord('3'): ("Mosaic", os.path.join(BASE_PATH, 'dnn_models', 'mosaic.t7')),
    ord('4'): ("The Scream", os.path.join(BASE_PATH, 'dnn_models', 'the_scream.t7')),
    ord('5'): ("Composition", os.path.join(BASE_PATH, 'dnn_models', 'composition_vii.t7')),
    ord('6'): ("Udnie", os.path.join(BASE_PATH, 'dnn_models', 'udnie.t7')),
    ord('7'): ("The Wave", os.path.join(BASE_PATH, 'dnn_models', 'the_wave.t7')),
    ord('8'): ("Candy", os.path.join(BASE_PATH, 'dnn_models', 'candy.t7')),
    ord('9'): ("Feathers", os.path.join(BASE_PATH, 'dnn_models', 'feathers.t7')),
}


def set_initial_model(path):
    net = cv2.dnn.readNet(path)
    backend = cv2.dnn.DNN_BACKEND_OPENCV
    target = cv2.dnn.DNN_TARGET_CPU
    net.setPreferableBackend(backend)
    net.setPreferableTarget(target)

    return net

def select_style(key):
    return STYLES.get(key, None)


def preprocess_frame(frame, input_size=(512, 320), mean=(103.939, 116.779, 123.68)):
    frame_resized = cv2.resize(frame, input_size)
    blob = cv2.dnn.blobFromImage(frame_resized, scalefactor=1.0, size=input_size, mean=mean)
    return frame_resized, blob


def apply_style(net, blob):
    net.setInput(blob)
    stylized = net.forward()
    return stylized


def postprocess_frame(stylized_frame, mean=(103.939, 116.779, 123.68)):
    stylized_frame = stylized_frame.reshape(3, stylized_frame.shape[2], stylized_frame.shape[3])
    stylized_frame[0] += mean[0]
    stylized_frame[1] += mean[1]
    stylized_frame[2] += mean[2]
    stylized_frame = stylized_frame.transpose(1, 2, 0)
    stylized_frame = stylized_frame.clip(0, 255).astype("uint8")
    stylized_frame = np.ascontiguousarray(stylized_frame)
    return stylized_frame


def blend_frames(original, stylized, alpha=0.8):
    return cv2.addWeighted(original, 1-alpha, stylized, alpha, 0)