import cv2
import numpy as np
import urllib.request
import os as _os

_DNN_PROTO = 'deploy.prototxt'
_DNN_MODEL = 'res10_300x300_ssd_iter_140000.caffemodel'


def _download_dnn_models():
    proto_url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
    model_url = 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
    try:
        if not _os.path.exists(_DNN_PROTO):
            print('[FaceDetect] Downloading deploy.prototxt …')
            urllib.request.urlretrieve(proto_url, _DNN_PROTO)
        if not _os.path.exists(_DNN_MODEL):
            print('[FaceDetect] Downloading ResNet-SSD model (~10 MB) …')
            urllib.request.urlretrieve(model_url, _DNN_MODEL)
        return True
    except Exception as e:
        print(f'[FaceDetect] Model download failed: {e}')
        return False


_dnn_net = None
if _download_dnn_models():
    try:
        _dnn_net = cv2.dnn.readNetFromCaffe(_DNN_PROTO, _DNN_MODEL)
        print('[FaceDetect] ✅ ResNet-SSD DNN face detector loaded')
    except Exception as e:
        print(f'[FaceDetect] DNN load failed, falling back to Haar: {e}')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')


def dnn_detect_faces(frame, conf_threshold=0.55):
    if _dnn_net is None:
        return haar_detect_faces(frame)
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                  (300, 300), (104.0, 177.0, 123.0))
    _dnn_net.setInput(blob)
    detections = _dnn_net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < conf_threshold:
            continue
        x1 = max(0, int(detections[0, 0, i, 3] * w))
        y1 = max(0, int(detections[0, 0, i, 4] * h))
        x2 = min(w, int(detections[0, 0, i, 5] * w))
        y2 = min(h, int(detections[0, 0, i, 6] * h))
        bw, bh = x2 - x1, y2 - y1
        if bw > 20 and bh > 20:
            boxes.append((x1, y1, bw, bh))
    return boxes


def haar_detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4,
                                           minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE)
    return list(faces) if len(faces) > 0 else []


def detect_faces(frame, conf_threshold=0.55):
    """Primary entry point — uses DNN if available, falls back to Haar."""
    return dnn_detect_faces(frame, conf_threshold) if _dnn_net else haar_detect_faces(frame)
