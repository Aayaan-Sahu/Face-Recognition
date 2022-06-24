from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import sys
import cv2
import time
import json
import imutils
import argparse
import numpy as np
from imutils.video import VideoStream

print('[INFO] loading labels file...')
labels = json.load(open('labels.json'))


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--person', type=str, required=True,
                        help='person to look for (should be one of the labels in the model you trained')
    args = vars(parser.parse_args())
    return args


args = handle_args()
if args['person'] not in labels.values():
    print(f'[INFO] {args["person"]} is not a valid label')
    print(f'[INFO] valid labels are: {list(labels.values())}')
    sys.exit(1)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell tensorflow to shut up


def detect_face_prediction(frame, face_net, detector, labels):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3: 7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            (start_x, start_y) = (max(0, startX), max(0, startY))
            (end_x, end_y) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[start_y: end_y, start_x: end_x]
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((start_x, start_y, end_x, end_y))

    if len(faces) > 0:
        faces = np.array(faces, dtype='float32')
        preds = detector.predict(faces, batch_size=32)
        preds = list(map(lambda x: labels[str(np.argmax(x))], preds))

    return (locs, preds)


print('[INFO] loading face detector model')
prototxt_path = 'deploy.prototxt'
weights_path = 'res10_300x300_ssd_iter_140000.caffemodel'
face_net = cv2.dnn.readNet(prototxt_path, weights_path)

print('[INFO] loading detector model...')
model_path = 'detector'
model = load_model(model_path)

print('[INFO] starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    (locs, preds) = detect_face_prediction(frame, face_net, model, labels)

    for (box, pred) in zip(locs, preds):
        (start_x, start_y, end_x, end_y) = box
        label = pred
        color = (0, 255, 0) if label == args['person'] else (0, 0, 255)

        cv2.putText(frame, label, (start_x, start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
