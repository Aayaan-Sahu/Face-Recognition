import os
import sys
import cv2
import json
import imutils
import argparse
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # tell tensorflow to shut up

import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True, help='path to image')
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    return vars(parser.parse_args())

args = handle_args()

if args['path']:
    if os.path.exists(args['path']):
        print(f'[INFO] loading image...')
        image_path = args['path']
    else:
        print(f'[INFO] {args["path"]} does not exist')
        sys.exit(1)

print('[INFO] loading face detector model')
prototxt_path = 'deploy.prototxt'
weights_path =  'res10_300x300_ssd_iter_140000.caffemodel'
face_net = cv2.dnn.readNet(prototxt_path, weights_path)

print('[INFO] loading detector model...')
model_path = 'detector'
model = load_model(model_path)

print('[INFO] performing image manipulations')
image = cv2.imread(image_path)
image = imutils.resize(image, width=400)
(h, w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

print('[INFO] computing face detections...')
face_net.setInput(blob)
detections = face_net.forward()

print('[INFO] loading labels file...')
labels = json.load(open('labels.json'))

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3 : 7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = box.astype('int')
        (start_x, start_y) = (max(0, start_x), max(0, start_y))
        (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

        face = image[start_y : end_y, start_x : end_x]
        face = cv2.resize(face, (224, 224))
        cv2.imshow('Model input', face)
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        print(model.predict(face))
        prediction = labels[str(np.argmax(model.predict(face)))]
        color = (0, 255, 0)

        cv2_label = prediction
        cv2.putText(image, cv2_label, (start_x, start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, 2)


cv2.imshow('Output', image)
cv2.waitKey(0)