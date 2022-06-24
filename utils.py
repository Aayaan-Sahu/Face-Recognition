import cv2
import numpy as np

def create_training_data_with_background(frame, unique_file_identifier):
    thing = frame.copy()
    thing = cv2.resize(thing, (224, 224))
    cv2.imwrite(f'background/background_{unique_file_identifier}.png', thing)

def create_training_data_with_person(frame, face_net, label, unique_file_identifier):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    locs = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3 : 7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype('int')

            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

            face = frame[start_y  : end_y, start_x : end_x]
            face = cv2.resize(face, (224, 224))
            cv2.imwrite(f'{label}/{label}_{unique_file_identifier}.png', face)

            locs.append((start_x, start_y, end_x, end_y))
    
    return locs
