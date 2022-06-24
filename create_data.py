import os
import sys
import time
import shutil
import argparse
import numpy as np

import cv2
import imutils
from imutils.video import VideoStream

import utils


# Program starts
def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--person', type=str, required=True, help='person to create training data for')
    parser.add_argument('-s', '--samples', type=int, required=True, help='how many training samples to create')
    parser.add_argument('-b', '--background', action='store_true', required=False, help='turn on to record background images')
    parser.add_argument('-d', '--delete', action='store_true', required=False, help='turn on to delete existing training data')
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    return vars(parser.parse_args())

args = handle_args()

if args['samples'] % 5 != 0:
    print('Samples must be divisible by 5')
    sys.exit(1)

if args['delete']:
    if os.path.exists(args['person']):
        print(f'[INFO] Deleting existing training data for {args["person"]}...')
        shutil.rmtree(args['person'])
    else:
        print(f'[INFO] {args["person"]} does not exist')
    sys.exit(0)

print('[INFO] loading face detecting model...')
prototxt_path = 'deploy.prototxt'
weights_path =  'res10_300x300_ssd_iter_140000.caffemodel'
face_net = cv2.dnn.readNet(prototxt_path, weights_path)

print('[INFO] staring video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)

if args['background']:
    print('[INFO] recording background images...')
    if os.path.exists('background'):
        shutil.rmtree('background')
    os.mkdir('background')
else:
    print(f'[INFO] using label {args["person"]}')
    if os.path.exists(args['person']):
        print(f'[INFO] {args["person"]} already exists, overwriting it...')
        print('[QUERY] Confirm overwrite (y/n): ')
        # break if user doesn't want to overwrite
        if input() != 'y':
            print('[INFO] exiting...')
            sys.exit(1)
        shutil.rmtree(args['person'])
    os.mkdir(args['person'])
        



unique_file_identifier = 0
counter = 0

if args['background']:
    while True:
        if counter == args['samples']:
            break

        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        utils.create_training_data_with_background(frame, unique_file_identifier)
        unique_file_identifier += 1
        cv2.imshow('Frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        counter += 1
else:
    instructions = ['look at the camera', 'look to the left', 'look to the right', 'look up', 'look down']
    num_instructions = len(instructions)

    while True:
        # check if need to print next instruction
        if counter >= args['samples'] / num_instructions:
            counter = 0
        if counter == 0:
            if len(instructions) == 0:
                break
            print(f'[INFO] {instructions.pop(0)}...')
            input()

        # do all the stuff
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        locs = utils.create_training_data_with_person(frame, face_net, args['person'], unique_file_identifier)
        unique_file_identifier += 1

        for box in locs:
            (start_x, start_y, end_x, end_y) = box
            label = args['person']
            color = (0, 255, 0)

            cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        counter += 1
        time.sleep(0.02)


cv2.destroyAllWindows()
vs.stop()