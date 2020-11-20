"""CLI for TrafficCV program"""

import os
import sys
import argparse
import warnings
import cv2 as cv
from pyfiglet import Figlet

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def print_logo():
    """Print program logo"""
    f = Figlet(font='chunky')
    print(f.renderText('TrafficCV'))

parser = argparse.ArgumentParser()
parser.add_argument("--test", help="Test if OpenCV can read from the default camera device.", 
                        action="store_true")
parser.add_argument("--model", help="The TensorFlow 1 model to use.")
parser.add_argument("--convert", help="Convert a TensorFlow model to a TensorFlow Lite model.", action="store_true")
parser.add_argument("--video", help="Path to a video file to use as input.", 
                        default=0)

args = parser.parse_args()
if args.model is None:
    model = os.path.join('models', "ssd_mobilenet_v2_coco_2018_03_29")
    print("Using default model ssd_mobilenet_v2_coco_2018_03_29.")
else:
    model = os.path.join('models', args.model)
video = args.video

print_logo()

if args.test:
    print("Streaming from default camera device. Press q to quit.")
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv.imshow('OpenCV Test', frame)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    sys.exit()
elif (not os.path.exists(model)) or (not os.path.isdir(model)):
    print(f"The path {model} does not exist or is not a directory. Download models from the TF1 detection model zoo: \n\
        https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md. \n\
        and place in the models subdirectory.")
    sys.exit(1)
elif model.startswith(os.path.join('models', 'ssd_mobilenet_v1_coco')):
    print(f"Using SSD with Mobilenet v1 configuration for MSCOCO: {model}.")
    from detectors import ssd_mobilenet
    ssd_mobilenet.run(model, video)
elif model.startswith(os.path.join('models', 'ssd_mobilenet_v2_coco')):
    print(f"Using SSD with Mobilenet v2 configuration for MSCOCO: {model}.")
    from detectors import ssd_mobilenet
    ssd_mobilenet.run(model, video)