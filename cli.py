"""TrafficCV CLI"""

import os
import sys
import argparse
import warnings

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
parser.add_argument("--model", help="The computer vision model to use.")
parser.add_argument("--video", help="Path to a video file to use as input.", 
                        default=0)
args = parser.parse_args()

print_logo()

if args.test:
    import cv2 as cv
    print("Streaming from default camera device. Press q to quit.")
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open default camera device.")
        sys.exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Did not read frame. Stopping.")
            break
        cv.imshow('OpenCV Test', frame)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    sys.exit(0)

if args.model is None:
    model = os.path.join('models', "ssd_mobilenet_v2_coco_2018_03_29")
    print("Using default model ssd_mobilenet_v2_coco_2018_03_29.")
else:
    model = os.path.join('models', args.model)
if (not os.path.exists(model)) or (not os.path.isdir(model)):
    print(f"The path {model} does not exist or is not a directory. Download models from the TF1 detection model zoo: \n\
        https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md. \n\
        and place in the models subdirectory.")
    sys.exit(1)
video = args.video

if model.startswith(os.path.join('models', 'ssd_mobilenet_v1_coco')):
    print(f"Using TF SSD with Mobilenet v1 configuration for MSCOCO.")
    from detectors import ssd_mobilenet
    ssd_mobilenet.run(model, video)
elif model.startswith(os.path.join('models', 'ssd_mobilenet_v2_coco')):
    print(f"Using TF SSD with Mobilenet v2 configuration for MSCOCO.")
    from detectors import ssd_mobilenet
    ssd_mobilenet.run(model, video)
elif model.startswith(os.path.join('models', 'ssd_mobilenet_caffe')):
    print(f"Using Caffe SSD with Mobilenet v1 configuration for MSCOCO.")
    from detectors import ssd_mobilenet_caffe
    ssd_mobilenet_caffe.run(video)
elif model.startswith(os.path.join('models', 'yolov3')):
    print(f"Using YOLOv3 for MSCOCO.")
    from detectors import yolov3
    yolov3.run(model, video)