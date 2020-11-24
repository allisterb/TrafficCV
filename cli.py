"""TrafficCV CLI"""

import os
import sys
import argparse
import warnings
import logging
from pyfiglet import Figlet

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def print_logo():
    """Print program logo."""
    fig = Figlet(font='chunky')
    print(fig.renderText('TrafficCV') + 'v0.1\n')
    
parser = argparse.ArgumentParser()
parser.add_argument("--debug", help="Enable debug-level logging.", 
                        action="store_true")
parser.add_argument("--test", help="Test if OpenCV can read from the default camera device.", 
                        action="store_true")
parser.add_argument("--model", help="The computer vision model to use.")
parser.add_argument("--video", help="Path to a video file to use as input.", 
                        default=0)
parser.add_argument("--args", help="Arguments to pass to the specified model and detector comma-delimited as key=value e.g --args \'ppm=4,fps=1\'")
parser.add_argument("--tflite", help="Execute a TFLite version of the model.", action="store_true", default=False)
parser.add_argument("--edgetpu", help="Execute a TFLite version of the model on an Edge TPU device like the Coral USB accelerator.", action="store_true", default=False)
args = parser.parse_args()

if args.debug:
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
    logging.info("Debug mode enabled.")
else:
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
info = logging.info
error = logging.error
warn = logging.warn
debug = logging.debug
print_logo()

if args.test:
    import cv2 as cv
    info('Streaming from default camera device. Press q to quit.')
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        error("Could not open default camera device.")
        sys.exit()
    height, width, fps = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FPS)) 
    info(f'Camera resolution: {width}x{height} {fps}fps.')
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

video = args.video

detector_args = {}
if args.args is not None:
    for a in args.args.split(','):
        kv = a.split('=')
        if len(kv) != 2:
            error(f'The detector argument {kv} is malformed.')
            sys.exit(1)
        k, v = kv[0], kv[1]
        detector_args[k] = v
    debug(f'Detector arguments are {detector_args}.')
if args.tflite and args.edgetpu:
    error ("Use only one of either the tflite or edgetpu parameters.")
    sys.exit(1)
detector_args['edgetpu'] = args.edgetpu
detector_args['tflite'] = args.tflite

if args.model is None:
    model = os.path.join('models', "ssd_mobilenet_v2_coco_2018_03_29")
    print("Using default model ssd_mobilenet_v2_coco_2018_03_29.")
else:
    model = os.path.join('models', args.model)
if (not os.path.exists(model)) or (not os.path.isdir(model)):
    print(f"The path {model} does not exist or is not a directory. You can download models from the TF1 detection model zoo: \n\
        https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md and place in the models subdirectory of TrafficCV.")
    sys.exit(1)

if model.startswith(os.path.join('models', 'ssd_mobilenet_v1_coco')) and not detector_args['tflite'] and not detector_args['edgetpu']:
    print("Using TensorFlow SSD MobileNetv1 for COCO. Press q in video window to quit.")
    from detectors import ssd_mobilenet
    ssd_mobilenet.run(model, video, detector_args)
elif model.startswith(os.path.join('models', 'ssd_mobilenet_v1_coco')) and detector_args['tflite']:
    print("Using TensorFlow Lite SSD MobileNetv1 for COCO. Press q in video window to quit.")
    from detectors import ssd_mobilenet_tflite
    ssd_mobilenet_tflite.run(model, video, detector_args)
elif model.startswith(os.path.join('models', 'ssd_mobilenet_v1_coco')) and detector_args['edgetpu']:
    print("Using TensorFlow Lite on Edge TPU SSD MobileNetv1 for COCO. Press q in video window to quit.")
    from detectors import ssd_mobilenet_edgetpu
    ssd_mobilenet_edgetpu.run(model, video, detector_args)
elif model.startswith(os.path.join('models', 'ssd_mobilenet_v2_coco')):
    print("Using TensorFlow SSD MobileNetv2 for COCO. Press q in video window to quit.")
    from detectors import ssd_mobilenet
    ssd_mobilenet.run(model, video, detector_args)
elif model.startswith(os.path.join('models', 'yolov3')):
    print("Using TensorFlow YOLOv3 for COCO. Press q in video window to quit.")
    from detectors import yolov3
    yolov3.run(model, video)
elif model.startswith(os.path.join('models', 'ssd_mobilenet_caffe')):
    print("Using Caffe SSD MobileNetv1 for COCO. Press q in video window to quit.")
    from detectors import ssd_mobilenet_caffe
    ssd_mobilenet_caffe.run(video)
elif model.startswith(os.path.join('models', 'haarcascade')):
    print("Using Haar cascade classifier. Press q in video window to quit.")
    from detectors import haarcascade_kraten
    haarcascade_kraten.run(model, video, detector_args)