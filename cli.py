"""CLI interface for TrafficCV"""

import os
import sys
import threading
import argparse
import warnings
import logging
from logging import info, error, debug
from pyfiglet import Figlet

import kbinput

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def print_logo():
    """Print program logo."""
    fig = Figlet(font='chunky')
    print(fig.renderText('TrafficCV') + 'v0.1\n')
    
parser = argparse.ArgumentParser()
parser.add_argument("--debug", help="Enable debug-level logging.", action="store_true")
parser.add_argument("--test", help="Test if OpenCV can read from the default camera device.", action="store_true")
parser.add_argument("--video", help="Path to a video file or source to use as input.", default=0)
parser.add_argument("--model", help="The computer vision model to use. A directory with this name must exist in the models sub-directory.")
parser.add_argument("--args", help="Arguments to pass to the specified model and detector comma-delimited as key=value e.g --args \'ppm=4,fps=1\'")
parser.add_argument("--nowindow", help="Don't display the video in a window.",  action="store_true", default=False)
parser.add_argument("--info", help="Print out info on the model only.",  action="store_true", default=False)
args = parser.parse_args()

if args.debug:
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%I:%M:%S %p', level=logging.DEBUG)
    info("Debug mode enabled.")
else:
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%I:%M:%S %p', level=logging.INFO)

print_logo()
threading.Thread(target=kbinput.kb_capture_thread, args=(), name='kb_capture_thread', daemon=True).start()

if args.nowindow:
    info('Video window disabled. Press ENTER key to stop.')
else:
    info('Video window enabled. Press ENTER key or press any key in the video window to stop.')

if args.test:
    import cv2 as cv
    info('Streaming from default camera device.')
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        error("Could not open default camera device.")
        sys.exit()
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv.CAP_PROP_FPS, 60)
    height, width, fps = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FPS)) 
    info(f'Camera resolution: {width}x{height} {fps}fps.')
    while not kbinput.KBINPUT:
        ret, frame = cap.read()
        if not ret:
            print("Did not read video frame. Stopping.")
            break
        cv.imshow(f'OpenCV Test: {width}x{height} {fps}fps.', frame)
        if cv.waitKey(1) != -1:
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
detector_args['info'] = args.info
detector_args['nowindow'] = args.nowindow

if args.model is None:
    model = os.path.join('models', "ssd_mobilenet_v2_coco_2018_03_29")
    print("Using default model ssd_mobilenet_v2_coco_2018_03_29.")
else:
    model_dir = os.path.join('models', args.model)
if (not os.path.exists(model_dir)) or (not os.path.isdir(model_dir)):
    print(f"The path \'{model_dir}\' does not exist or is not a directory. You can download models from the TF1 detection model zoo:"
        "https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md and place in the"  
        "models subdirectory of TrafficCV.")
    sys.exit(1)

if model_dir.startswith(os.path.join('models', 'haarcascade')):
    info("Using Haar cascade classifier on CPU.")
    from detectors import haarcascade_kraten
    haarcascade_kraten.Detector(model_dir, video, detector_args).run()

elif model_dir == (os.path.join('models', 'ssd_mobilenet_v1_coco_tflite')):
    from detectors import ssd_mobilenet_tflite
    ssd_mobilenet_tflite.Detector(model_dir, video, detector_args).run()

elif model_dir == (os.path.join('models', 'ssd_mobilenet_v1_coco_edgetpu')):
    from detectors import ssd_mobilenet_edgetpu
    ssd_mobilenet_edgetpu.Detector(model_dir, video, detector_args).run()

elif model_dir.startswith(os.path.join('models', 'ssd_mobilenet_v1_coco')) and detector_args['edgetpu']:
    info("Using TensorFlow Lite on Edge TPU SSD MobileNetv1 for COCO.")
    # from detectors import ssd_mobilenet_edgetpu
    # ssd_mobilenet_edgetpu.run(model_dir, video, detector_args)
    from detectors import ssd_mobilenet_d
    ssd_mobilenet_d.Detector(model_dir, video, detector_args).run()

elif model_dir.startswith(os.path.join('models', 'ssd_mobilenet_v1_coco')):
    info("Using TensorFlow SSD MobileNetv1 for COCO.")
    from detectors import ssd_mobilenet
    ssd_mobilenet.run(model_dir, video, detector_args)

elif model_dir.startswith(os.path.join('models', 'ssd_mobilenet_v2_coco')):
    info("Using TensorFlow SSD MobileNetv2 for COCO.")
    from detectors import ssd_mobilenet
    ssd_mobilenet.run(model_dir, video, detector_args)
elif model_dir.startswith(os.path.join('models', 'yolov3')):
    info("Using TensorFlow YOLOv3 for COCO.")
    from detectors import yolov3
    yolov3.run(model_dir, video)
elif model_dir.startswith(os.path.join('models', 'ssd_mobilenet_caffe')):
    info("Using Caffe SSD MobileNetv1 for COCO.")
    from detectors import ssd_mobilenet_caffe
    ssd_mobilenet_caffe.run(video)
