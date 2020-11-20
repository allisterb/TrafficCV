import argparse
import sys
import cv2 as cv

from pyfiglet import Figlet

#from detectors import mobilenet_vehicle_detector

def print_logo():
    """Print program logo"""
    f = Figlet(font='chunky')
    print(f.renderText('TrafficCV'))

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="The TensorFlow 1 model to use.", default="ssd_mobilenet_v1_coco_2018_01_28")
parser.add_argument("--test", help="Test if OpenCV can read from the default camera device.", action="store_true")
args = parser.parse_args()
model = args.model

if args.test:
    print_logo()
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
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    sys.exit()
#mobilenet_vehicle_detector.run()
