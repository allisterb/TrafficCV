import os
import argparse
import logging

from pyfiglet import Figlet

from detectors import mobilenet_vehicle_detector

def print_logo():
    """Print program logo"""
    f = Figlet(font='chunky')
    print(f.renderText('TrafficCV'))

parser = argparse.ArgumentParser()
#parser.add_argument("display", help="The X Server display to use.")
#parser.add_argument("model", help="The TensorFlow model to use.")
#args = parser.parse_args()
#model = args.model
mobilenet_vehicle_detector.run()
