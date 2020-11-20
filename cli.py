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
parser.add_argument("model", help="The TensorFlow 1 model to use.", required=True)
args = parser.parse_args()
model = args.model
mobilenet_vehicle_detector.run()
