import os
import argparse
import logging

from detectors import vehicle_detector
vehicle_detector.run()
# parser = argparse.ArgumentParser()
# parser.add_argument("model", help="The TensorFlow model to use.")
# args = parser.parse_args()
# ns = args.namespace