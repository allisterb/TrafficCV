import os
import sys
import platform
import time
import math
from logging import info, error, debug, warn

import cv2
import dlib
import tflite_runtime.interpreter as tflite
from PIL import Image

import kbinput
import detector

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

class Detector(detector.Detector):
    def load_labels(self, path, encoding='utf-8'):
        """Loads labels from file (with or without index numbers).

        Args:
            path: path to label file.
            encoding: label file encoding.
        Returns:
            Dictionary mapping indices to labels.
        """
        with open(path, 'r', encoding=encoding) as f:
            lines = f.readlines()
            if not lines:
                return {}
            if lines[0].split(' ', maxsplit=1)[0].isdigit():
                pairs = [line.split(' ', maxsplit=1) for line in lines]
                return {int(index): label.strip() for index, label in pairs}
            else:
                return {index: line.strip() for index, line in enumerate(lines)}

    def make_interpreter(self, model_file):
        """Create TensorFlow Lite interpreter for Edge TPU."""
        model_file, *device = model_file.split('@')
        return tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=[
                tflite.load_delegate(EDGETPU_SHARED_LIB,
                                    {'device': device[0]} if device else {})
            ])

    def __init__(self, model_dir, video_source, args):
        super().__init__(model_dir, video_source, args)
        self.model_file = os.path.join(model_dir, 'ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite')        
        self.labels_file = os.path.join('labels', 'coco_labels.txt')
        if not os.path.exists(self.model_file):
            error(f'The TF Lite model file {self.model_file} does not exist.')
            sys.exit(1)
        if not os.path.exists(self.labels_file):
            error(f'The TF Lite labels file {self.labels_file} does not exist.')
            sys.exit(1)
        self.labels = self.load_labels(self.labels_file)
        self.interpreter = self.make_interpreter(self.model_file)