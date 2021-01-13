# Uses code from the TensorFlow Lite object detection examples:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# See THIRD-PARTY NOTICES for full attribution and license notice.
# ===============================================================================
import os
import sys
import platform
import time
import math
from logging import info, error, debug, warn

import numpy as np
import cv2
import dlib
import tflite_runtime.interpreter as tflite
from PIL import Image

import kbinput
from bbox import Object, BBox
import detector

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def input_size(interpreter):
  """Returns input image size as (width, height) tuple."""
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  return width, height

def input_tensor(interpreter):
  """Returns input tensor view as numpy array of shape (height, width, 3)."""
  tensor_index = interpreter.get_input_details()[0]['index']
  return interpreter.tensor(tensor_index)()[0]

def set_input(interpreter, size, resize):
  """Copies a resized and properly zero-padded image to the input tensor.

  Args:
    interpreter: Interpreter object.
    size: original image size as (width, height) tuple.
    resize: a function that takes a (width, height) tuple, and returns an RGB
      image resized to those dimensions.
  Returns:
    Actual resize ratio, which should be passed to `get_output` function.
  """
  width, height = input_size(interpreter)
  w, h = size
  scale = min(width / w, height / h)
  w, h = int(w * scale), int(h * scale)
  tensor = input_tensor(interpreter)
  tensor.fill(0)  # padding
  _, _, channel = tensor.shape
  tensor[:h, :w] = np.reshape(resize((w, h)), (h, w, channel))
  return scale, scale

def output_tensor(interpreter, i):
  """Returns output tensor view."""
  tensor = interpreter.tensor(interpreter.get_output_details()[i]['index'])()
  return np.squeeze(tensor)

def get_output(interpreter, score_threshold, image_scale=(1.0, 1.0)):
  """Returns list of detected objects."""
  boxes = output_tensor(interpreter, 0)
  class_ids = output_tensor(interpreter, 1)
  scores = output_tensor(interpreter, 2)
  count = int(output_tensor(interpreter, 3))

  width, height = input_size(interpreter)
  image_scale_x, image_scale_y = image_scale
  sx, sy = width / image_scale_x, height / image_scale_y

  def make(i):
    ymin, xmin, ymax, xmax = boxes[i]
    return Object(
        id=int(class_ids[i]),
        score=float(scores[i]),
        bbox=BBox(xmin=xmin,
                  ymin=ymin,
                  xmax=xmax,
                  ymax=ymax).scale(sx, sy).map(int))

  return [make(i) for i in range(count) if scores[i] >= score_threshold]


class Detector(detector.Detector):
    """SSD MobileNet v1 neural network using 80 COCO categories on TensorFlow Lite runtime on Edge TPU."""
    
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
        super().__init__("SSD MobileNet v1 neural network using 80 COCO categories on TensorFlow Lite runtime on Edge TPU", model_dir, video_source, args)
        self.model_file = os.path.join(model_dir, 'ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite')        
        self.labels_file = os.path.join(model_dir, 'coco_labels.txt')
        if not os.path.exists(self.model_file):
            error(f'The TF Lite model file {self.model_file} does not exist.')
            sys.exit(1)
        if not os.path.exists(self.labels_file):
            error(f'The TF Lite labels file {self.labels_file} does not exist.')
            sys.exit(1)
        self.labels = self.load_labels(self.labels_file)
        self.interpreter = self.make_interpreter(self.model_file)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # check the type of the input tensor
        self.floating_model = self.input_details[0]['dtype'] == np.float32

        self.input_mean = 127.5
        if 'input_mean' in self.args:
            self.input_mean = self.args['input_mean']
            info(f'Using input mean {self.input_mean}.')
        else:
            info('Using default input mean 127.5')

        self.input_stddev = 127.5
        if 'input_stddev' in self.args:
            self.input_stddev = self.args['input_stddev']
            info(f'Using input stddev {self.input_stddev}.')
        else:
            info('Using default input stddev 127.5')
        # NxHxWxC, H:1, W:2
        self.image_height = self.input_details[0]['shape'][1]
        self.image_width = self.input_details[0]['shape'][2]
    
    def get_label_for_index(self, idx):
        self.labels.get(idx, idx)

    def detect_objects(self, frame, score_threshold):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t = set_input(self.interpreter, image.size,
                    lambda size: image.resize(size, Image.ANTIALIAS))
        self.interpreter.invoke()
        return get_output(self.interpreter, score_threshold, t)

    def print_model_info(self):
        """
        docstring
        """
        pass