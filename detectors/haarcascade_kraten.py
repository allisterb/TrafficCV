# Uses code from label_image.py in the TensorFlow Lite object detection examples:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.

# Contains code from https://github.com/kraten/vehicle-speed-check/blob/master/speed_check.py 
# by Kartike Bansal.

# See THIRD-PARTY NOTICES for full attribution and license notice.
# ===============================================================================
import os
import sys
from logging import info, error, debug, warn

import cv2

from bbox import BBox, Object
import detector
  
class Detector(detector.Detector):
    """Haar cascade classifier running on CPU."""
    
    def __init__(self, model_dir, video_source, args):
        super().__init__("Haar cascade classifier on CPU", model_dir, video_source, args)
        self.model_file = os.path.join(model_dir, 'haarcascade_kraten.xml')
        if not os.path.exists(self.model_file):
            error(f'{self.model_file} does not exist.')
            sys.exit(1)
        self.classifier = cv2.CascadeClassifier(self.model_file)
        self.video_width = 1280
        self.video_height = 720
    
    def get_label_for_index(self, _):
        return 'car'

    def detect_objects(self, frame, score_threshold):
        image = cv2.cvtColor(cv2.resize(frame, (self.video_width, self.video_height)), cv2.COLOR_BGR2GRAY)
        cars = self.classifier.detectMultiScale(image, 1.1, 13, 18, (24, 24))
        scale_x, scale_y = (self._width / self.video_width), (self._height / self.video_height) 
        def make(i):
            x, y, w, h = cars[i]
            return Object(
                id=0,
                score=None,
                bbox=BBox(xmin = x,
                    ymin= y,
                    xmax=x+w,
                    ymax=y+h).scale(scale_x, scale_y))
        return [make(i) for i in range(len(cars))]
        
    def print_model_info(self):
        """
        docstring
        """
        pass