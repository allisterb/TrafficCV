import sys
import abc
import math
import time
from logging import info, error, warn, debug

import numpy as np
import cv2
import dlib
import kbinput

class Detector(abc.ABC):
    """A video object detector using a neural network or other deep learning model."""

    @abc.abstractmethod
    def get_label_for_index(self, i):
        """Get the string label for an integer index."""

    @abc.abstractmethod
    def detect_objects(self, video_frame):
        """Detect objects in a video frame."""

    @abc.abstractmethod
    def print_model_info(self):
        """Print out information on model."""

    def __init__(self, name, model_dir, video_source, args):
        self.name = name
        self.model_dir = model_dir
        self.video_source = video_source
        self.video = cv2.VideoCapture(self.video_source)
        self.args = args

    def estimate_speed(self, ppm, fps, location1, location2):
        """Estimate the speed of a vehicle assuming pixel-per-metre and fps constants."""
        d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
        d_meters = d_pixels / ppm
        speed = d_meters * fps * 3.6
        return speed

    def run(self):
        """Run the classifier and detector."""
        
        if self.args['info']:
            self.print_model_info()
            sys.exit(0)
        
        nowindow = self.args['nowindow']
        ppm = 8.8
        if 'ppm' in self.args:
            ppm = self.args['ppm']
        else:
            info ('ppm argument not specified. Using default value 8.8.')
        fps = 18
        if 'fps' in self.args:
            fps = self.args['fps']
        else:
            info ('fps argument not specified. Using default value 18.')
        fc = 10
        if 'fc' in self.args:
            fc = self.args['fc']
        else:
            info ('fc argument not specified. Using default value 10.')
        RECT_COLOR = (0, 255, 0)
        frame_counter = 0
        fps = 0
        current_car_id = 0
        car_tracker = {}
        car_location_1 = {} # Previous car location
        car_location_2 = {} # Current car location
        speed = [None] * 1000
        while not kbinput.KBINPUT: 
            start_time = time.time()
            _, frame = self.video.read()
            if frame is None:
                break
            result = frame.copy()
            frame_counter += 1 
            car_ids_to_delete = []
            for car_id in car_tracker.keys():
                psr = car_tracker[car_id].update(frame)
                if psr < 7:
                    car_ids_to_delete.append(car_id)
            for car_id in car_ids_to_delete:
                debug(f'Removing car id {car_id} + from list of tracked cars.')
                car_tracker.pop(car_id, None)
                car_location_1.pop(car_id, None)
                car_location_2.pop(car_id, None)
            
            if not (frame_counter % fc):
                #image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                #input_tensor = tflite_detect.set_input(interpreter, image.size,
                #            lambda size: image.resize(size, Image.ANTIALIAS))
                #interpreter.invoke()
                #cars = tflite_detect.get_output(interpreter, 0.6 , input_tensor)
                #cars = classifier.detectMultiScale(gray, 1.1, 13, 18, (24, 24))        
                cars = self.detect_objects(frame)
                for c in cars:
                    info('Object detected: %s (%.2f).' % (self.get_label_for_index(c.id), c.score))
                    x = int(c.bbox.xmin)
                    y = int(c.bbox.ymin)
                    w = int(c.bbox.xmax - c.bbox.xmin)
                    h = int(c.bbox.ymax - c.bbox.ymin)
                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h 
                    matched_car_id = None
                    for car_id in car_tracker.keys():
                        tracked_position = car_tracker[car_id].get_position()
                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())
                        
                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h
                    
                        if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                            matched_car_id = car_id
                    
                    if matched_car_id is None:
                        debug (f'Creating new car tracker with id {current_car_id}.' )
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(result, dlib.rectangle(x, y, x + w, y + h))
                        car_tracker[current_car_id] = tracker
                        car_location_1[current_car_id] = [x, y, w, h]
                        current_car_id += 1
            
            for car_id in car_tracker.keys():
                tracked_position = car_tracker[car_id].get_position()
                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())
                cv2.rectangle(result, (t_x, t_y), (t_x + t_w, t_y + t_h), RECT_COLOR, 4)
                car_location_2[car_id] = [t_x, t_y, t_w, t_h]
            
            end_time = time.time()
            if not (end_time == start_time):
                fps = 1.0/(end_time - start_time)
            cv2.putText(result, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            for i in car_location_1.keys():	
                if frame_counter % 1 == 0:
                    [x1, y1, w1, h1] = car_location_1[i]
                    [x2, y2, w2, h2] = car_location_2[i]
                    car_location_1[i] = [x2, y2, w2, h2]
                    if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                        # Estimate speed for a car object as it passes through a ROI.
                        if (speed[i] is None) and y1 >= 275 and y1 <= 285:
                            speed[i] = self.estimate_speed(ppm, fps, [x1, y1, w1, h1], [x2, y2, w2, h2])
                        if speed[i] is not None and y1 >= 180:
                            cv2.putText(result, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            if not nowindow:
                cv2.imshow('TrafficCV Haar cascade classifier speed detector. Press q to quit.', result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()


