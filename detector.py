import sys
import abc
import math
import time
from queue import Queue
from logging import info, error, warn, debug

import numpy as np
import cv2
import dlib
import psutil

import kbinput

class Detector(abc.ABC):
    """A video object detector using a neural network or other computer vision learning model."""

    @abc.abstractmethod
    def get_label_for_index(self, i):
        """Get the string label for an integer index."""

    @abc.abstractmethod
    def detect_objects(self, frame, score_threshold):
        """Detect objects in a video frame."""

    @abc.abstractmethod
    def print_model_info(self):
        """Print out information on model."""

    def __init__(self, name, model_dir, video_source, args):
        self.name = name
        self.model_dir = model_dir
        self.video_source = video_source
        self.video = cv2.VideoCapture(self.video_source)
        self.video_end = False
        if self.video_source == 0:
                self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.video.set(cv2.CAP_PROP_FPS, 60)
        self._height, self._width, self._fps = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FPS)) 
        info(f'Video resolution: {self._width}x{self._height} {self._fps}fps.')
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
        fc = 10
        if 'fc' in self.args:
            fc = self.args['fc']
        else:
            info ('fc argument not specified. Using default value 10.')
        score = 0.6
        if 'score' in self.args:
            score = self.args['score']
        else:
            info ('score argument not specified. Using default score threshold 0.6.')

        RECT_COLOR = (0, 255, 0)
        frame_counter = 0.0
        fps = 0.0
        current_car_id = 0
        car_tracker = {}
        car_location_1 = {} # Previous car location
        car_location_2 = {} # Current car location
        speed = [None] * 1000
        start_time = time.time()
        while not kbinput.KBINPUT:
            _, frame = self.video.read()
            if frame is None:
                break
            result = frame.copy()
            frame_counter += 1.0 
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
                cars = self.detect_objects(frame, score)
                for c in cars:
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
                        info (f'New object detected at {x, y, w, h} with id {current_car_id} and confidence score {c.score}.')
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
            fps = frame_counter / (end_time - start_time)
            for i in car_location_1.keys():	                
                [x1, y1, w1, h1] = car_location_1[i]
                [x2, y2, w2, h2] = car_location_2[i]
                car_location_1[i] = [x2, y2, w2, h2]
                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    speed[i] = self.estimate_speed(ppm, fps, [x1, y1, w1, h1], [x2, y2, w2, h2])
                    cv2.putText(result, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
            cv2.putText(result, 'Source FPS: ' + str(int(self._fps)), (0, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)
            cv2.putText(result, 'Internal FPS: ' + str(int(fps)), (0, 45),cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)
            if (int(frame_counter) % fc == 0):
                cpu_p = ""
                for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=None)):
                    cpu_p += f"CPU#{i + 1}: {percentage}%; " 
                info (f'Internal FPS: {int(fps)}; {cpu_p.strip()} Objects currently tracked: {len(car_tracker)}.')
            if not nowindow:
                cv2.imshow(f'TrafficCV {self.name}. Press any key to quit. ', result)
                if cv2.waitKey(1) != -1:
                    break
        cv2.destroyAllWindows()


