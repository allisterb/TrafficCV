# Based on https://github.com/kraten/vehicle-speed-check/blob/master/speed_check.py 
# by Kartike Bansal.

import os
import sys
import time
import math
import logging

import cv2
import dlib

def estimate_speed(location1, location2):
    """Estimate the speed of a vehicle assuming pixel-per-metre and fps constants."""
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # Pixel-per-m constant
    ppm = 8.8
    # fps constant
    fps = 18
    d_meters = d_pixels / ppm
    speed = d_meters * fps * 3.6
    return speed

def run(model_dir, video_source):
    """Run the classifier and detector."""
    info = logging.info
    error = logging.error
    warn = logging.warn
    debug = logging.debug

    model_file = os.path.join(model_dir, 'haarcascade_kraten.xml')
    if not os.path.exists(model_file):
        error(f'{model_file} does not exist.')
        sys.exit(1)
    classifier = cv2.CascadeClassifier(model_file)
    video = cv2.VideoCapture(video_source)
    
    VIDEO_WIDTH = 1280
    VIDEO_HEIGHT = 720
    RECT_COLOR = (0, 255, 0)
    
    frame_counter = 0
    fps = 0
    current_car_id = 0
    car_tracker = {}
    car_location_1 = {} # Previous car location
    car_location_2 = {} # Current car location
    speed = [None] * 1000
    
    while True:
        start_time = time.time()
        _, image = video.read()
        if image is None:
            break
        image = cv2.resize(image, (VIDEO_WIDTH, VIDEO_HEIGHT))
        result = image.copy()
        frame_counter += 1 
        car_ids_to_delete = []

        for car_id in car_tracker.keys():
            tracking_quality = car_tracker[car_id].update(image)
            
            if tracking_quality < 7:
                car_ids_to_delete.append(car_id)
                
        for car_id in car_ids_to_delete:
            debug('Removing car id ' + str(car_id) + ' from list of tracked cars.')
            # debug('Removing vehicle id ' + str(car_id) + ' previous location.')
            # debug('Removing vehicle id ' + str(car_id) + ' current location.')
            car_tracker.pop(car_id, None)
            car_location_1.pop(car_id, None)
            car_location_2.pop(car_id, None)
        
        if not (frame_counter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = classifier.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
            
            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)
            
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
                    debug ('Creating new vehicle tracker with id ' + str(current_car_id))
                    
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                    
                    car_tracker[current_car_id] = tracker
                    car_location_1[current_car_id] = [x, y, w, h]

                    current_car_id = current_car_id + 1
        
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
                    if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        speed[i] = estimate_speed([x1, y1, w1, h1], [x2, y2, w2, h2])

                    #if y1 > 275 and y1 < 285:
                    if speed[i] != None and y1 >= 180:
                        cv2.putText(result, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow('OpenCV Haar Cascade', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()