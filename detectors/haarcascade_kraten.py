# Based on https://github.com/kraten/vehicle-speed-check/blob/master/speed_check.py 
# by Kartike Bansal.

import os
import sys
import time
import math
import logging

import cv2
import dlib

def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # ppm = location2[2] / carWidht
    ppm = 8.8
    d_meters = d_pixels / ppm
    #print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
    fps = 18
    speed = d_meters * fps * 3.6
    return speed

def run(model_dir, video_source):
    info = logging.info
    error = logging.error
    warn = logging.warn
    debug = logging.debug

    model_file = os.path.join(model_dir, 'haarcascade_kraten.xml')
    if not os.path.exists(model_file):
        error(f'{model_file} does not exist.')
        sys.exit(1)
    cascade = cv2.CascadeClassifier(model_file)
    video = cv2.VideoCapture(video_source)
    
    video_width = 1280
    video_height = 720
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0
    
    carTracker = {}
    carNumbers = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000
    
    while True:
        start_time = time.time()
        rc, image = video.read()
        if type(image) == type(None):
            break
        image = cv2.resize(image, (video_width, video_height))
        resultImage = image.copy()
        frameCounter = frameCounter + 1
        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)
            
            if trackingQuality < 7:
                carIDtoDelete.append(carID)
                
        for carID in carIDtoDelete:
            debug('Removing vehicle id ' + str(carID) + ' from list of trackers.')
            debug('Removing vehicle id ' + str(carID) + ' previous location.')
            debug('Removing vehicle id ' + str(carID) + ' current location.')
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)
        
        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = cascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
            
            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)
            
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h
                
                matchCarID = None
            
                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()
                    
                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())
                    
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h
                
                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID
                
                if matchCarID is None:
                    debug ('Creating new vehicle tracker for id ' + str(currentCarID))
                    
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                    
                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]

                    currentCarID = currentCarID + 1
        
        #cv2.line(resultImage,(0,480),(1280,480),(255,0,0),5)


        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()
                    
            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())
            
            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
            
            # speed estimation
            carLocation2[carID] = [t_x, t_y, t_w, t_h]
        
        end_time = time.time()
        
        if not (end_time == start_time):
            fps = 1.0/(end_time - start_time)
        
        cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


        for i in carLocation1.keys():	
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]
        
                #debug ('previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i]))
                carLocation1[i] = [x2, y2, w2, h2]
        
                #debug('new previous location: ' + str(carLocation1[i]))
                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

                    #if y1 > 275 and y1 < 285:
                    if speed[i] != None and y1 >= 180:
                        cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        #info ('Car id ' + str(i) + ': speed is ' + str(int(speed[i])) + " km/hr")

                    #else:
                    #	cv2.putText(resultImage, "Far Object", (int(x1 + w1/2), int(y1)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        #print ('CarID ' + str(i) + ' Location1: ' + str(carLocation1[i]) + ' Location2: ' + str(carLocation2[i]) + ' speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')
        cv2.imshow('OpenCV Haar Cascade', resultImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    cv2.destroyAllWindows()


