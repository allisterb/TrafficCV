# Based on speed_estimation_dl_video.py 
# OpenCV Vehicle Detection, Tracking, and Speed Estimation by Adrian Rosebrock
# https://www.pyimagesearch.com/2019/12/02/opencv-vehicle-detection-tracking-and-speed-estimation/
# © 2020 PyImageSearch

import time
import os
import argparse
from datetime import datetime
from threading import Thread
from collections import OrderedDict
import json
import logging

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils.io import TempFile
from imutils.video import FPS
from json_minify import json_minify
import numpy as np
import imutils
import dlib
import cv2

class Conf:
    def __init__(self, confPath):
        conf = json.loads(json_minify(open(confPath).read()))
        self.__dict__.update(conf)

    def __getitem__(self, k):
        return self.__dict__.get(k, None)

class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

        # store the maximum distance between centroids to associate
        # an object -- if the distance is larger than this maximum
        # distance we'll start to mark the object as "disappeared"
        self.maxDistance = maxDistance

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                # if the distance between centroids is greater than
                # the maximum distance, do not associate the two
                # centroids to the same object
                if D[row, col] > self.maxDistance:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects

class TrackableObject:
    def __init__(self, objectID, centroid):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]

        # initialize a dictionaries to store the timestamp and
        # position of the object at various points
        self.timestamp = {"A": 0, "B": 0, "C": 0, "D": 0}
        self.position = {"A": None, "B": None, "C": None, "D": None}
        self.lastPoint = False
        
        # initialize the object speeds in MPH and KMPH
        self.speedMPH = None
        self.speedKMPH = None

        # initialize two booleans, (1) used to indicate if the
        # object's speed has already been estimated or not, and (2)
        # used to indidicate if the object's speed has been logged or
        # not
        self.estimated = False
        self.logged = False

        # initialize the direction of the object
        self.direction = None

    def calculate_speed(self, estimatedSpeeds):
        # calculate the speed in KMPH and MPH
        self.speedKMPH = np.average(estimatedSpeeds)
        MILES_PER_ONE_KILOMETER = 0.621371
        self.speedMPH = self.speedKMPH * MILES_PER_ONE_KILOMETER

def run(video):
    info = logging.info	
    error = logging.error
    warn = logging.warn
    
    debug = logging.debug
    # load the configuration file
    conf = Conf(os.path.join('models', 'ssd_mobilenet_caffe', 'config.json'))

    # initialize the list of class labels MobileNet SSD was trained to
    # detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    # load our serialized model from disk
    info("Loading Caffe model...")
    net = cv2.dnn.readNetFromCaffe(os.path.join('models', 'ssd_mobilenet_caffe', 'MobileNetSSD_deploy.prototxt'),
        os.path.join('models', 'ssd_mobilenet_caffe', 'MobileNetSSD_deploy.caffemodel'))
    #net.setPreferableTarget(cv2.dnn.)

    # initialize the video stream and allow the camera sensor to warmup
    vs = cv2.VideoCapture(video)
    #time.sleep(2.0)

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    H = None
    W = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=conf["max_disappear"],
        maxDistance=conf["max_distance"])
    trackers = []
    trackableObjects = {}

    # keep the count of total number of frames
    totalFrames = 0

    # initialize the log file
    logFile = None

    # initialize the list of various points used to calculate the avg of
    # the vehicle speed
    points = [("A", "B"), ("B", "C"), ("C", "D")]

    # start the frames per second throughput estimator
    fps = FPS().start()

    # loop over the frames of the stream
    while True:
        # grab the next frame from the stream, store the current
        # timestamp, and store the new date
        ret, frame  = vs.read()
        ts = datetime.now()
        newDate = ts.strftime("%m-%d-%y")

        # check if the frame is None, if so, break out of the loop
        if frame is None:
            break

        if logFile is None:
            # build the log file path and create/open the log file
            logPath = os.path.join(conf["output_path"], conf["csv_name"])
            logFile = open(logPath, mode="a")
        pos = logFile.seek(0, os.SEEK_END)
        if pos == 0:
            logFile.write("Year,Month,Day,Time (in MPH),Speed\n")

        # resize the frame
        frame = imutils.resize(frame, width=conf["frame_width"])
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            meterPerPixel = conf["distance"] / W

        # initialize our list of bounding box rectangles returned by
        # either (1) our object detector or (2) the correlation trackers
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % conf["track_object"] == 0:
            # initialize our new set of object trackers
            trackers = []

            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300),
                ddepth=cv2.CV_8U)
            net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5,
                127.5, 127.5])
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence`
                # is greater than the minimum confidence
                if confidence > conf["confidence"]:
                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])

                    # if the class label is not a car, ignore it
                    if CLASSES[idx] != "car":
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    trackers.append(tracker)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing
        # throughput
        else:
            # loop over the trackers
            for tracker in trackers:
                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, if there is a trackable object and its speed has
            # not yet been estimated then estimate it
            elif not to.estimated:
                # check if the direction of the object has been set, if
                # not, calculate it, and set it
                if to.direction is None:
                    y = [c[0] for c in to.centroids]
                    direction = centroid[0] - np.mean(y)
                    to.direction = direction

                # if the direction is positive (indicating the object
                # is moving from left to right)
                if to.direction > 0:
                    # check to see if timestamp has been noted for
                    # point A
                    if to.timestamp["A"] == 0 :
                        # if the centroid's x-coordinate is greater than
                        # the corresponding point then set the timestamp
                        # as current timestamp and set the position as the
                        # centroid's x-coordinate
                        if centroid[0] > conf["speed_estimation_zone"]["A"]:
                            to.timestamp["A"] = ts
                            to.position["A"] = centroid[0]

                    # check to see if timestamp has been noted for
                    # point B
                    elif to.timestamp["B"] == 0:
                        # if the centroid's x-coordinate is greater than
                        # the corresponding point then set the timestamp
                        # as current timestamp and set the position as the
                        # centroid's x-coordinate
                        if centroid[0] > conf["speed_estimation_zone"]["B"]:
                            to.timestamp["B"] = ts
                            to.position["B"] = centroid[0]

                    # check to see if timestamp has been noted for
                    # point C
                    elif to.timestamp["C"] == 0:
                        # if the centroid's x-coordinate is greater than
                        # the corresponding point then set the timestamp
                        # as current timestamp and set the position as the
                        # centroid's x-coordinate
                        if centroid[0] > conf["speed_estimation_zone"]["C"]:
                            to.timestamp["C"] = ts
                            to.position["C"] = centroid[0]

                    # check to see if timestamp has been noted for
                    # point D
                    elif to.timestamp["D"] == 0:
                        # if the centroid's x-coordinate is greater than
                        # the corresponding point then set the timestamp
                        # as current timestamp, set the position as the
                        # centroid's x-coordinate, and set the last point
                        # flag as True
                        if centroid[0] > conf["speed_estimation_zone"]["D"]:
                            to.timestamp["D"] = ts
                            to.position["D"] = centroid[0]
                            to.lastPoint = True

                # if the direction is negative (indicating the object
                # is moving from right to left)
                elif to.direction < 0:
                    # check to see if timestamp has been noted for
                    # point D
                    if to.timestamp["D"] == 0 :
                        # if the centroid's x-coordinate is lesser than
                        # the corresponding point then set the timestamp
                        # as current timestamp and set the position as the
                        # centroid's x-coordinate
                        if centroid[0] < conf["speed_estimation_zone"]["D"]:
                            to.timestamp["D"] = ts
                            to.position["D"] = centroid[0]

                    # check to see if timestamp has been noted for
                    # point C
                    elif to.timestamp["C"] == 0:
                        # if the centroid's x-coordinate is lesser than
                        # the corresponding point then set the timestamp
                        # as current timestamp and set the position as the
                        # centroid's x-coordinate
                        if centroid[0] < conf["speed_estimation_zone"]["C"]:
                            to.timestamp["C"] = ts
                            to.position["C"] = centroid[0]

                    # check to see if timestamp has been noted for
                    # point B
                    elif to.timestamp["B"] == 0:
                        # if the centroid's x-coordinate is lesser than
                        # the corresponding point then set the timestamp
                        # as current timestamp and set the position as the
                        # centroid's x-coordinate
                        if centroid[0] < conf["speed_estimation_zone"]["B"]:
                            to.timestamp["B"] = ts
                            to.position["B"] = centroid[0]

                    # check to see if timestamp has been noted for
                    # point A
                    elif to.timestamp["A"] == 0:
                        # if the centroid's x-coordinate is lesser than
                        # the corresponding point then set the timestamp
                        # as current timestamp, set the position as the
                        # centroid's x-coordinate, and set the last point
                        # flag as True
                        if centroid[0] < conf["speed_estimation_zone"]["A"]:
                            to.timestamp["A"] = ts
                            to.position["A"] = centroid[0]
                            to.lastPoint = True

                # check to see if the vehicle is past the last point and
                # the vehicle's speed has not yet been estimated, if yes,
                # then calculate the vehicle speed and log it if it's
                # over the limit
                if to.lastPoint and not to.estimated:
                    # initialize the list of estimated speeds
                    estimatedSpeeds = []

                    # loop over all the pairs of points and estimate the
                    # vehicle speed
                    for (i, j) in points:
                        # calculate the distance in pixels
                        d = to.position[j] - to.position[i]
                        distanceInPixels = abs(d)

                        # check if the distance in pixels is zero, if so,
                        # skip this iteration
                        if distanceInPixels == 0:
                            continue

                        # calculate the time in hours
                        t = to.timestamp[j] - to.timestamp[i]
                        timeInSeconds = abs(t.total_seconds())
                        timeInHours = timeInSeconds / (60 * 60)

                        # calculate distance in kilometers and append the
                        # calculated speed to the list
                        distanceInMeters = distanceInPixels * meterPerPixel
                        distanceInKM = distanceInMeters / 1000
                        estimatedSpeeds.append(distanceInKM / timeInHours)

                    # calculate the average speed
                    to.calculate_speed(estimatedSpeeds)

                    # set the object as estimated
                    to.estimated = True
                    print("Speed of the vehicle that just passed"\
                        " is: {:.2f} MPH".format(to.speedMPH))

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10)
                , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4,
                (0, 255, 0), -1)

            # check if the object has not been logged
            if not to.logged:
                # check if the object's speed has been estimated and it
                # is higher than the speed limit
                if to.estimated and to.speedMPH > conf["speed_limit"]:
                    # set the current year, month, day, and time
                    year = ts.strftime("%Y")
                    month = ts.strftime("%m")
                    day = ts.strftime("%d")
                    time = ts.strftime("%H:%M:%S")

                    # log the event in the log file
                    info = "{},{},{},{},{}\n".format(year, month,
                        day, time, to.speedMPH)
                    logFile.write(info)

                    # set the object has logged
                    to.logged = True

        # if the *display* flag is set, then display the current frame
        # to the screen and record if a user presses a key
        if conf["display"]:
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key is pressed, break from the loop
            if key == ord("q"):
                break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("elapsed time: {:.2f}".format(fps.elapsed()))
    print("approx. FPS: {:.2f}".format(fps.fps()))
    # check if the log file object exists, if it does, then close it
    if logFile is not None:
        logFile.close()
    cv2.destroyAllWindows()
    vs.release()