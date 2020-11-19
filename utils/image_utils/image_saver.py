# Based on https://github.com/ahmetozlu/vehicle_counting_tensorflow/blob/master/utils/image_utils/image_saver.py
# image_saver.py is Copyright (c) 2018 Ozlu
import os
import cv2

vehicle_count = [0]

def save_image(source_image):
    """Write a detected vehicle image to disk."""
    cv2.imwrite(os.path.join(os.getcwd(), 'detected_vehicles', 'vehicle'
                + str(len(vehicle_count)) + '.png', source_image))
    vehicle_count.insert(0, 1)