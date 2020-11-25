# Based on https://github.com/ahmetozlu/vehicle_counting_tensorflow/blob/master/vehicle_detection_main.py
# vehicle_detection_main.py is Copyright (c) 2018 Ahmet Özlü
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 27th January 2018
# ----------------------------------------------
import os, sys
import numpy as np
import tensorflow as tf
import cv2

from utils import label_map_util
from utils import visualization_utils as vis_util

def make_interpreter(model_file):
  return tf.lite.Interpreter(model_path=model_file)

def load_image_into_numpy_array(image):
    """Load image into Numpy array"""
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,
            3)).astype(np.uint8)

def run(model, video, args):
    """Run detector and classifier."""
    model_file = os.path.join(model, 'ssd_mobilenet_v1_coco_quant_postprocess.tflite')
    labels = os.path.join('data', 'mscoco_label_map.pbtxt')
    num_classes = 90
    interpreter = make_interpreter(model_file)
    cap = cv2.VideoCapture(video)
    if args['info']:
        print(interpreter.get_input_details())
        sys.exit(0)

    # Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(labels)
    categories = label_map_util.convert_label_map_to_categories(label_map,
            max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    total_passed_vehicle = 0
    speed = 'waiting...'
    direction = 'waiting...'
    size = 'waiting...'
    color = 'waiting...'
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:

            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while cap.isOpened():
                (ret, frame) = cap.read()

                if not ret:
                    print ('end of the video file...')
                    break

                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = \
                    sess.run([detection_boxes, detection_scores,
                             detection_classes, num_detections],
                             feed_dict={image_tensor: image_np_expanded})

                # Visualization of the results of a detection.
                (counter, csv_line) = \
                    vis_util.visualize_boxes_and_labels_on_image_array(
                    cap.get(1),
                    input_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4,
                    )

                total_passed_vehicle = total_passed_vehicle + counter

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected Vehicles: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )

                # when the vehicle passed over line and counted, make the color of ROI line green
                if counter == 1:
                    cv2.line(input_frame, (0, 200), (640, 200), (0, 0xFF, 0), 5)
                else:
                    cv2.line(input_frame, (0, 200), (640, 200), (0, 0, 0xFF), 5)

                # insert information text to video frame
                cv2.rectangle(input_frame, (10, 275), (230, 337), (180, 132, 109), -1)
                cv2.putText(
                    input_frame,
                    'ROI Line',
                    (545, 190),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                    )
                cv2.putText(
                    input_frame,
                    'LAST PASSED VEHICLE INFO',
                    (11, 290),
                    font,
                    0.5,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )
                cv2.putText(
                    input_frame,
                    '-Movement Direction: ' + direction,
                    (14, 302),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.putText(
                    input_frame,
                    '-Speed(km/h): ' + speed,
                    (14, 312),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.putText(
                    input_frame,
                    '-Color: ' + color,
                    (14, 322),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.putText(
                    input_frame,
                    '-Vehicle Size/Type: ' + size,
                    (14, 332),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.imshow('OpenCV SSD MobileNet', input_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
