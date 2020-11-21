"""Convert models to TFLite"""
import os
import sys

import tensorflow as tf

def convert(model_dir):
    """Convert a TensorFlow model to TensorFlow Lite."""
    if os.path.isdir(model_dir):
        print(f'Attempting to convert model in directory {model_dir} to TFLite.')
        converter = tf.compat.v1.lite.TFLiteConverter.from_session(//(from_saved_model(model_dir)
        tflite_model = converter.convert()
        with open(model_dir + '.tflite', 'wb') as f:
            f.write(tflite_model)
        size = os.path.getsize(tflite_model)
        print(f'Size of {tflite_model} is {size / (1024 * 1024)}M.')
    else:
        print(f'Converting file {model_dir} is not supported.')
        sys.exit(1)
