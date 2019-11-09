#!/usr/bin/env python
# coding: utf-8
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
import os

MODEL_NAME = 'inceptionv3_perceptual_model.h5'
MODELS_FILE = 'perceptual_models'
MODEL_PATH = os.path.join(MODELS_FILE, MODEL_NAME)


def create_file_structure():
    if not os.path.exists(MODELS_FILE):
        os.makedirs(MODELS_FILE)

def create_perceptual_model(layer_num):
    """
    Creates smaller model by cropping trained InceptionV3 model.

    Params:
        layer_num (int): Number of convolutional blocks to keep in perceptual model.
    Returns:
        Smaller keras model obtained from InceptionV3
    """
    # Create folder to save the model inside it
    create_file_structure()
    # Download InceptionV3 model with imagenet weights
    print("Downloading InceptionV3 model")
    inception_model = InceptionV3(include_top=False, weights='imagenet')
    # Get smaller model by cropping InceptionV3 till desired conv layer
    last_layer = inception_model.get_layer('conv2d_{}'.format(layer_num))
    # Create new model and save it
    cropped_model = Model(inception_model.input, last_layer.output, name='perceptual-model')
    print("Saving model")
    cropped_model.save(MODEL_PATH)
    print("Summary of saved model")
    cropped_model.summary()
    return cropped_model

def get_perceptual_model():
    """
    Return perceptual model
    """
    if not os.path.exists(MODEL_PATH):
        print("Cannot found perceptual model. Call create_perceptual_model first")
        return None
    return load_model(MODEL_PATH)


if __name__ == '__main__':
    create_perceptual_model(4)
