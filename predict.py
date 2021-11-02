#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:45:05 2020

@author: premchand
"""

import numpy as np
import tensorflow as tf
#from  tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

class dogcat:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogcat(self):
        # load model artifacts\model\model_at_Thu_Oct_28_11_03_40_2021_.h5
        model = load_model('trained_vgg_flowers.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        logging.info(result)
        

        if round(result[0][0]) == 1:
            prediction = 'backpack'
            return [{ "image" : prediction}]
        elif round(result[0][1]) == 1:
            prediction = 'footwear'
            return [{ "image" : prediction}]

        elif round(result[0][2]) == 1:
            prediction = 'Glasses'
            return [{ "image" : prediction}]
        elif round(result[0][3]) == 1:
            prediction = 'Watch'
            return [{ "image" : prediction}]

        else:
            prediction = 'Wrong input image'
            return [{ "image" : prediction}]


