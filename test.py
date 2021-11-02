       
import numpy as np
import tensorflow as tf
#from  tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging
import os
       
def test():

        model = load_model('trained_vgg_flowers.h5')

        # summarize model
        #model.summary()
        imagename = "00000482.jpg"
        test_image = image.load_img(imagename, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        print(round(result[0][1]))
        print(round(result[0][1]))
        print(round(result[0][2]))
        print(round(result[0][3]))
        

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

if __name__=="__main__":

   output= test()
   print(output)