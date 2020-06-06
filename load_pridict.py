#!/usr/bin/env python
# coding: utf-8


from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from numpy import array, expand_dims
from keras.preprocessing import image


#Loading Our Traied Model
model = load_model("face_actor_VGG_TL.h5")


# load an image from file
testing_image = "./data/httpabsolumentgratuitfreefrimagesbenaffleckjpg.jpg"
image = load_img(testing_image, target_size=(224, 224))
#Showing the image
image.show(testing_image)

# Convert to array and make 4D
image = array(image)
image = expand_dims(image, axis=0)

#Decoding Predictions
if model.predict(image)[0][0] > 0.9:
    print("BEN_AFFLEK")
if model.predict(image)[0][1] > 0.9:
    print("ELTON_JOHN")
if model.predict(image)[0][2] > 0.9:
    print("JERRY_SEINFELD")
if model.predict(image)[0][3] > 0.9:
    print("MADONNA")
if model.predict(image)[0][4] > 0.9:
    print("MINDY_KALING")
