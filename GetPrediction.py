import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image
import requests


def getPrediction(url):
    img_url = url

    new_model = tf.keras.models.load_model('detection_model2')

    # Check its architecture
    # new_model.summary()

    # predictions = new_model.predict('1.JPG')
    # print("predictions shape:", predictions.shape)

    img_data = requests.get(img_url).content
    with open('input_image.jpg', 'wb') as handler:
        handler.write(img_data)

    test_image = image.load_img('input_image.jpg', target_size=(150, 150))
    # print(test_image)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, 0)
    test_image /= 255.

    test_image = test_image.reshape(1, 150, 150, 3)

    result = new_model.predict(test_image, batch_size=1)
    index = np.argmax(result)

    if(index == 0):
        output_result = "\nTomato Spider Mite Damage"
    elif (index == 1):
        output_result = "\nTomato Early Blight"
    elif (index == 2):
        output_result = "\nTomato Late Blight"
    elif (index == 3):
        output_result = "\nTomato Leaf Mold"

    # print(result)

    return output_result


image_url = "https://firebasestorage.googleapis.com/v0/b/svd-attendance.appspot.com/o/images_for_prediction%2F22.png?alt=media&token=c1e6b2c5-32ea-4efb-ae5f-5b6cbe71a56d"

result = getPrediction(image_url)

print(result)
