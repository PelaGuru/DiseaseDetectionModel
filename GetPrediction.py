import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image
import requests

# function which predicts the disease of a given image(URL) using the saved pre-trained model


def getPrediction(url):
    img_url = url

    # loading the pre-trained model
    new_model = tf.keras.models.load_model('detection_model2')

    # Check its architecture
    # new_model.summary()

    # getting the image URL and then convert it to a PIL image instance
    img_data = requests.get(img_url).content
    with open('input_image.jpg', 'wb') as handler:
        handler.write(img_data)

    # pre-processing the image to be predicted
    test_image = image.load_img('input_image.jpg', target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, 0)
    test_image /= 255.
    test_image = test_image.reshape(1, 150, 150, 3)

    # getting the prediction of the image by sending it to the model
    result = new_model.predict(test_image, batch_size=1)

    # to get the index of the max result value out of 4 disease categories
    index = np.argmax(result)

    # to get the max result value out of 4 disease categories
    value = np.amax(result)

    # to get the prediction shape
    # print("predictions shape:", result.shape)

    # mapping the indexe with the disease ID
    if(index == 0):
        # disease category - Tomato Spider Mite Damage
        output_result = "TddVuqg5EF83D0XsAGrXH5obFI"
    elif (index == 1):
        # disease category - Tomato Early Blight
        output_result = "hFDZAhAHnRDCpREGIZkrFTLALj"
    elif (index == 2):
        # disease category - Tomato Late Blight
        output_result = "ChTVgFRaBB5YCSCVGiFAjmBGST"
    elif (index == 3):
        # disease category - Tomato Leaf Mold
        output_result = "lJwXpFFBDMpCoADABiASPOMlpB"

    # creating a python dictionary to store results
    d = dict()
    d['predicted_result'] = output_result
    d['value'] = value

    # return the results dictionary
    return d


# to test an image prediction
# when calling the API this part will not be executed
image_url = "https://firebasestorage.googleapis.com/v0/b/pelaguru-dev.appspot.com/o/uploadImages%2F11.jpg?alt=media&token=2cb6e9f4-e1f1-41cd-9abd-665bfe0b7b20"
result = getPrediction(image_url)
print(result)
