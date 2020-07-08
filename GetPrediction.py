import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image

new_model = tf.keras.models.load_model('detection_model2')

# Check its architecture
new_model.summary()

# predictions = new_model.predict('1.JPG')
# print("predictions shape:", predictions.shape)


test_image = image.load_img('22.JPG', target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, 0)
test_image /= 255.

test_image = test_image.reshape(1, 150, 150, 3)

result = new_model.predict(test_image, batch_size=1)
index = np.argmax(result)

if(index == 0):
    print("\nTomato Spider Mite Damage")
elif (index == 1):
    print("\nTomato Early Blight")
elif (index == 2):
    print("\nTomato Late Blight")
elif (index == 3):
    print("\nTomato Leaf Mold")

print(result)
