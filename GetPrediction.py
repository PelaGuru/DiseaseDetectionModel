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


test_image = image.load_img('5.JPG', target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, 0)

test_image = test_image.reshape(1, 150, 150, 3)

result = new_model.predict(test_image, batch_size=1)
print(result)
