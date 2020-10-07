import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# setting the root directory path of the model
ROOT_DIR = 'C:\\Users\\damsa\\Desktop\\Pelaguru\\Disease Detection Model'

# settiing the directory of the dataset
PATH = os.path.join(ROOT_DIR, 'Dataset')

# setting the training and validation image directory paths
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# loading training images
train_cat1_dir = os.path.join(train_dir, '1 - Tomato_Spider_Mite_Damage')
train_cat2_dir = os.path.join(train_dir, '2 - Tomato_Early _Blight')
train_cat3_dir = os.path.join(train_dir, '3 - Tomato_Late_Blight')
train_cat4_dir = os.path.join(train_dir, '4 - Tomato_Leaf_Mold')

# loading validation images
validation_cat1_dir = os.path.join(
    validation_dir, '1 - Tomato_Spider_Mite_Damage')
validation_cat2_dir = os.path.join(validation_dir, '2 - Tomato_Early _Blight')
validation_cat3_dir = os.path.join(validation_dir, '3 - Tomato_Late_Blight')
validation_cat4_dir = os.path.join(validation_dir, '4 - Tomato_Leaf_Mold')

# to get the no of training images in each disease category
num_cat1_tr = len(os.listdir(train_cat1_dir))
num_cat2_tr = len(os.listdir(train_cat2_dir))
num_cat3_tr = len(os.listdir(train_cat3_dir))
num_cat4_tr = len(os.listdir(train_cat4_dir))

# to get the no of validation images in each disease category
num_cat1_val = len(os.listdir(validation_cat1_dir))
num_cat2_val = len(os.listdir(validation_cat2_dir))
num_cat3_val = len(os.listdir(validation_cat3_dir))
num_cat4_val = len(os.listdir(validation_cat4_dir))

# total no of test images
total_train = num_cat1_tr + num_cat2_tr + num_cat3_tr + num_cat4_tr

# total no of validation images
total_val = num_cat1_val + num_cat2_val + num_cat3_val + num_cat4_val

print('total training cat1 images:', num_cat1_tr)
print('total training cat2 images:', num_cat2_tr)
print('total training cat3 images:', num_cat3_tr)
print('total training cat4 images:', num_cat4_tr)

print('total validation cat1 images:', num_cat1_val)
print('total validation cat2 images:', num_cat2_val)
print('total validation cat3 images:', num_cat3_val)
print('total validation cat4 images:', num_cat4_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

# variables to use while pre-processing the dataset and training the network
batch_size = 128
epochs = 50
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Generator for our training data
train_image_generator = ImageDataGenerator(
    rescale=1./255,  rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5)

# Generator for our validation data
validation_image_generator = ImageDataGenerator(
    rescale=1./255)

# flow_from_directory method load images from the disk, applies rescaling, and resizes the images into the required dimensions
train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size, directory=train_dir, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical')

val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=batch_size, directory=validation_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical')


# next function returns a batch from the dataset. The return value of next function is in form of (x_train, y_train)
sample_training_images, _ = next(train_data_gen)


# function to plot images in the form of a grid with 1 row and 5 columns where images are placed in each column

# def plotImages(images_arr):
#     fig, axes = plt.subplots(1, 5, figsize=(20, 20))
#     axes = axes.flatten()
#     for img, ax in zip(images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()


# plotImages(sample_training_images[:5])
# augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)

def create_model():
    # creating the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(
            IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(4)
    ])

    # compiling the model
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(
        from_logits=True), metrics=['accuracy'])

    return model


# Create a basic model instance
model = create_model()

# to get an idea about the model like no of layers and their shape etc.
model.summary()

# setting the checkpoint path
# (checkpoints save parameter values in different iterations of the training process)
checkpoint_path = "training_4/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1)

# training the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
    callbacks=[cp_callback]
)

# Save the entire model which could be used later for getting predictions
model.save('detection_model4')

# data used when plotting accuracy and loss values
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# plotting the training accuracy and validation accuracy
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# plotting the training loss and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
