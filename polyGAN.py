import os
import tensorflow as tf
from keras.optimizers import SGD, Nadam, Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
import numpy as np
import math
import cv2
import keras.backend as K

# Defining image properties
input_dims = 100  # Number of random inputs per generated image.
img_side = 100  #length of the image in pixels
n_color_channels = 1  # Number of color channels: 3 for colored images (like cifar) and 1 for grayscale (like mnist).
batch_size = 20  # Number of images per batch.
epochs = 300  # Number of training epochs.


# Laoding healthy image dataset from folder
def load_from_folder(folder_name="D:\kavya\Healthy", subsample=2000):
    img_names = [f for f in os.listdir(folder_name) if not f.startswith('.')]
    x = []
    for img_name in img_names:
        im = cv2.imread(folder_name+"/" + img_name, cv2.IMREAD_GRAYSCALE)
        x.append(im)

    x = np.asarray(x)
    x = x.reshape(x.shape[0],x.shape[1],x.shape[2],1)
    print(x.shape)
    return x[:subsample]


#defining generator
def generator():
    model = Sequential()
    model.add(Dense(1024, input_dim=input_dims, activation="tanh"))
    model.add(Dense(64 * (img_side // 4) * (img_side // 4)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((img_side // 4, img_side // 4, 64), input_shape=(64 * (img_side // 4) * (img_side // 4,))))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(32, (5, 5), padding="same"))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(n_color_channels, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


# Defining Discriminator
def discriminator():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(img_side, img_side, n_color_channels)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

# Generating GAN by combining Discriminator and Generator
def GAN(gen, discrim):
    model = Sequential()
    model.add(gen)
    discrim.trainable = False
    model.add(discrim)
    return model

# combining generated images to display
def combine_images(gen_images):
    num = gen_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = gen_images.shape[1:]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]),
                     dtype=gen_images.dtype)
    for index, img in enumerate(gen_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img[:, :, :]
    return image

# Defining Training and plotting process
def train(training_images, batch_size=16, epochs=10, display_window=2):
    X_train = training_images

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5  # Convert from range (0 to 255) to range (-1 to 1)

    d = discriminator()
    g = generator()
    gan = GAN(g, d)
    d_optim = SGD()    # TODO: Try new optimizers. SGD is kinda shitty but fast.
    g_optim = SGD()
    gan.compile(loss="binary_crossentropy", optimizer=g_optim)
    d.trainable = True
    d.compile(loss="binary_crossentropy", optimizer=d_optim)
    for epoch in range(1, epochs + 1):
        print("Epoch is ", epoch)
        # print("Number of batches", int(X_train.shape[0] / batch_size))
        for index in range(X_train.shape[0] // batch_size):
            noise = np.random.uniform(-1, 1, size=(batch_size, input_dims))
            image_batch = X_train[index * batch_size: (index + 1) * batch_size]
            generated_images = g.predict(noise, verbose=0)

            if index % display_window == 0:
                # Stitch images into one image
                image = combine_images(generated_images)
                original = combine_images(image_batch)

                # Convert pixel intensity back to range (0-255) and cast to int
                image = (image * 127.5 + 127.5).astype(np.uint8)
                original = (original * 127.5 + 127.5).astype(np.uint8)

                zoomfactor = 2  # Zooming levels during display

                image = cv2.resize(image, (image.shape[0] * zoomfactor, image.shape[0] * zoomfactor))
                original = cv2.resize(original, (original.shape[0] * zoomfactor, original.shape[0] * zoomfactor))
                cv2.imshow('Output', image)
                cv2.imshow('Input', original)
                cv2.waitKey(1)

            # print(image_batch.shape, generated_images.shape)
            X = np.concatenate((image_batch, generated_images))
            y = [1] * batch_size + [0] * batch_size
            d_loss = d.train_on_batch(X, y)
            noise = np.random.uniform(-1, 1, size=(batch_size, input_dims))
            d.trainable = False
            g_loss = gan.train_on_batch(noise, [1] * batch_size)
            d.trainable = True

            # Saving images at the given interval
            if epoch % display_window == 0:
                cv2.imwrite('D:\kavya\code\outputs\image' + str(epoch) + ".png", image)


# Load appropriate input data
if __name__ == '__main__':
    input_data = load_from_folder(subsample=400)
    train(training_images= input_data,batch_size=batch_size, epochs=epochs, display_window=10)