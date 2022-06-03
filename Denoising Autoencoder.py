#!/usr/bin/env python
# coding: utf-8

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Imports


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist


# Load the MNIST dataset

(X_train_orig, _), (X_test, _) = mnist.load_data()
# verify size
np.concatenate((X_train_orig, X_test)).shape


# Function to add noise to an image array

def add_noise(images, amount=0.1):
    corrupted = []
    for image in images:
        s_vs_p = 0.5
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, j - 1, int(num_salt))
                  for j in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, j - 1, int(num_pepper))
                  for j in image.shape]
        out[coords] = 0
        corrupted.append(tuple(out))
    return np.array(corrupted)


# Normalize RGB codes and create corrupted data

max_value = float(X_train_orig.max())
X_Train = X_train_orig.astype(np.float32) / max_value
X_Test = X_test.astype(np.float32) / max_value

#Train and validation split
X_train, X_valid = X_Train, X_Test
X_corrupted = add_noise(X_train, amount=0.4)
X_train.shape, X_valid.shape, X_corrupted.shape


# Plot the given image

def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")


# Create a plot to show 10 original and reconstructed images beneath each other.
# 
# When noise is True, the corrupted image will be displayed as well

def show_reconstructions(model, images=X_valid, n_images=10):
    reconstructions = model.predict(images[:n_images])
    plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        # add original image to plot
        plt.subplot(3, n_images, 1 + image_index)
        plot_image(images[image_index])
        # add noisy image to plot, if needed
        plt.subplot(3, n_images, 1 + n_images + image_index)
        plot_image(X_corrupted[image_index])
        # add reconstructed  image to plot
        plt.subplot(3, n_images, 1 + 2 * n_images + image_index)
        plot_image(reconstructions[image_index])


# 
# Train the autoencoder on a combination of different dense layer sizes and count, with noise.
# 
# | number of dense layers | dense layer size(ordered)  |
# |-----------------------:|:--------------------------:|
# |                      1 |            392             |
# |                      3 |      392 - 196 - 392       |
# |                      5 | 392 - 196 - 98 - 196 - 392 |

for i in range(3):
    inputs = keras.Input(shape=(28, 28))
    layer_flatten = keras.layers.Flatten()(inputs)
    if i == 0:
        layer1 = keras.layers.Dense(392, activation="selu")(layer_flatten)
        final_layer = keras.layers.Dense(28 * 28, activation="sigmoid")(layer1)
    elif i == 1:
        layer1 = keras.layers.Dense(392, activation="selu")(layer_flatten)
        layer2 = keras.layers.Dense(196, activation="selu")(layer1)
        layer3 = keras.layers.Dense(392, activation="selu")(layer2)
        final_layer = keras.layers.Dense(28 * 28, activation="sigmoid")(layer3)
    else:
        lr0 = keras.layers.Dense(392, activation="selu")(layer_flatten)
        layer1 = keras.layers.Dense(196, activation="selu")(lr0)
        layer2 = keras.layers.Dense(98, activation="selu")(layer1)
        layer3 = keras.layers.Dense(196, activation="selu")(layer2)
        layer4 = keras.layers.Dense(392, activation="selu")(layer3)
        final_layer = keras.layers.Dense(28 * 28, activation="sigmoid")(layer4)

    outputs = keras.layers.Reshape([28, 28])(final_layer)
    autoencoder = keras.models.Model(inputs, outputs)
    autoencoder.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=0.1))
    autoencoder.summary()
    callback = keras.callbacks.TensorBoard(
        log_dir="logs/noise_only/layers_" + str(i * 2 + 1),
        histogram_freq=0,
        write_graph=True,
        write_steps_per_second=False,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
    )

    h_stack = autoencoder.fit(X_corrupted, X_corrupted, epochs=50, validation_data=[X_valid, X_valid],
                              callbacks=[callback])
    show_reconstructions(autoencoder)

