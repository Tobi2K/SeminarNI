#!/usr/bin/env python
# coding: utf-8

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Inspired by <https://medium.com/@sahoo.puspanjali58/a-beginners-guide-to-build-stacked-autoencoder-and-tying-weights-with-it-9daee61eab2b>
# 
# Imports

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist


# Select which configurations to run
epochs = False
dense_layer_size = False
dense_layer_count = False
compare_noise = True


# Load the MNIST dataset
(X_train_orig, _), (X_test, _) = mnist.load_data()
# verify size
np.concatenate((X_train_orig, X_test)).shape


# Function to add noise to an image array, taken from <https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv>

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
# needed for displaying later
X_valid_but_noisy = add_noise(X_valid, amount=0.4)
X_train.shape, X_valid.shape, X_corrupted.shape


# Plot the given image

def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")


# Create a plot to show 10 original and reconstructed images beneath each other.

# When noise is True, the corrupted image will be displayed as well

#Displays the original images and their reconstructions
def show_reconstructions(model, images=X_valid, n_images=10, noise=False):
    reconstructions = model.predict(images[:n_images])
    plt.figure(figsize=(n_images * 1.5, 3))
    cols = 2
    if noise:
        cols = 3
    for image_index in range(n_images):
        # add original image to plot
        plt.subplot(cols, n_images, 1 + image_index)
        plot_image(images[image_index])
        # add noisy image to plot, if needed
        if noise:
            plt.subplot(cols, n_images, 1 + n_images + image_index)
            plot_image(X_valid_but_noisy[image_index])
        # add reconstructed  image to plot
        plt.subplot(cols, n_images, 1 + (cols - 1) * n_images + image_index)
        plot_image(reconstructions[image_index])


# Train the autoencoder on a different number of epochs (runs)

if epochs:
    runs = [1, 5, 10, 20, 50]

    for i in runs:
        #Stacked Autoencoder with functional model
        #encoder
        inputs = keras.Input(shape=(28, 28))
        layer_flatten = keras.layers.Flatten()(inputs)
        layer1 = keras.layers.Dense(392, activation="selu")(layer_flatten)
        layer2 = keras.layers.Dense(28 * 28, activation="sigmoid")(layer1)
        outputs = keras.layers.Reshape([28, 28])(layer2)
        autoencoder = keras.models.Model(inputs, outputs)
        autoencoder.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=0.1))
        autoencoder.summary()
        callbacks = keras.callbacks.TensorBoard(
            log_dir="logs/epochs/number_of_epochs_" + str(i),
            histogram_freq=0,
            write_graph=True,
            write_steps_per_second=False,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )

        h_stack = autoencoder.fit(X_train, X_train, epochs=i, validation_data=[X_valid, X_valid], callbacks=[callbacks])
        show_reconstructions(autoencoder)


# Train the autoencoder on different dense layer sizes. Here only one dense layer is used.

if dense_layer_size:
    layer_size = [392, 196, 98]

    for i in layer_size:
        #encoder
        inputs = keras.Input(shape=(28, 28))
        lr_flatten = keras.layers.Flatten()(inputs)
        layer1 = keras.layers.Dense(i, activation="selu")(lr_flatten)
        layer2 = keras.layers.Dense(28 * 28, activation="sigmoid")(layer1)
        outputs = keras.layers.Reshape([28, 28])(layer2)
        autoencoder = keras.models.Model(inputs, outputs)
        autoencoder.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=0.1))
        autoencoder.summary()
        callbacks = keras.callbacks.TensorBoard(
            log_dir="logs/layer_size/layer_size_" + str(i),
            histogram_freq=0,
            write_graph=True,
            write_steps_per_second=False,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )

        h_stack = autoencoder.fit(X_train, X_train, epochs=50, validation_data=[X_valid, X_valid], callbacks=[callbacks])
        show_reconstructions(autoencoder)


# Train the autoencoder on a combination of different dense layer sizes and count.
# 
# |  number of dense layers |         dense layer size (ordered)         |
# |------------------------:|:------------------------------------------:|
# |                       1 |                    392                     |
# |                       3 |                392-196-392                 |
# |                       3 |                 392-98-392                 |
# |                       3 |                 196-98-196                 |
# |                       5 |             392-196-98-196-392             |

if dense_layer_count:
    for i in range(5):
        #Stacked Autoencoder with functional model
        #encoder
        inputs = keras.Input(shape=(28, 28))
        lr_flatten = keras.layers.Flatten()(inputs)
        if i == 0:
            layer1 = keras.layers.Dense(392, activation="selu")(lr_flatten)
            final_layer = keras.layers.Dense(28 * 28, activation="sigmoid")(layer1)
        elif i == 1:
            layer1 = keras.layers.Dense(392, activation="selu")(lr_flatten)
            layer2 = keras.layers.Dense(196, activation="selu")(layer1)
            layer3 = keras.layers.Dense(392, activation="selu")(layer2)
            final_layer = keras.layers.Dense(28 * 28, activation="sigmoid")(layer3)
        elif i == 2:
            layer1 = keras.layers.Dense(392, activation="selu")(lr_flatten)
            layer2 = keras.layers.Dense(98, activation="selu")(layer1)
            layer3 = keras.layers.Dense(392, activation="selu")(layer2)
            final_layer = keras.layers.Dense(28 * 28, activation="sigmoid")(layer3)
        elif i == 3:
            layer1 = keras.layers.Dense(196, activation="selu")(lr_flatten)
            layer2 = keras.layers.Dense(98, activation="selu")(layer1)
            layer3 = keras.layers.Dense(196, activation="selu")(layer2)
            final_layer = keras.layers.Dense(28 * 28, activation="sigmoid")(layer3)
        else:
            layer1 = keras.layers.Dense(392, activation="selu")(lr_flatten)
            layer2 = keras.layers.Dense(196, activation="selu")(layer1)
            layer3 = keras.layers.Dense(98, activation="selu")(layer2)
            layer4 = keras.layers.Dense(196, activation="selu")(layer3)
            layer5 = keras.layers.Dense(392, activation="selu")(layer4)
            final_layer = keras.layers.Dense(28 * 28, activation="sigmoid")(layer5)


        outputs = keras.layers.Reshape([28, 28])(final_layer)
        autoencoder = keras.models.Model(inputs, outputs)
        autoencoder.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=0.1))
        autoencoder.summary()
        callback = keras.callbacks.TensorBoard(
            log_dir="logs/layer_count/run_" + str(i),
            histogram_freq=0,
            write_graph=True,
            write_steps_per_second=False,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )

        h_stack = autoencoder.fit(X_train, X_train, epochs=50, validation_data=[X_valid, X_valid], callbacks=[callback])
        show_reconstructions(autoencoder)


# Train the autoencoder on a combination of different dense layer sizes and count. Additionally, every run is repeated on the corrupted dataset.
# 
# |  number of dense layers |         dense layer size (ordered)         | noise |
# |------------------------:|:------------------------------------------:|-------|
# |                       1 |                    392                     | none  |
# |                       3 |                392-196-392                 | none  |
# |                       5 |             392-196-98-196-392             | none  |
# |                       1 |                    392                     | yes   |
# |                       3 |                392-196-392                 | yes   |
# |                       5 |             392-196-98-196-392             | yes   |

# In[14]:


if compare_noise:
    for i in range(3):
        inputs = keras.Input(shape=(28, 28))
        lr_flatten = keras.layers.Flatten()(inputs)
        if i == 0:
            layer1 = keras.layers.Dense(392, activation="selu")(lr_flatten)
            final_layer = keras.layers.Dense(28 * 28, activation="sigmoid")(layer1)
        elif i == 1:
            layer1 = keras.layers.Dense(392, activation="selu")(lr_flatten)
            layer2 = keras.layers.Dense(196, activation="selu")(layer1)
            layer3 = keras.layers.Dense(392, activation="selu")(layer2)
            final_layer = keras.layers.Dense(28 * 28, activation="sigmoid")(layer3)
        else:
            layer1 = keras.layers.Dense(392, activation="selu")(lr_flatten)
            layer2 = keras.layers.Dense(196, activation="selu")(layer1)
            layer3 = keras.layers.Dense(98, activation="selu")(layer2)
            layer4 = keras.layers.Dense(196, activation="selu")(layer3)
            layer5 = keras.layers.Dense(392, activation="selu")(layer4)
            final_layer = keras.layers.Dense(28 * 28, activation="sigmoid")(layer5)


        outputs = keras.layers.Reshape([28, 28])(final_layer)
        autoencoder = keras.models.Model(inputs, outputs)
        autoencoder.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=0.1))
        autoencoder.summary()
        callback = keras.callbacks.TensorBoard(
            log_dir="logs/compare_noise/no_noise_layers_" + str(i * 2 + 1),
            histogram_freq=0,
            write_graph=True,
            write_steps_per_second=False,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )

        h_stack = autoencoder.fit(X_train, X_train, epochs=50, validation_data=[X_valid, X_valid], callbacks=[callback])
        show_reconstructions(autoencoder)

    for i in range(3):
        inputs = keras.Input(shape=(28, 28))
        lr_flatten = keras.layers.Flatten()(inputs)
        if i == 0:
            layer1 = keras.layers.Dense(392, activation="selu")(lr_flatten)
            final_layer = keras.layers.Dense(28 * 28, activation="sigmoid")(layer1)
        elif i == 1:
            layer1 = keras.layers.Dense(392, activation="selu")(lr_flatten)
            layer2 = keras.layers.Dense(196, activation="selu")(layer1)
            layer3 = keras.layers.Dense(392, activation="selu")(layer2)
            final_layer = keras.layers.Dense(28 * 28, activation="sigmoid")(layer3)
        else:
            layer1 = keras.layers.Dense(392, activation="selu")(lr_flatten)
            layer2 = keras.layers.Dense(196, activation="selu")(layer1)
            layer3 = keras.layers.Dense(98, activation="selu")(layer2)
            layer4 = keras.layers.Dense(196, activation="selu")(layer3)
            layer5 = keras.layers.Dense(392, activation="selu")(layer4)
            final_layer = keras.layers.Dense(28 * 28, activation="sigmoid")(layer5)


        outputs = keras.layers.Reshape([28, 28])(final_layer)
        autoencoder = keras.models.Model(inputs, outputs)
        autoencoder.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=0.1))
        autoencoder.summary()
        callback = keras.callbacks.TensorBoard(
            log_dir="logs/compare_noise/with_noise_layers_" + str(i * 2 + 1),
            histogram_freq=0,
            write_graph=True,
            write_steps_per_second=False,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )

        h_stack = autoencoder.fit(X_corrupted, X_corrupted, epochs=50, validation_data=[X_valid, X_valid], callbacks=[callback])
        show_reconstructions(autoencoder, noise=True)
