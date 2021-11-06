
# flowers_nn.py - Fang Xi Lin 2021

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow.keras.utils as ku
import keras.models as km
import keras.layers as kl
import keras.regularizers as kr
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_data(image_path = '50x50flowers.images.npy', label_path = '50x50flowers.targets.npy', shuffle = True):
    """
    This functon loads image and label files, then split them into a training
    (80 %) and test (20%) set.

    Inputs:
        image_path: str, relative path to image file.
        label_path: str, relative path to targets file.
        shuffle: bool, whether to shuffle the data before splitting.

    Returns:
        Tuple of uint8 Numpy arrays: `(images_train, labels_train),
            (images_test, labels_test)`.

        **images_train, images_test**: uint8 arrays of RGB image data
            with shapes (num_samples, 50, 50, 3).

        **labels_train, labels_test**: uint8 arrays of flower labels (integers
            in range 0-16) with shapes (num_samples,).
    """

    print('Loading flowers data')
    images = np.load(image_path)

    print('Loading labels')
    labels = np.load(label_path) - 1  #fixes range

    images = images.astype(np.uint8)
    labels = labels.astype(np.uint8)
    N = images.shape[0]

    if shuffle:
        print('Shuffling data')
        rng = np.random.default_rng()
        shuffle_idx = rng.permutation(N)
        images = images[shuffle_idx]
        labels = labels[shuffle_idx]

    images_train = images[N//5:]
    labels_train = labels[N//5:]

    images_test = images[:N//5]
    labels_test = labels[:N//5]

    return (images_train, labels_train), (images_test, labels_test)

def show_image_samples(images, labels, title=''):

    """
    This functon plots the first 32 images with given labels of the dataset.

    Inputs:
        images: uint8 Numpy array, image data with shape
            (num_samples, 50, 50, 3).
        labels: uint8 Numpy array, one-hot encoded labels with shape
            (num_samples, 17).
        title: str, title of the plot.

    Returns:
        Nothing returned.
    """

    plt.figure(figsize=(12,12))
    plt.suptitle(title, fontsize=30)
    for i in range(32):
        plt.subplot(4,8,i+1)
        plt.imshow(images[i].astype(int))
        plt.title('{}'.format(np.where(labels[i])[0][0]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def get_model(numfm, numnodes, drate = 0.35, input_shape = (50, 50, 3),
              output_size = 17):

    """
    This function returns a convolution neural network Keras model,
    with numfm feature maps in the first convolutional layer, 2 *
    numfm in the second convolutional layer, 3 * numfm in the third
    convolutional layer, a dropout layer with rate drate and numnodes
    neurons in the fully-connected layer.

    Inputs:
        numfm: int, the number of feature maps in the convolution layer.

        numnodes: int, the number of nodes in the fully-connected layer.

        drate: float, dropout rate in the fully-connected layer.

        intput_shape: tuple, the shape of the input data,
            default = (50, 50, 3).

        output_size: int, the number of nodes in the output layer,
            default = 17.

    Output: the constructed Keras model.

    """

    model = km.Sequential()

    # First convolution layer with max pooling
    model.add(kl.Conv2D(numfm, kernel_size = (3, 3), input_shape = input_shape, activation = 'relu'))
    model.add(kl.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    # Second convolution layer with max pooling
    model.add(kl.Conv2D(2*numfm, kernel_size = (3, 3), activation = 'relu'))
    model.add(kl.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    # Third convolution layer with max pooling
    model.add(kl.Conv2D(3*numfm, kernel_size = (3, 3), activation = 'relu'))
    model.add(kl.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    # Flattening layer
    model.add(kl.Flatten())

    # Dense layer with dropout
    model.add(kl.Dense(numnodes, activation = 'relu'))
    model.add(kl.Dropout(drate))

    # Softmax for classification
    model.add(kl.Dense(output_size, activation = 'softmax'))

    return model

def main(image_path = '50x50flowers.images.npy', label_path = '50x50flowers.targets.npy', show_images=False, show_history=False, epochs=300):
    (images_train, labels_train), (images_test, labels_test) = load_data(image_path = image_path, label_path = label_path)

    # one-hot encode labels
    labels_train = ku.to_categorical(labels_train)
    labels_test = ku.to_categorical(labels_test)

    print('Building data augmenter')
    # Rotating and flipping flowers still look like flowers. Shift and zoom ranges is good for the small image size. Shear is a good approximation of perspective. 'reflect' fill avoids overfitting on the edge effects of other fill modes, and brightness range is reasonable for the natural lighting in the dataset.
    datagen = ImageDataGenerator(rotation_range=180, horizontal_flip=True, vertical_flip=True, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.05, shear_range=20, fill_mode='reflect', brightness_range=[0.8,1.2])
    datagen.fit(images_train)

    if show_images:
        print('Plotting sample data')
        show_image_samples(images_train, labels_train, title = 'Data')
        show_image_samples( *(next(datagen.flow(images_train, labels_train, batch_size=32))), title = 'Augmented data')

    print('Buinding network')
    model = get_model(32, 128, output_size = 17)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('Training network')
    # if interrupted, just evaluate the network
    try:
        fit = model.fit(datagen.flow(images_train, labels_train, batch_size=32),
            epochs = epochs, verbose=1)
    except:
        fit = None
        pass

    print('Evalutating network')
    train_score = model.evaluate(images_train, labels_train)
    test_score = model.evaluate(images_test, labels_test)

    print('Training score is {}'.format(train_score))
    print('Test score is {}'.format(test_score))
    print('Accuracy difference is {} %'.format(100 * (train_score[1] - test_score[1]) ) )

    # only plot if history exists
    if show_history and fit:
        plt.figure(figsize=(10,10))
        plt.suptitle('test loss {}\n test acc {}'.format(test_score[0], test_score[1]))
        plt.subplot(212)
        plt.plot(np.arange(1,len(fit.history['loss'])),fit.history['loss'][1:])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.subplot(211)
        plt.plot(np.arange(1,len(fit.history['accuracy'])),fit.history['accuracy'][1:])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Neural network for downsized flowers dataset.')
    parser.add_argument('--image-path', help='Relative path to images file')
    parser.add_argument('--label-path', help='Relative path to labels file')
    parser.add_argument('--show-images', action="store_true", help='Display sample and augmented data')
    parser.add_argument('--show-history', action="store_true", help='Display loss and accuracy history after training')
    parser.add_argument('-e', '--epochs', type=int, help='Number to epochs to train')
    args = parser.parse_args()

    main(image_path=args.image_path, label_path=args.label_path, show_images=args.show_images, show_history=args.show_history, epochs=args.epochs)
