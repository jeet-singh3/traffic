import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_category_images(folder_path, label):
    """
    Loads and processes all images in a category directory.
    Returns a tuple of (images, labels) for that category.
    """

    images = []
    labels = []

    # Loop over all files in the category directory
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        # Read and resize image
        resized_image = read_and_resize_image(image_path)

        # ignore if image is None
        if resized_image is not None:
            images.append(resized_image)
            labels.append(label)

    return images, labels


def read_and_resize_image(image_path):
    """
    Read an image from disk and resize it to the target dimensions.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray or None: Resized image array, or None if reading fails.
    """
    try:
        image = cv2.imread(image_path)  # Reads the image file into a NumPy array
        if image is not None:
            return cv2.resize(
                image, (IMG_WIDTH, IMG_HEIGHT)
            )  # Resizes the image to (width, height), returns np.ndarray
    except Exception as e:
        print(f"Could not read {image_path}: {e}")  # Error handling

    return None  # None if reading fails


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    # Loop over all category labels from 0 to range(NUM_CATEGORIES) aka NUM_CATEGORIES - 1
    for label in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(label))

        if not os.path.isdir(category_dir):
            continue

        # Load all images and corresponding labels from this category
        category_images, category_labels = load_category_images(category_dir, label)

        # Append to the master image/label lists
        images.extend(category_images)
        labels.extend(category_labels)

    return images, labels


# Brian's Lecture Model
def lecture_model():
    """
    A simple CNN model lifted from the CS50 AI handwriting lecture.
    Architecture: Conv → MaxPool → Flatten → Dense → Dropout → Output
    """
    model = tf.keras.models.Sequential()

    # Lifted from lecture
    # Convolutional layer. Learn 32 filters using a 3x3 kernel
    model.add(
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        )
    )

    # Max-pooling layer, using 2x2 pool size
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten 2D feature maps into 1D vector
    model.add(tf.keras.layers.Flatten())

    # Add a hidden layer with dropout
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))

    # Output layer with softmax over NUM_CATEGORIES classes
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))

    # Compile model with Adam optimizer as used in handwriting.py
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def model_look_twice():
    """
    Deeper Model
    Adds a second layer.
    Looking at the image twice before making a decision might help.
    """
    # Starting the same as base lecture model
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Adds a second layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Everything else is same as the base model
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def model_think_harder():
    """
    Wider Model
    Uses bigger filters and a larger dense layer.
    Thinking harder with more "neurons" might help it learn better.
    """
    # Starts differently from base: uses more filters and a larger kernel, (64, (5, 5)) > (32, (3, 3))
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            64, (5, 5), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Adds a second layer with even more filters, 128 > 64
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Larger dense layer than base model, 256 > 128
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def model_remembers_everything():
    """
    No Dropout Model
    Same as model_look_twice, but doesn’t forget anything.
    Might learn faster, but could overfit.
    """
    # Starts the same as base lecture model: first conv layer with (32, (3, 3))
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Adds a second conv layer, same as model_look_twice
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Same dense structure as base, but without dropout
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def model_try_minimum():
    """
    Tiny Model
    Very small and simple. Just one small conv and one small dense layer.
    Tests how little work the model can do and still learn.
    """
    # Smaller than base: only 16 filters (vs 32) and no pooling
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            16, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        )
    )

    # No pooling layer here

    # Smaller dense layer: 64 < 128
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def model_out_of_order():
    """
    Out of Order Model
    Dense layers come before the final conv layer.
    Not a smart setup — just testing weird stuff.
    """
    # Starts like base with first conv layer (32, (3, 3))
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        )
    )

    # Goes out of order: flatten and dense layers before final conv
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))

    # Reshape into fake "image"
    model.add(tf.keras.layers.Reshape((IMG_WIDTH - 2, IMG_HEIGHT - 2, 1)))

    # Second conv layer comes after flatten/dense, very unusual
    model.add(tf.keras.layers.Conv2D(16, (1, 1), activation="relu"))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    return lecture_model()


if __name__ == "__main__":
    main()
