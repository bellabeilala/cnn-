import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
from time import *

def data_load(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="training",
        seed=123,
        color_mode="rgb",
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=123,
        color_mode="rgb",
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    return train_ds, val_ds, class_names


def model_load(IMG_SHAPE=(224, 224, 3), class_num=2):
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train(epochs):
    begin_time = time()
    train_ds, val_ds, class_names = data_load("../data/photots_aug/", 224, 224, 4)
    print(class_names)
    model = model_load(class_num=len(class_names))
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save("models/cnn_fruits.h5")
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")  # 该循环程序运行时间： 1.4201874732


if __name__ == '__main__':
    train(epochs=10)
