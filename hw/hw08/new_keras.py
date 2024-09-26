import keras
import numpy as np
from keras import layers

my_model = keras.Sequential(
    [
        keras.Input(shape=(64,)),
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
print(keras.initializers.RandomNormal(stddev=0.01))