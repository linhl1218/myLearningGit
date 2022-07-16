import os
import tensorflow as tf
import numpy as np
input1 = tf.keras.Input()
dense = tf.keras.Layer.Dense(128,1)(input1)
dropout = tf.keras.Layer.Dropout(0.3)(dense)
output = tf.keras.Layer.Dense(3)(dropout)
model = tf.keras.Model(input=input1,output=output)
