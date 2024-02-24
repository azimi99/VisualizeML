"""
Converts Keras Model To Tensorflow JS Model
"""

import numpy as np
import keras
import tensorflowjs as tfjs

if __name__=="__main__":
    model = keras.saving.load_model("./numerals.keras", custom_objects=None, compile=True)
    tfjs.converters.save_keras_model(model, "./tfjs-model")
    
     