"""
    Load and Evaluate and Existing Model
"""
import numpy as np
import keras

if __name__ == "__main__":
    ## Load Model ##
    model = keras.saving.load_model("./numerals.keras", custom_objects=None, compile=True)
    ## Run Evaluation ##
    _, (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255
    y_test = keras.utils.to_categorical(y_test, 10)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
