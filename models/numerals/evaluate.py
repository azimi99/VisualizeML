"""
    Load and Evaluate and Existing Model
"""
import numpy as np
import keras

if __name__ == "__main__":
    ## Load Model ##
    model = keras.saving.load_model("./numerals.keras", custom_objects=None, compile=True)
    ## Run Evaluation ##
    # _, (x_test, y_test) = keras.datasets.mnist.load_data()
    # x_test = x_test.astype("float32") / 255
    # y_test = keras.utils.to_categorical(y_test, 10)
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print("Test loss:", score[0])
    # print("Test accuracy:", score[1])
    model.summary()
    print(model.layers[1].get_weights()[0].shape)
    for i in range(1,5):
        weights =  model.layers[i].get_weights()[0]
        with open(f"layer_{i}", 'w') as file:
            for row in weights:
                # Convert each row to a string of comma-separated values, and add a comma at the end of the line
                line = ','.join(map(str, row)) + ',\n'
                file.write(line)
    
