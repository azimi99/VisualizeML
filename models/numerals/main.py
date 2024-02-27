import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Args():
    visualize=True
    num_classes = 10
    input_shape = (28, 28)
    
    ## Training Params
    batch_length = 128
    epochs = 30

## Visualize Dataset ##
def visualize(image, label):
    """Saves an image file visualization purposes

    Args:
        image (_type_): numpy.ndarray
        label (_type_): string
    """
    fig = plt.figure
    plt.imsave(f"{label}.png", image)
    
if __name__ == "__main__":
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    args = Args()
    if args.visualize:
        visualize(x_train[2], y_train[2])
    
    ## Normalize Images ##
    x_train = x_train.astype("float32")/255
    x_test = x_test.astype("float32")/255
    
    print(x_train.shape, x_test.shape)
    
    # One hot encode categories
    print(y_train[0:10])
    y_train = keras.utils.to_categorical(y_train, args.num_classes)
    print(y_train[0:10])
    # y_test = keras.utils.to_categorical(y_test, args.num_classes)
    
    print(y_train[0]) ## Look at sample label
    
    
    ## Create Simple Sequential Model
    
    model = keras.Sequential([
        keras.Input(shape=args.input_shape), # Input Layer
        layers.Flatten(),
        layers.Dense(8, activation="relu", use_bias=False), # Hidden Layers
        layers.Dense(8, activation="relu", use_bias=False),
        layers.Dense(8, activation="relu", use_bias=False),
        layers.Dense(args.num_classes, activation="softmax", use_bias=False) # Output Layer
    ])
    model.summary()
    
    ## Training Loop ##
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=args.batch_length, epochs=args.epochs, validation_split=0.1)
    
    
    ## Evaluate Model ##
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    
    ## Save Model ##
    keras.saving.save_model(model, "numerals.keras")
    
    