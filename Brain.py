import numpy as np
from os import system
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

                        # v Number of output neurons # v Type of activation
model = Sequential([Dense(10, input_shape=(784,), activation='sigmoid')])

(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

# v Will make a scaling of the values
X_train = X_train/255
X_test = X_test/255

system('cls')                               # v Dimensions
XT_flattened = X_train.reshape(len(X_train), 784) # < Will change the array's shape
                                # ^ quantities of arrays

                                    # v Amount that a model should seek to minimize during training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(XT_flattened, Y_train, epochs=5)                                      # ^ Judgment of model performance
                                    # ^ Number of trainings

YP = model.predict(XT_flattened)
                   # ^ Will make the classification
system('cls')
                            # v Will show only the predicted numbers
print(f"Predicted number [{np.argmax(YP[2])}]\n")