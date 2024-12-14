import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical

from keras.layers import Input, Dense 
from keras.models import Model
 
is_init = False
size = -1

label = []
dictionary = {}
c = 0

for i in os.listdir():
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):
        if not is_init:
            is_init = True
            X = np.load(i)
            # Check if X is 1-dimensional and add a new axis if needed
            if X.ndim == 1:
                X = X[:, np.newaxis]
            size = X.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
        else:
            # Load the array and add a new axis if needed
            loaded_array = np.load(i)
            if loaded_array.ndim == 1:
                loaded_array = loaded_array[:, np.newaxis]
            if loaded_array.shape[1] == X.shape[1]:
                X = np.concatenate((X, loaded_array))
                y = np.concatenate((y, np.array([i.split('.')[0]] * size).reshape(-1, 1)))
            else:
                print(f"Dimension mismatch in file: {i}")

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c = c + 1


for i in range(y.shape[0]):
	y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")


y = to_categorical(y)

X_new = X.copy()
y_new = y.copy()
counter = 0 

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt: 
	X_new[counter] = X[i]
	y_new[counter] = y[i]
	counter = counter + 1


ip = Input(shape=(X.shape[1]))

m = Dense(128, activation="tanh")(ip)
m = Dense(64, activation="tanh")(m)

op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.fit(X_new, y_new, epochs=80)


model.save("model.h5")
np.save("labels.npy", np.array(label))
