# %%
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers

# %%
# empty neural network
model = Sequential()
# layer with 5 neurons and expects 7 features (age,income,sex,politics,religion(prosp,cath or other))
# relu = negative values is set to 0, with hidden layers, value can become negative
model.add(Dense(5, input_dim=7, activation="relu"))
model.add(Dense(3, activation= "softmax"))
adam = optimizers.Adam(learning_rate = 0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=adam, metrics=['accuracy'])

# %%
# Data prep. (1,0) = female and (0,1) = male
# (0,1,0) = Prespetarian, (1,0,0) = Catholic, (0,0,1) = Other
X = np.array([
    [0.27, 0.24,1,0,0,1,0],
    [0.48, 0.98,0,1,1,0,0],
    [0.33,0.44,0,1,0,0,1],
    [0.30,0.29,1,0,1,0,0],
    [0.66,0.51,0,1,0,1,0]
])
# Dem = 1,0,0 (0), Rep = 0,1,0 (1), Ind = 0,0,1 (2)
y = np.array([0, 1, 1, 2, 0])

# %%
model.fit(X,y,epochs=400, batch_size=4, verbose=1)

# %%
np.set_printoptions(suppress=True, precision=4)
model.predict(np.array([[0.38,0.51,0,1,0,1,0]]))

# %%
loss, accuracy = model.evaluate(X,y)


