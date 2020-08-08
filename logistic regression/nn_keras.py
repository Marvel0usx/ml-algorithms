from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
import numpy as np

X = np.array([[0, 0, 1],
              [1, 1, 1],
              [1, 0, 1],
              [0, 1, 0]])
y = np.array([[0, 1, 1, 0]]).T

model = Sequential()
model.add(Dense(1, input_dim=3, activation='sigmoid', name='my_layer'))
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# compile the model with keras
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# fit the keras model to the data
model.fit(X, y, epochs=1000, batch_size=4)
# evalute the model (on the training set)

_, accuracy = model.evaluate(X, y)
print(f"The accuracy was {accuracy * 100:.4f}")
print(model.predict(np.array([[0, 0, 1]])))
print(model.predict(np.array([[1, 1, 1]])))
print(model.predict(np.array([[1, 0, 1]])))
print(model.predict(np.array([[0, 1, 0]])))
