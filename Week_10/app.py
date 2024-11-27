
# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train_flatten = x_train.reshape(len(x_train), 28*28)
x_test_flatten = x_test.reshape(len(x_test), 28*28)

model = keras.Sequential([
    keras.layers.Dense(10, input_shape = (784,), activation = 'sigmoid' )
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(x_train_flatten, y_train, epochs = 10)
history1 = model.fit(x_train_flatten, y_train, epochs=10)

model.evaluate(x_test_flatten, y_test)


y_predicted = model.predict(x_test_flatten)

y_predicted_labels = [np.argmax(i) for i in y_predicted]


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import tensorflow as tf
from tensorflow import keras
import seaborn as sn
from sklearn.metrics import precision_score, recall_score, f1_score

# Evaluate model_3 with more metrics
loss, accuracy = model.evaluate(x_test_flatten, y_test)
y_predicted_labels = [np.argmax(i) for i in y_predicted]

# Calculate additional metrics
precision = precision_score(y_test, y_predicted_labels, average='macro')
recall = recall_score(y_test, y_predicted_labels, average='macro')
f1 = f1_score(y_test, y_predicted_labels, average='macro')

# Print evaluation metrics
print(f"Model 3 Evaluation:")
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")


import matplotlib.pyplot as plt

# Extract loss and accuracy values from the training history
train_loss = history1.history['loss']
train_accuracy = history1.history['accuracy']

# Create the line graph
epochs = range(1, len(train_loss) + 1)  # Epochs for x-axis
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, train_accuracy, label='Training Accuracy')

# Customize the plot
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.title('Loss vs. Accuracy over Epochs')
plt.legend()  # Show legend for labels

# Display the plot
plt.show()

x_train = x_train / 255
x_test = x_test / 255

x_train_flatten = x_train.reshape(len(x_train), 28*28)
x_test_flatten = x_test.reshape(len(x_test), 28*28)

model_2 = keras.Sequential([
    keras.layers.Dense(10, input_shape = (784,), activation = 'relu' ),
    keras.layers.Dense(10, activation = 'sigmoid')
])

model_2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_2.fit(x_train_flatten, y_train, epochs=5)
history2 = model_2.fit(x_train_flatten, y_train, epochs=10)

y_predicted_2 = model_2.predict(x_test_flatten)

y_predicted_labels_2 = [np.argmax(i) for i in y_predicted_2]


cm2 = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels_2)

plt.figure(figsize = (10,7))
sn.heatmap(cm2, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import tensorflow as tf
from tensorflow import keras
import seaborn as sn
from sklearn.metrics import precision_score, recall_score, f1_score

# Evaluate model_3 with more metrics
loss, accuracy = model_2.evaluate(x_test_flatten, y_test)
y_predicted_labels_2 = [np.argmax(i) for i in y_predicted_2]

# Calculate additional metrics
precision = precision_score(y_test, y_predicted_labels_2, average='macro')
recall = recall_score(y_test, y_predicted_labels_2, average='macro')
f1 = f1_score(y_test, y_predicted_labels_2, average='macro')

# Print evaluation metrics
print(f"Model 2 Evaluation:")
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# Extract loss and accuracy values from the training history
train_loss = history2.history['loss']
train_accuracy = history2.history['accuracy']

# Create the line graph
epochs = range(1, len(train_loss) + 1)  # Epochs for x-axis
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, train_accuracy, label='Training Accuracy')

# Customize the plot
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.title('Loss vs. Accuracy over Epochs')
plt.legend()  # Show legend for labels

# Display the plot
plt.show()

model_3 = keras.Sequential([
    keras.layers.Dense(100, input_shape = (784,), activation = 'relu' ),
    keras.layers.Dense(10, activation = 'sigmoid')
])

model_3.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_3.fit(x_train_flatten, y_train, epochs=10)
history = model_3.fit(x_train_flatten, y_train, epochs=10)

y_predicted_3 = model_3.predict(x_test_flatten)

y_predicted_labels_3 = [np.argmax(i) for i in y_predicted_3]


cm3 = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels_2)

model_3.evaluate(x_test_flatten,y_test)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import tensorflow as tf
from tensorflow import keras
import seaborn as sn
from sklearn.metrics import precision_score, recall_score, f1_score

# Evaluate model_3 with more metrics
loss, accuracy = model_3.evaluate(x_test_flatten, y_test)
y_predicted_labels_3 = [np.argmax(i) for i in y_predicted_3]

# Calculate additional metrics
precision = precision_score(y_test, y_predicted_labels_3, average='macro')
recall = recall_score(y_test, y_predicted_labels_3, average='macro')
f1 = f1_score(y_test, y_predicted_labels_3, average='macro')

# Print evaluation metrics
print(f"Model 3 Evaluation:")
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

import matplotlib.pyplot as plt

# Extract loss and accuracy values from the training history
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']

# Create the line graph
epochs = range(1, len(train_loss) + 1)  # Epochs for x-axis
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, train_accuracy, label='Training Accuracy')

# Customize the plot
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.title('Loss vs. Accuracy over Epochs')
plt.legend()  # Show legend for labels

# Display the plot
plt.show()


import pickle as pkl

model_pkl_file = "Digit_classifier.pkl"
with open(model_pkl_file, 'wb') as file:
    pkl.dump(model_3, file)


from keras.models import load_model

model_3.save('Digit_classifier.h5')  # creates a HDF5 file 'my_model.h5'
# deletes the existing model
