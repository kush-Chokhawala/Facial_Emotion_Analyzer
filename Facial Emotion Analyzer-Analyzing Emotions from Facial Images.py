!ls

import tarfile

fname = 'fer2013.tar.gz'

if fname.endswith("tar.gz"):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()
elif fname.endswith("tar"):
    tar = tarfile.open(fname, "r:")
    tar.extractall()
    tar.close()

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot

from sklearn.model_selection import train_test_split

df = pd.read_csv('fer2013/fer2013.csv')

df

df.emotion.unique()

label_to_text = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

np.array(df.pixels.loc[0].split(' ')).reshape(48,48)

pyplot.imshow(np.array(df.pixels.loc[0].split(' ')).reshape(48,48).astype('float'))

# Visualize sample dataset
fig = pyplot.figure(1, (14, 14))
k = 0
for label in sorted(df.emotion.unique()):
    for j in range(3):
        px = df[df.emotion==label].pixels.iloc[k]
        px = np.array(px.split(' ')).reshape(48, 48).astype('float32')
        k += 1
        ax = pyplot.subplot(7, 7, k)
        ax.imshow(px)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label_to_text[label])
        pyplot.tight_layout()

img_array = df.pixels.apply(lambda x : np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))

img_array = np.stack(img_array, axis=0)

img_array.shape

pyplot.imshow(img_array[110])

labels = df.emotion.values

X_train, X_test, y_train, y_test = train_test_split(img_array, labels, test_size= .1)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

X_train = X_train/255
X_test = X_test/255

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

# Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

# Learning Rate Scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

# Model Checkpoint
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_accuracy',
    save_best_only=True
)

# Train the model with callbacks
history = basemodel.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=100,  # You can adjust the number of epochs
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler, model_checkpoint]
)

# Evaluate the model on the test set
test_loss, test_accuracy = basemodel.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Load the best model (model with the highest validation accuracy)
best_model = tf.keras.models.load_model('best_model.h5')

# Make predictions with the best model
y_pred = best_model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Print classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred_labels))
confusion = confusion_matrix(y_test, y_pred_labels)
print(confusion)

basemodel.summary()

# Print classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred_labels))
confusion = confusion_matrix(y_test, y_pred_labels)
print(confusion)

# Evaluate the model on the training set
train_loss, train_accuracy = basemodel.evaluate(X_train, y_train)
print(f"Training Accuracy: {train_accuracy}")

# Assuming you have already defined your model and data (X_train, y_train, X_test, y_test)

# Evaluate the model on the training data
_, train_accuracy = basemodel.evaluate(X_train, y_train)

# Evaluate the model on the test data
_, test_accuracy = basemodel.evaluate(X_test, y_test)

# Display the accuracy values
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

import matplotlib.pyplot as plt

def plot_accuracy_and_loss(history):
    # Extract accuracy and loss values from the history object
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Create subplots for accuracy and loss
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot accuracy
    ax1.plot(train_accuracy, label='Training Accuracy')
    ax1.plot(val_accuracy, label='Validation Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Plot loss
    ax2.plot(train_loss, label='Training Loss')
    ax2.plot(val_loss, label='Validation Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.show()

# Usage: Call the function and pass the training history
plot_accuracy_and_loss(history)

import matplotlib.pyplot as plt
import numpy as np

# Assuming y_pred_labels contains the predicted labels and y_test contains the true labels
# y_pred_labels = [your predicted labels]
# y_test = [your true labels]

# Plot the distribution of predicted labels
plt.figure(figsize=(10, 6))
plt.hist(y_pred_labels, bins=range(8), alpha=0.5, label='Predicted Labels')
plt.hist(y_test, bins=range(8), alpha=0.5, label='True Labels')
plt.xlabel('Emotion Class')
plt.ylabel('Count')
plt.xticks(range(7), [label_to_text[i] for i in range(7)], rotation=45)
plt.legend()
plt.title('Distribution of Predicted vs. True Labels')
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Compute the confusion matrix
confusion = confusion_matrix(y_test, y_pred_labels)

# Define class labels if you haven't already
class_labels = [label_to_text[i] for i in range(7)]

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues, values_format="d")
plt.title('Confusion Matrix')
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Assuming df contains your dataset and has an 'emotion' column
# df = your DataFrame

# Count the occurrences of each emotion class
emotion_counts = df['emotion'].value_counts()

# Map emotion class labels to text
emotion_labels = [label_to_text[i] for i in emotion_counts.index]

# Create a bar chart to visualize the label distribution
plt.figure(figsize=(10, 6))
plt.bar(emotion_labels, emotion_counts)
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.title('Label Distribution of Different Emotions')
plt.xticks(rotation=45)
plt.show()

import matplotlib.pyplot as plt

# Assuming y_train contains the training labels

# Count the occurrences of each emotion class in the training data
emotion_counts = [len(y_train[y_train == i]) for i in range(7)]

# Define class labels if you haven't already
class_labels = [label_to_text[i] for i in range(7)]

# Create a bar chart to visualize the label distribution
plt.figure(figsize=(10, 6))
plt.bar(class_labels, emotion_counts)
plt.xlabel('Emotion Class')
plt.ylabel('Count')
plt.title('Label Distribution in Training Data')
plt.xticks(rotation=45)
plt.show()

