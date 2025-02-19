#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing Required Libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# In[4]:


# Data Augmentation with Enhanced Techniques
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,  # Increase rotation range
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,  # Adding shear transformations
    zoom_range=0.2,   # Adding zoom range
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]  # Adjust brightness
)


# In[17]:


# Define a Custom CNN Model (Without Pre-trained Base)
model = models.Sequential(name='canNet')
    # First convolutional block
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3))),
model.add(layers.MaxPooling2D(2, 2)),
    
    # Second convolutional block
model.add(layers.Conv2D(64, (3, 3), activation='relu')),
model.add(layers.MaxPooling2D(2, 2)),
    
    # Third convolutional block
model.add(layers.Conv2D(128, (3, 3), activation='relu')),
model.add(layers.MaxPooling2D(2, 2)),
    
    # Fourth convolutional block
model.add(layers.Conv2D(256, (3, 3), activation='relu')),
model.add(layers.MaxPooling2D(2, 2)),
    
    # Flatten the output for the fully connected layers
model.add(layers.Flatten()),
    
    # Fully connected (dense) layer
model.add(layers.Dense(512, activation='relu')),
model.add(layers.Dropout(0.5)),  # Dropout to avoid overfitting
    
    # Output layer (binary classification: malignant vs. benign)
model.add(layers.Dense(2, activation='softmax'))  # Two classes: malignant and benign


# Compile the model

optimizer = Adam(learning_rate=0.0001)  # Lower learning rate
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model architecture
print(f'Model name: {model.name}')
model.summary()


# In[29]:


# Loading Training Data with Enhanced Augmentation
train_data = datagen.flow_from_directory(
    'C:/Users/manvi/Downloads/archive (1)/train',  # Replace with the correct path
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


# In[38]:


# Set up Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the Model without class weights and validate shape compatibility
try:
    history = model.fit(
        train_data, 
        epochs=20, 
        callbacks=[early_stopping],  # Early stopping to avoid overfitting
            # Adding validation data if available
    )
except ValueError as e:
    print("Error during training:", e)
    print("Debugging shapes of the first batch...")

    # Check the batch shapes to understand the mismatch
    for images, labels in train_data:
        print("Images batch shape:", images.shape)   # Expected shape: (batch_size, height, width, channels)
        print("Labels batch shape:", labels.shape)   # Expected shape for binary: (batch_size,)
        break


# In[36]:


# Test Data Augmentation
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_data = test_datagen.flow_from_directory(
    'C:/Users/manvi/Downloads/archive (1)/test',  # Replace with the correct path to test data
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_data)
print(f'Test Accuracy: {test_accuracy}')

# Get predictions
predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_labels = test_data.classes

# Classification Report
print(classification_report(true_labels, predicted_classes))

# Confusion Matrix
conf_matrix = confusion_matrix(true_labels, predicted_classes)
print('Confusion Matrix:')
print(conf_matrix)


# In[ ]:




