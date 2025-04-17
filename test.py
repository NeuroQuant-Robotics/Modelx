
# --- modelx.ipynb ---
# Deep Emotion Classifier for 26 Complex Emotions
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout,
                                     Flatten, Dense, GlobalAveragePooling2D, Multiply, Reshape, 
                                     Activation, Add, SpatialDropout2D)
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Attention Modules
class SEBlock(tf.keras.layers.Layer):
    def __init__(self, filters, reduction=8):
        super(SEBlock, self).__init__()
        self.gap = GlobalAveragePooling2D()
        self.fc1 = Dense(filters // reduction, activation='relu')
        self.fc2 = Dense(filters, activation='sigmoid')
        self.reshape = Reshape((1, 1, filters))
        self.multiply = Multiply()

    def call(self, inputs):
        x = self.gap(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshape(x)
        return self.multiply([inputs, x])

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = Conv2D(1, kernel_size=1, activation='sigmoid', padding='same')

    def call(self, inputs):
        return inputs * self.conv(inputs)

def conv_block(x, filters):
    shortcut = x
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    x = SEBlock(filters)(x)
    x = SpatialAttention()(x)
    x = Add()([x, shortcut]) if shortcut.shape[-1] == filters else x
    x = MaxPooling2D()(x)
    x = SpatialDropout2D(0.3)(x)
    return x

def build_model(input_shape=(256, 256, 3), num_classes=26):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    
    for f in [64, 128, 256]:
        x = conv_block(x, f)

    x = Flatten()(x)
    x = Dense(512, activation='swish')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

# Data
train_dir = 'Annotations/emotion_classes'

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=20,
)

train_gen = datagen.flow_from_directory(
    train_dir, target_size=(256, 256), batch_size=32,
    class_mode='categorical', subset='training'
)
val_gen = datagen.flow_from_directory(
    train_dir, target_size=(256, 256), batch_size=32,
    class_mode='categorical', subset='validation'
)

model = build_model()

# Compile
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    optimizer=AdamW(learning_rate=1e-4),
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', verbose=1)
]

# Train
history = model.fit(
    train_gen, validation_data=val_gen, epochs=50, callbacks=callbacks
)

# Graphs
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()

# Evaluation
val_gen.reset()
y_pred = np.argmax(model.predict(val_gen), axis=1)
y_true = val_gen.classes

print(classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys())))
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=val_gen.class_indices.keys(), yticklabels=val_gen.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
