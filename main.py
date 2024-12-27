# main.py
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# 1. Load VGG16 Without the Top Layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model layers

# 2. Add Custom Layers for Your Task
model = models.Sequential([
    base_model,  # Add the VGG16 base model
    layers.Flatten(),  # Flatten the output from the convolutional layers
    layers.Dense(512, activation='relu'),  # Fully connected layer (increased size)
    layers.Dropout(0.5),  # Dropout layer to prevent overfitting
    layers.Dense(3, activation='softmax')  # Output layer (3 classes: Healthy, Powdery, Rust)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Set up Data Generators for Training, Validation, and Testing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,  # Added rotation for more augmentation
    width_shift_range=0.2,  # Added width shift
    height_shift_range=0.2,  # Added height shift
    brightness_range=[0.2, 1.0]  # Adjust brightness for more variation
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/Train/Train/',  
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # Multi-class classification
)

validation_generator = validation_datagen.flow_from_directory(
    'dataset/Validation/Validation/',  
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 4. Train the Model with Early Stopping and ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=20,  # Increased number of epochs for more training time
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    callbacks=[early_stopping, reduce_lr]
)

# 5. Optionally, Fine-Tune the Model (Unfreeze Some Layers)
for layer in base_model.layers[-4:]:  # Unfreeze last few layers (adjust as necessary)
    layer.trainable = True

# Re-compile the model after unfreezing
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Continue training the model with the unfreezed layers
history_fine_tune = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,  # Additional epochs for fine-tuning
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    callbacks=[early_stopping, reduce_lr]
)

# 6. Evaluate the Model on Test Data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'dataset/Test/Test', 
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // 32)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# 7. Save the Trained Model
model.save('plant_disease_vgg16.h5')
