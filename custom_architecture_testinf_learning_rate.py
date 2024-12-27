import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten,
                                     Dropout, BatchNormalization, Input)
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import json
import time


def create_custom_cnn():
    """Creates a custom CNN architecture for plant disease classification."""
    inputs = Input(shape=(224, 224, 3))

    # First Convolutional Block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Second Convolutional Block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Third Convolutional Block
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Flatten and Dense Layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def train_and_evaluate(learning_rate, train_generator, validation_generator, test_generator):
    """Train and evaluate model with specific learning rate."""

    # Create and compile model
    model = create_custom_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=4,
        min_lr=1e-7
    )

    # Train the model
    start_time = time.time()
    history = model.fit(
        train_generator,
        epochs=30,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    training_time = time.time() - start_time

    # Evaluate on test set
    test_results = model.evaluate(test_generator, verbose=0)

    # Get predictions for metrics
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_true, y_pred)

    return {
        'learning_rate': learning_rate,
        'training_time': training_time,
        'history': history.history,
        'test_loss': test_results[0],
        'test_accuracy': test_results[1],
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': conf_matrix.tolist(),
        'epochs_trained': len(history.history['accuracy']),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'best_train_accuracy': float(max(history.history['accuracy']))
    }


# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATES = [1e-3, 1e-4, 1e-5]  # Different learning rates to test

# Directory paths - UPDATE THESE with your actual paths
TRAIN_DIR = "C:\\Users\\SelmaB\\Desktop\\Plant_test\\train_1"
VALID_DIR = "C:\\Users\\SelmaB\\Desktop\\Plant_test\\validation_1"
TEST_DIR = "C:\\Users\\SelmaB\\Desktop\\Plant_test\\test_1"

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest'
)

valid_test_datagen = ImageDataGenerator(rescale=1. / 255)

# Create generators
print("Loading data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = valid_test_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = valid_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Train and evaluate with different learning rates
results = {}
for lr in LEARNING_RATES:
    print(f"\nTraining with learning rate: {lr}")
    results[str(lr)] = train_and_evaluate(lr, train_generator, validation_generator, test_generator)

# Save comprehensive results
with open('learning_rate_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Print comparative summary
print("\n=== Learning Rate Comparison Summary ===")
print("\nTest Accuracy Comparison:")
for lr, result in results.items():
    print(f"\nLearning Rate {lr}:")
    print(f"Test Accuracy: {result['test_accuracy']:.4f}")
    print(f"Best Validation Accuracy: {result['best_val_accuracy']:.4f}")
    print(f"Training Time: {result['training_time'] / 60:.2f} minutes")
    print(f"Epochs Trained: {result['epochs_trained']}")
    print(f"F1 Score: {result['f1_score']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")

print("\nExperiment completed! Check learning_rate_comparison_results.json for detailed results.")