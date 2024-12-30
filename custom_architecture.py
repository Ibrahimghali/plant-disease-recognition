# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Input)
# from tensorflow.keras.models import Model
# import matplotlib.pyplot as plt
# import json
# import numpy as np
# from tensorflow.keras.regularizers import l2
# from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
# import seaborn as sns


# def plot_training_history(history, model_name):
#     """Creates comprehensive training history plots."""
#     plt.figure(figsize=(15, 10))

#     # Plot accuracy
#     plt.subplot(2, 2, 1)
#     plt.plot(history.history['accuracy'], label='Training', marker='o')
#     plt.plot(history.history['val_accuracy'], label='Validation', marker='o')
#     plt.title('Model Accuracy over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.grid(True)

#     # Plot loss
#     plt.subplot(2, 2, 2)
#     plt.plot(history.history['loss'], label='Training', marker='o')
#     plt.plot(history.history['val_loss'], label='Validation', marker='o')
#     plt.title('Model Loss over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.savefig(f'{model_name}_training_history.png')
#     plt.close()


# def plot_confusion_matrix(conf_matrix, class_names, model_name):
#     """Creates and saves confusion matrix visualization."""
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=class_names, yticklabels=class_names)
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     plt.savefig(f'{model_name}_confusion_matrix.png')
#     plt.close()


# def save_experiment_results(history, test_results, experiment_name, experiment_info, model, test_generator):
#     """Saves comprehensive experiment results."""
#     # Get predictions for confusion matrix and metrics
#     predictions = model.predict(test_generator)
#     y_pred = np.argmax(predictions, axis=1)
#     y_true = test_generator.classes

#     # Calculate precision, recall, f1-score
#     precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

#     # Calculate confusion matrix
#     conf_matrix = confusion_matrix(y_true, y_pred)

#     # Class names
#     class_names = list(test_generator.class_indices.keys())

#     # Create comprehensive results dictionary
#     results = {
#         "experiment_info": experiment_info,
#         "training_metrics": {
#             "best_accuracy": {
#                 "training": float(max(history.history['accuracy'])),
#                 "validation": float(max(history.history['val_accuracy'])),
#                 "test": float(test_results[1])
#             },
#             "final_loss": {
#                 "training": float(history.history['loss'][-1]),
#                 "validation": float(history.history['val_loss'][-1]),
#                 "test": float(test_results[0])
#             },
#             "final_epoch": len(history.history['accuracy']),
#             "stopped_early": len(history.history['accuracy']) < experiment_info['epochs']
#         },
#         "per_class_metrics": {
#             class_name: {
#                 "precision": float(precision[i]),
#                 "recall": float(recall[i]),
#                 "f1_score": float(f1[i])
#             } for i, class_name in enumerate(class_names)
#         },
#         "confusion_matrix": conf_matrix.tolist(),
#     }

#     # Save results to JSON
#     with open(f'{experiment_name}_results.json', 'w') as f:
#         json.dump(results, f, indent=4)

#     # Create plots
#     plot_training_history(history, experiment_name)
#     plot_confusion_matrix(conf_matrix, class_names, experiment_name)

#     return results


# def create_improved_cnn():
#     """Creates a lighter CNN architecture suitable for small datasets with strong regularization."""
#     inputs = Input(shape=(224, 224, 3))

#     # First Convolutional Block
#     x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(inputs)
#     x = BatchNormalization()(x)
#     x = MaxPooling2D((2, 2))(x)

#     # Second Convolutional Block
#     x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
#     x = BatchNormalization()(x)
#     x = MaxPooling2D((2, 2))(x)

#     # Third Convolutional Block
#     x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
#     x = BatchNormalization()(x)
#     x = MaxPooling2D((2, 2))(x)
#     x = Dropout(0.3)(x)

#     # Flatten and Dense Layers
#     x = Flatten()(x)
#     x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.4)(x)
#     outputs = Dense(3, activation='softmax')(x)

#     model = Model(inputs=inputs, outputs=outputs)
#     return model


# # Configuration
# IMG_SIZE = 224
# BATCH_SIZE = 16
# EPOCHS = 50
# INITIAL_LR = 1e-3

# # Data augmentation
# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=True,
#     zoom_range=0.15,
#     shear_range=0.15,
#     fill_mode='nearest',
#     validation_split=0.2
# )

# test_datagen = ImageDataGenerator(rescale=1. / 255)

# train_generator = train_datagen.flow_from_directory(
#     'dataset/Train/Train/',
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='training'
# )

# validation_generator = train_datagen.flow_from_directory(
#     'dataset/Validation/Validation/',
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation'
# )

# test_generator = test_datagen.flow_from_directory(
#     'dataset/Test/Test/',
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# # Model
# model = create_improved_cnn()

# # Compile
# learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=INITIAL_LR, decay_steps=1000, decay_rate=0.9
# )
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # Callbacks
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_accuracy', patience=15, restore_best_weights=True
# )
# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='val_loss', factor=0.2, patience=8, min_lr=1e-6
# )

# # Training
# history = model.fit(
#     train_generator,
#     epochs=EPOCHS,
#     validation_data=validation_generator,
#     callbacks=[early_stopping, reduce_lr],
#     verbose=1
# )

# # Evaluation
# test_results = model.evaluate(test_generator)

# # Save results
# experiment_info = {
#     'type': 'custom_cnn',
#     'model': 'CustomCNN',
#     'epochs': EPOCHS,
#     'batch_size': BATCH_SIZE,
#     'learning_rate': INITIAL_LR,
#     'trainable_parameters': int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
# }

# save_experiment_results(
#     history, test_results, 'custom_cnn', experiment_info, model, test_generator
# )

# print("\nExperiment completed! Check the generated files:")
# print("1. custom_cnn_training_history.png - Training plots")
# print("2. custom_cnn_results.json - Detailed metrics")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Input)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import json
import numpy as np
from tensorflow.keras.regularizers import l2
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns

def plot_training_history(history, model_name):
    """Creates comprehensive training history plots."""
    plt.figure(figsize=(15, 10))

    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Training', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation', marker='o')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Training', marker='o')
    plt.plot(history.history['val_loss'], label='Validation', marker='o')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.close()

def plot_confusion_matrix(conf_matrix, class_names, model_name):
    """Creates and saves confusion matrix visualization."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

def save_experiment_results(history, test_results, experiment_name, experiment_info, model, test_generator):
    """Saves comprehensive experiment results."""
    # Get predictions for confusion matrix and metrics
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # Calculate precision, recall, f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Class names
    class_names = list(test_generator.class_indices.keys())

    # Create comprehensive results dictionary
    results = {
        "experiment_info": experiment_info,
        "training_metrics": {
            "best_accuracy": {
                "training": float(max(history.history['accuracy'])),
                "validation": float(max(history.history['val_accuracy'])),
                "test": float(test_results[1])
            },
            "final_loss": {
                "training": float(history.history['loss'][-1]),
                "validation": float(history.history['val_loss'][-1]),
                "test": float(test_results[0])
            },
            "final_epoch": len(history.history['accuracy']),
            "stopped_early": len(history.history['accuracy']) < experiment_info['epochs']
        },
        "per_class_metrics": {
            class_name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i])
            } for i, class_name in enumerate(class_names)
        },
        "confusion_matrix": conf_matrix.tolist(),
    }

    # Save results to JSON
    with open(f'{experiment_name}_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Create plots
    plot_training_history(history, experiment_name)
    plot_confusion_matrix(conf_matrix, class_names, experiment_name)

    return results

def create_improved_cnn():
    """Creates a lighter CNN architecture suitable for small datasets with strong regularization."""
    inputs = Input(shape=(224, 224, 3))

    # First Convolutional Block
    x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Second Convolutional Block
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Third Convolutional Block
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # Flatten and Dense Layers
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
INITIAL_LR = 1e-3

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.15,
    shear_range=0.15,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'dataset/Train/Train/',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'dataset/Train/Train/',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    'dataset/Test/Test/',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Model
model = create_improved_cnn()

# Define optimizer with learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INITIAL_LR,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Evaluation
test_results = model.evaluate(test_generator)

# Save results
experiment_info = {
    'type': 'custom_cnn',
    'model': 'CustomCNN',
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': INITIAL_LR,
    'trainable_parameters': int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
}

save_experiment_results(
    history, test_results, 'custom_cnn', experiment_info, model, test_generator
)

print("\nExperiment completed! Check the generated files:")
print("1. custom_cnn_training_history.png - Training plots")
print("2. custom_cnn_results.json - Detailed metrics")
