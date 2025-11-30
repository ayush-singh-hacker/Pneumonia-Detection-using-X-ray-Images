import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# --- 1. HYPERPARAMETERS ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_EPOCHS = 25
INITIAL_LR = 1e-4 # 0.0001
DATA_DIR = r'C:\Projects\Pneumonia detection\Data' # Use r'' for Windows paths

# --- 2. DATA PREPARATION AND AUGMENTATION ---

# Rescaling all images by 1/255 for normalization
# and applying data augmentation to the training set.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Only rescaling and normalizing the validation/test sets (no augmentation)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches from the directory
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary' # 'Pneumonia' vs 'Normal'
)

# Flow validation images
validation_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'val'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# --- 3. MODEL DEFINITION (A simple, effective CNN) ---
# 
model = Sequential([
    # First Conv Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    
    # Second Conv Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Third Conv Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Classifier Head
    Flatten(),
    Dropout(0.5), # Regularization to prevent overfitting
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid') # Sigmoid for binary classification
])

# --- 4. CALLBACKS for Auto-Adjustment ---

# Stop training when validation loss stops improving
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

# Reduce learning rate when validation accuracy plateaus
lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy', 
    factor=0.1, 
    patience=3, 
    min_lr=1e-6, 
    verbose=1
)

# --- 5. MODEL COMPILATION AND TRAINING ---

model.compile(
    optimizer=Adam(learning_rate=INITIAL_LR),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nStarting model training...")

history = model.fit(
    train_generator,
    epochs=MAX_EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, lr_scheduler]
)

# --- 6. SAVE MODEL ---
MODEL_SAVE_PATH = 'pneumonia_detection_model.h5'
model.save(MODEL_SAVE_PATH)
print(f"\nTraining finished. Model saved to {MODEL_SAVE_PATH}")   