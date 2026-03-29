import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# IMAGE SETTINGS
IMG_SIZE = 224
BATCH_SIZE = 8

# DATASET PATH
train_path = "dataset"
val_path = "dataset"

# DATA GENERATOR
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_gen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = train_gen.flow_from_directory(
    val_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# LOAD MOBILENET
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

# FREEZE BASE MODEL
for layer in base_model.layers:
    layer.trainable = False

# ADD CUSTOM LAYERS
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# COMPILE
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("🚀 Training Started...")

# TRAIN
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# SAVE MODEL (IMPORTANT: .h5)
model.save("model/ai_detector.h5")

print("✅ Model Saved Successfully!")