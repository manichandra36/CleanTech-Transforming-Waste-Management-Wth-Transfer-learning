# train_model.py  —  uses Keras 3 directly
import tensorflow as tf            # optional: still handy for tf.__version__ or GPU checks
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
import os

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 10

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.20,
    rotation_range=15,
    zoom_range=0.20,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    "dataset/",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "dataset/",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# base model: MobileNetV2 (frozen)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# custom head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(3, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# save in the modern .keras format
model.save("waste_classifier.keras")
print("✅ Model trained and saved as waste_classifier.keras")
