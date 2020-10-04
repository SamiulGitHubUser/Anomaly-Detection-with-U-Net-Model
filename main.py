from tensorflow import keras
from model import unet_model
from data import Washer_Data, get_data_path

img_size = (160, 160)
batch_size = 4
num_classes = 1

model = unet_model(img_size, num_classes)
#model.summary()

train_path, valid_path = get_data_path()

train_gen = Washer_Data(batch_size, img_size, train_path)
val_gen = Washer_Data(batch_size, img_size, valid_path)


model.compile(optimizer="adam", loss="categorical_crossentropy")

callbacks = [ keras.callbacks.ModelCheckpoint("washer_segmentation.h5", save_best_only=True) ]

# Train the model, doing validation at the end of each epoch.
epochs = 30
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)