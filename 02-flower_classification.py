# load in imports
# %%
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf
import keras
from keras import layers
from keras import Sequential


#%%
# lets look at the data we're working with

data_dir = Path("Images/flower_images")
print(data_dir.absolute())

# lets see how many images there are:
image_count = len(list(data_dir.glob("*/*.jpg")))
print(f"there are {image_count} images.")

# lets grab just the roses
roses = list(data_dir.glob('roses/*'))
for i in range(5):
    print(f"Rose Image {i}:")
    my_image = Image.open(str(roses[i]))
    plt.imshow(my_image)
    plt.show()
# %%
# there are a lot of images in there, so well want to use some of the keras utilities
# to load it in. This will allow us to load entire directories of images in just a few lines
# you can also do it from scratch using tensorflows data module

# lets set soome parameters for the loader
BATCH_SIZE = 32
# our sample images are not uniform, so lets impose some constraints on height and width
IMG_HEIGHT = 180
IMG_WIDTH = 180

# well also want to have a validation split - so in this case, well try using 80%
# of the images for training, and 20% for validation
train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT,IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT,IMG_WIDTH),
    batch_size=BATCH_SIZE
)


# itll automatically use the directories as "class names" which is usually what we want
class_names = train_ds.class_names
print(class_names)

# %%
# lets visualize the first nine images from the training dataset:
plt.figure(figsize=(10,10))

# lets take 1 batch from our dataset:
for images, labels in train_ds.take(1):
    print("One image batch:")
    print(images.shape)
    print("One label batch:")
    print(labels.shape)
    print(labels)
    for i in range(9):
        ax = plt.subplot(3,3, i + 1)
        plt.imshow(images[i].numpy().astype(np.uint8))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()

# %%
# loading these images in, we do want to configure for performance
# theres lots of tweaking you can do, but two obvious ones are setting the cache size
# and doing prefetch so the data preprocessing and th emodel execution overlap
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# well also want to standardize the data
# the numbers in our RGB channels are going to be one byte - 0-255
# that isnt ideal for neural networks - typically we want small input values between
# 0 and 1
# so when we build our model, well use a rescaling layer to standardize the values
# to be in the 0 to 1 range
normalization_layer = layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# we could apply this to our data right now using dataset.map, but well just
# throw it in at the start of our model
#%%
# Let's do some data augmentation
# Data augmentation generates additional training data from existing examples
# Keras provides some pre-processing layers to make doing that easier

data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ]
)

# plt.figure(figsize=(10,10))
# for images, _ in train_ds.take(1):
#     for i in range(3):
#         for j in range(1,4):
#             augmented_images = data_augmentation(images)
#             ax = plt.subplot(3,3,1*3+j)
#             plt.imshow(augmented_images[i].numpy().astype(np.uint8))
#             plt.axis("off")

# %%
num_classes = len(class_names)

# Building the model
model = Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    data_augmentation,
    normalization_layer,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# Compile the model
model.compile(optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

print(model.summary())
# %%

# Now we can train the model:

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
#%%
# Let's visualize the results of our training, so we can make some decisions based on it:
# Grab the accuracy values:
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Grab the loss values:
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# Plot some graphs
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# Graph the Accuracy
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range,acc,label='Training Accuracy')
plt.plot(epochs_range,val_acc,label="Validation Accuracy")
plt.legend(loc="upper right")
plt.title("Training and Validation Accuracy")

# Do the same but for the Loss
plt.subplot(1,2,2)
plt.plot(epochs_range,loss,label='Training Loss')
plt.plot(epochs_range,val_loss,label="Validation loss")
plt.legend(loc="upper right")
plt.title("Traininig and validation loss")
plt.show()
# %%

model_path = Path("Images/models/flower_model.keras")
model_path.parent.mkdir(exist_ok=True, parents=True)
model.save(model_path)