#%%
# Demonstrate loading in images
from pathlib import Path
# PIL is the Python imaging livrary - install pillow to get access to it

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from keras import models

# Create a path pointing to the directory we want to load
cat_dir = Path("Images/cat/")
# Want to see what the directory we're actually pointing at?
print(cat_dir.absolute())

# Let's check how many images we have
cat_files = list(cat_dir.glob("*.png"))
image_count = len(cat_files)
print(f"We have {image_count} pictures of cats")

#%%

# Let's load in our images
cat_image = Image.open(str(cat_files[0]))
print(f"The cat image is size {cat_image.size}, its type is {type(cat_image)}")
print(f"The cat image mode is {cat_image.mode} and its format is {cat_image.format}")

plt.imshow(cat_image)
plt.show()
#%%
cat_img_array = np.array(cat_image)
print("Cat as array:")
print(cat_img_array)
print(f"The shape of the cat array is {cat_img_array.shape}")

#%%
# Can also easily go the opposite direction - from a numpy array to an image
new_cat_img = Image.fromarray(cat_img_array)
plt.imshow(new_cat_img)
plt.show()
#%%
# This is a good time examing things like the image format - like is it channel first
# or channel last? What does out ML library expect
print(f"Keras expects images to be {keras.backend.image_data_format}")
# If that wasn't what we wanted we could set it:
# keras.backend.image_data_format('channels_last')

# If our image was in the wrong shapre, we could easily use numpy's roll function
# to change the order of the dimensions
rolled_cat_img_array = np.rollaxis(cat_img_array, 2, 0)
print(rolled_cat_img_array)
print(rolled_cat_img_array.shape)
#%%
# Since this is a numpy array, we can do whatever we can do to numpy arrays on it
# mutated_cat_array = cat_img_array * np.array([0,1,0])

mutated_cat_array = np.where(cat_img_array < 100, 255, cat_img_array)
mutated_cat_array = mutated_cat_array * 0.25
mutated_cat_array = mutated_cat_array.astype(np.uint8)
mutated_cat_img = Image.fromarray(mutated_cat_array)
plt.imshow(mutated_cat_img)
plt.show()
#%%
# What if we want to save this resulting image?

# Let's first create a path for it
mutated_cat_file = Path("Images/mutated_images/cat_1_mutated.png")
print(f"I might need to make my parent directory: {mutated_cat_file.parents[0]}")
mutated_cat_file.parents[0].mkdir(exist_ok=True)
print(f"The name of my mutated_cat_file is {mutated_cat_file.name} and its absolute path is {mutated_cat_file.absolute()}")

# To actually save it, just use Image.save
mutated_cat_img.save(mutated_cat_file, "PNG")
#%%
# How can keras do image augmentation for us?
# Keras has a bunch of random layers, and we can just pass stuff through
# TensorFlow expects a 1d array of images, so it won't like just one single cat
cat_images_array = cat_img_array.reshape((1,) + cat_img_array.shape)
print(cat_images_array.shape)

# Let's add some of these keras preprocessing layers:
image_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(cat_img_array.shape)),
        layers.RandomRotation(0.1),
        layers.RandomZoon(0,1)
    ]
)
# Let's generate a few random variations
for i in range(9):
    augmented_images = image_augmentation(cat_images_array)
    plt.imshow(augmented_images[0].numpy().astype(np.uint8))
    plt.show()
    print(i)

#%%


