import numpy as np
import keras
from PIL import Image
model = keras.models.load_model('Models/numbers.keras')
test_image = Image.open('Images/three.png')
test_array = np.array(test_image)
print("Shape is ", test_array.shape)

test_array = test_array.reshape((1,) + test_array.shape)
print("Shape is ", test_array.shape)
print(model.predict(test_array))