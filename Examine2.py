import numpy as np
import keras
import cv2
from PIL import Image
import matplotlib.pyplot as plt

model = keras.models.load_model('Models/numbers.keras')
test_image = Image.open("Images/--insert image--")

plt.imshow(test_image)
plt.show(test_image)

test_array = np.array(test_image)

print("Image shape is: ", test_array.shape)
grey_image = cv2.imread("Images/--insert image--",cv2.IMREAD_GRAYSCALE)
grey_image = cv2.resize(grey_image,(28,28))
print("Resized image is ", grey_image.shape)

plt.imshow(grey_image)
plt.show()
grey_array = np.array(grey_image)
grey_array = grey_array.reshape((1,) + grey_array.shape)
grey_array = cv2.bitwise_not(grey_image)

print(model.predict(grey_array))
