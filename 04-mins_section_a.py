from pathlib import Path

import keras
import numpy as np
from keras import layers
import matplotlib.pyplot as plt
from keras.src.datasets import mnist
from PIL import Image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)

plt.imshow(X_train[12])
plt.show()

save_image = Image.fromarray(X_train[12])
save_image.save("Images/three.png")

# Normalized the data so we can work with it in a more efficient manner
X_test = X_test.astype("float32")/255.0
X_train = X_train.astype("float32")/255.0

print("X_train shape:", X_train.shape)
X_train = np.expand_dims(X_train,-1)
X_test = np.expand_dims(X_test,-1)
print("X_train shape:", X_train.shape)

num_classes=10
print(y_train[12])
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)
print(y_train[12])

input_shape = (28,28,1)
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32,kernel_size=(3,3),activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2)),
        # layers.FlattenLayer(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ]
)

batch_size = 128
epochs=15

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
history = model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs, validation_split=0.1)

#Save the model and evaluate 4 different images from the training data set
model_path = Path("Models/numbers.keras")
model_path.parent.mkdir(exist_ok=True, parents=True)
model.save(model_path)
# On Runtime
# score = model.evaluate(X_test,y_test,verbose=0)
# print("Total loss", score[0])
# print("Accuracy".score[1])

# plt.imshow(X_train[12])
# plt.show()
