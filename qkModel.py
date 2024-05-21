import keras
import numpy as np
from numpy import array

model = keras.models.load_model("Models/wine_sample.keras")
md_wine = np.array([[7.5,0.71,0,1.8,0.76,10.88,34.2,.99,3.5,0.55,9.5]], dtype=np.float64)
y_pred = model.predict(md_wine)
max_index = np.argmax(y_pred, axis=1)
print(f" The max index is {max_index}")

