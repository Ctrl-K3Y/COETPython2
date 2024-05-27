import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from pathlib import Path
from keras import layers
"""
Filename: loan_classification.py
Assignment: COET 295 Assignment 2
Author: Kizza Alba (alba0877)
Instructors: Wade Lahoda, Bryce Barrie
Date: 2024-05-26
"""
def acc_chart(results):
    """Presents accuracy progression graph of the model's training results"""
    plt.title("Accuracy Graph")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.legend(["train","test"], loc="upper left")
    plt.show()

def loss_chart(results):
    """Presents validation loss graph of the model's training results"""
    plt.title("Loss Graph")
    plt.ylabel("Loss")
    plt.xlabel("epoch")
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.legend(["train","test"], loc="upper left")
    plt.show()
crop_df = pd.read_csv('../Data/Crop_Recommendation.csv',index_col=False)
copied_crop_df = crop_df.copy()
# copied_crop_df.drop(columns=['Crop'],axis=1, inplace=True)
copied_crop_df['Crop'] = copied_crop_df['Crop'].map({'Apple': 0, 'Banana': 1, 'Blackgram': 2, 'ChickPea': 3, 'Coconut': 4, 'Coffee': 5, 'Cotton': 6,
                                             'Grapes': 7, 'Jute': 8, 'KidneyBeans': 9, 'Lentil': 10, 'Maize': 11, 'Mango': 12, 'MothBeans': 13,
                                             'MungBean': 14, 'Muskmelon': 15,
                                             'Orange': 16, 'Papaya': 17, 'PigeonPeas': 18, 'Pomegranate': 19, 'Rice': 20, 'Watermelon': 21})
X = copied_crop_df.drop(["Crop"], axis=1)
Y = copied_crop_df["Crop"]

# Nitrogen,Phosphorus,Potassium,Temperature,Humidity,pH_Value,Rainfall,Crop
crop_model = keras.Sequential()
crop_model.add(layers.Dense(32, activation='relu', input_dim=7))
crop_model.add(layers.Dense(16, activation='relu'))
crop_model.add(layers.Dense(22, activation="softmax")) # better for this situation
crop_model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy,  metrics=['accuracy'])
results = crop_model.fit(X, Y, validation_split=0.22, batch_size=100, epochs=150)
acc_chart(results)
loss_chart(results)
