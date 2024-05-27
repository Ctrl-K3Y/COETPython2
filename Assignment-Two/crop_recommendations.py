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
copied_crop_df['Crop'] = copied_crop_df['Crop'].map({'Apple': 1, 'Banana': 2, 'Blackgram': 3, 'ChickPea': 4, 'Coconut': 5, 'Coffee': 6, 'Cotton': 7,
                                             'Grapes': 8, 'Jute': 9, 'KidneyBeans': 10, 'Lentil': 11, 'Maize': 12, 'Mango': 13, 'MothBeans': 14,
                                             'MungBean': 15, 'Muskmelon': 16,
                                             'Orange': 17, 'Papaya': 18, 'PigeonPeas': 19, 'Pomegranate': 20, 'Rice': 21, 'Watermelon': 22})
X = copied_crop_df.drop(["Crop"], axis=1)
Y = copied_crop_df["Crop"]


crop_model = keras.Sequential()
crop_model.add(layers.Dense(25, activation='relu'))
crop_model.add(layers.Dropout(0.4))
crop_model.add(layers.Dense(16, activation='relu'))
crop_model.add(layers.BatchNormalization())
crop_model.add(layers.Dense(14, activation='relu'))
crop_model.compile(optimizer='adam', loss='mse',  metrics=['accuracy'])
results = crop_model.fit(X, Y, validation_split=0.33, batch_size=30, epochs=80)
acc_chart(results)
loss_chart(results)
