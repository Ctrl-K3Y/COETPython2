import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from pathlib import Path
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

def heat_map(loan_df):
    """Creates heat map for loan data"""
    plt.figure(figsize=(10, 10))
    sns.heatmap(loan_df.corr(), annot=True)
    plt.title('Heatmap of loans')
    plt.show()

def loan_histogram(loan_df):
    """Creates histograms comparing loan status against variables such as age, marital status, and education level"""
    status_approved = loan_df['loan_status'] == 1
    status_denied = loan_df['loan_status'] == 0

    plt.hist(loan_df[status_approved]['age'], color='b', alpha=0.5, bins=15, label="Approved")
    plt.hist(loan_df[status_denied]['age'], color='g', alpha=0.5, bins=15, label="Denied")
    plt.title("Loan status and Age")
    plt.legend()
    plt.show()


    plt.hist(loan_df[status_approved]['marital_status'], color='b', alpha=0.5, bins=10, label="Approved")
    plt.hist(loan_df[status_denied]['marital_status'], color='g', alpha=0.5, bins=15, label="Denied")
    plt.title("Loan status and Marital status")
    plt.legend()
    plt.show()

    plt.hist(loan_df[status_approved]['education_level'], color='b', alpha=0.5, bins=15, label="Approved")
    plt.hist(loan_df[status_denied]['education_level'], color='g', alpha=0.5, bins=15, label="Denied")
    plt.title("Loan status and Education Level")
    plt.legend()
    plt.show()

# Modifying working data:
loan_data_frame = pd.read_csv("../Data/loan.csv")

copied_loan_df = loan_data_frame.copy()
copied_loan_df.drop(["occupation"], axis=1, inplace=True)
copied_loan_df['education_level'] = copied_loan_df['education_level'].map({"High School": 0, "Bachelor's": 1, "Associate's": 1, "Master's": 2, "Doctoral": 2})
copied_loan_df['gender'] = copied_loan_df['gender'].map({"Male": 0, "Female": 1})
copied_loan_df['marital_status'] = copied_loan_df['marital_status'].map({"Single": 0, "Married": 1})
copied_loan_df['loan_status'] = copied_loan_df['loan_status'].map({'Approved': 1, "Denied": 0})
heat_map(copied_loan_df)
loan_histogram(copied_loan_df)

# Creating the model:

X = copied_loan_df.drop('loan_status', axis=1)
Y = copied_loan_df['loan_status']

loan_model = keras.Sequential()
loan_model.add(keras.layers.Dense( 13,activation='relu'))
loan_model.add(keras.layers.Dense( 6, activation='relu'))
loan_model.add(keras.layers.Dense( 1, activation='sigmoid'))
loan_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
result = loan_model.fit(X,Y, validation_split=0.33, batch_size=10, epochs=70)
acc_chart(result)
loss_chart(result)

model_path = Path("Models/loan_model.keras")
model_path.parent.mkdir(exist_ok=True, parents=True)
loan_model.save(model_path)
