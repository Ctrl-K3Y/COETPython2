import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras import models
from keras import layers
def acc_chart(results):
    plt.title("accuracy Graph")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

def loss_chart(results):
    plt.title("Loss Graph")
    plt.ylabel("Loss")
    plt.xlabel("epoch")
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

df_wine = pd.read_csv("Data/winequality-red.csv")
print(df_wine.head(5).to_string())
print(df_wine.isna().sum())

# sns.heatmap(df_wine.corr(), annot=True)
# plt.show()

# This problem is actually going to have 10 possible outpyts
# This corresponds to the possible quantities that might be produced from
# this.
X = df_wine.drop("quality", axis=1)
Y = df_wine['quality']

print("Shape of X is %s" % str(X.shape))
print("Shape of Y is %s" % str(Y.shape))


# Multi classification problem requires that our output layer
# has the same shapep as the # of the conditions we are trying to map
model = models.Sequential()
model.add(layers.Dense(11,activation='relu'))
model.add(layers.Dense(5,activation='relu'))
# model.add(layers.Dense(10,activation='sigmoid'))
model.add(layers.Dense(10, activation="softmax")) # better for this situation

model.compile(loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
history = model.fit(X, Y, epochs=150, validation_split=0.2, batch_size=100)

# All entries will be run through the system each epoch
# The batch size will indicate the sample size we are using
# for this batch size - each epoch will have 16 different sample sets run through it

loss_chart(history)
acc_chart(history)

md_wine = np.array([[7.5,0.71,0,1.8,0.76,10.88,34.2,.99,3.5,0.55,9.5]], dtype=np.float64)
y_pred = model.predict(md_wine)
print(y_pred)
print(y_pred)
max_index = np.argmax(y_pred, axis=1)
print("The max index is %d" % max_index[0])

model.save("Models/wine_sample.keras")




















